#!/usr/bin/env python3
"""Refresh the public publication record directly from the NASA ADS API."""

import json
import os
import re
import sys
from html import unescape
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'assets/data/ads-publications.json'
AUTHOR = 'author:"Cifuentes, C." AND collection:astronomy'
QUERIES = {
    'articles': (AUTHOR + ' AND ((property:refereed AND '
                 '(bibstem:"A&A" OR bibstem:"MNRAS" OR bibstem:"AJ" OR bibstem:"ApJ" OR bibstem:"Sci" OR bibstem:"RNAAS")) '
                 'OR bibcode:2021RNAAS...5..129C)'),
    'catalogues': AUTHOR + ' AND (doctype:catalog OR doctype:dataset OR bibstem:yCat)',
    'proceedings': AUTHOR + ' AND doctype:inproceedings',
    'other': AUTHOR + ' AND (doctype:circular OR bibcode:2023PhDT.........1C)',
}
JOURNALS = {'A&A', 'MNRAS', 'AJ', 'ApJ', 'Sci', 'RNAAS'}
JOURNAL_LABELS = {'Sci': 'Science', 'RNAAS': 'Research Notes of the American Astronomical Society'}
FIELDS = 'bibcode,title,author,year,pub,pubdate,bibstem,volume,page,doi,identifier,citation_count,property,doctype'


def byline(authors):
    position = next((index + 1 for index, author in enumerate(authors)
                     if re.match(r'^Cifuentes, C(?:arlos|\.|\b)', author, re.I)), None)
    if position is None:
        raise ValueError('ADS record lacks Carlos Cifuentes in its author list')
    surnames = [author.split(',', 1)[0].strip() for author in authors]
    if position > 7:
        return position, surnames[:7] + ['et al. (incl. Cifuentes)']
    names = surnames[:position]
    if len(authors) > position:
        names.append('et al.')
    return position, names


def clean_record(doc, category='articles'):
    bibcode = doc['bibcode']
    stems = set(doc.get('bibstem', []))
    journal = next((stem for stem in JOURNALS if stem in stems), None)
    authors = doc['author']
    title = doc['title'][0]
    count = doc['citation_count']
    special_rnaas = bibcode == '2021RNAAS...5..129C'
    if (category == 'articles' and not special_rnaas and
            (journal is None or doc.get('doctype') != 'article' or
             'REFEREED' not in doc.get('property', []))) or (
            not isinstance(authors, list) or not authors or
            not isinstance(title, str) or not title or
            type(count) is not int or count < 0):
        raise ValueError(f'Invalid ADS {category} record: {bibcode}')
    position, display_authors = byline(authors)
    title = unescape(re.sub(r'</?(?:SUP|SUB)>', '', title, flags=re.I))
    title = re.sub(r'\s+([.,:;])', r'\1', title)
    identifiers = doc.get('identifier', [])
    arxiv = next((value.removeprefix('arXiv:') for value in identifiers
                  if isinstance(value, str) and value.startswith('arXiv:')), '')
    return {
        'bibcode': bibcode,
        'year': int(doc['year']),
        'pubdate': doc.get('pubdate') or str(doc['year']),
        'title': title,
        'category': category,
        'journal': JOURNAL_LABELS.get(journal, doc.get('pub', '')),
        'volume': doc.get('volume', ''),
        'page': (doc.get('page') or [''])[0],
        'byline': display_authors,
        'author_position': position,
        'arxiv': arxiv,
        'citation_count': count,
    }


def fetch_query(token, category, query):
    if not token or not token.strip():
        raise ValueError('Set ADS_API_TOKEN or ADS_DEV_KEY')
    docs = []
    total = None
    while total is None or len(docs) < total:
        params = urlencode({'q': query, 'fl': FIELDS, 'rows': 200,
                            'start': len(docs), 'sort': 'date desc, bibcode desc'})
        request = Request('https://api.adsabs.harvard.edu/v1/search/query?' + params,
                          headers={'Authorization': 'Bearer ' + token.strip(),
                                   'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)['response']
        page = payload['docs']
        found = payload['numFound']
        if (type(found) is not int or found < 0 or
                (total is not None and found != total) or
                not isinstance(page, list) or (not page and found)):
            raise ValueError('ADS returned an incomplete publication list')
        total = found
        if not found:
            break
        docs.extend(page)
        if len(docs) > total:
            raise ValueError('ADS returned too many publication records')
    return [clean_record(doc, category) for doc in docs]


def fetch(token):
    if not token or not token.strip():
        raise ValueError('Set ADS_API_TOKEN or ADS_DEV_KEY')
    records = []
    for category, query in QUERIES.items():
        records.extend(fetch_query(token, category, query))
    if len({record['bibcode'] for record in records}) != len(records):
        raise ValueError('ADS returned duplicate publication records')
    return sorted(records, key=lambda record: (record['pubdate'], record['bibcode']), reverse=True)


def refresh(token, destination=CACHE):
    records = fetch(token)
    data = {'source': 'NASA ADS',
            'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'queries': QUERIES, 'records': records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return data


if __name__ == '__main__':
    try:
        result = refresh(os.environ.get('ADS_API_TOKEN') or os.environ.get('ADS_DEV_KEY'))
    except HTTPError as error:
        print(f'ADS returned HTTP {error.code}; previous publication cache kept.', file=sys.stderr)
        sys.exit(1)
    except (URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
        print('ADS publication update failed; previous cache kept.', file=sys.stderr)
        sys.exit(1)
    print(f"Updated {len(result['records'])} ADS publications.")
