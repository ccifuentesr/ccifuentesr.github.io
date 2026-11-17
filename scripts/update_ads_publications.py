#!/usr/bin/env python3
"""Refresh refereed journal articles directly from the NASA ADS API."""

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
QUERY = 'author:"Cifuentes, C." AND property:refereed AND (bibstem:"A&A" OR bibstem:"MNRAS" OR bibstem:"AJ" OR bibstem:"ApJ" OR bibstem:"Sci")'
JOURNALS = {'A&A', 'MNRAS', 'AJ', 'ApJ', 'Sci'}
JOURNAL_LABELS = {'Sci': 'Science'}
FIELDS = 'bibcode,title,author,year,pubdate,bibstem,volume,page,doi,identifier,citation_count,property,doctype'


def byline(authors):
    position = next((index + 1 for index, author in enumerate(authors)
                     if re.match(r'^Cifuentes, C(?:arlos|\.|\b)', author, re.I)), None)
    if position is None:
        raise ValueError('ADS record lacks Carlos Cifuentes in its author list')
    surnames = [author.split(',', 1)[0].strip() for author in authors]
    if position > 8:
        return position, surnames[:6] + ['et al. (incl. Cifuentes)']
    names = surnames[:position]
    if len(authors) > position:
        names.append('et al.')
    return position, names


def clean_record(doc):
    bibcode = doc['bibcode']
    stems = set(doc['bibstem'])
    journal = next((stem for stem in JOURNALS if stem in stems), None)
    authors = doc['author']
    title = doc['title'][0]
    count = doc['citation_count']
    if (journal is None or doc.get('doctype') != 'article' or
            'REFEREED' not in doc.get('property', []) or
            not isinstance(authors, list) or not authors or
            not isinstance(title, str) or not title or
            type(count) is not int or count < 0):
        raise ValueError(f'Invalid ADS journal record: {bibcode}')
    position, display_authors = byline(authors)
    title = unescape(re.sub(r'</?(?:SUP|SUB)>', '', title, flags=re.I))
    title = re.sub(r'\s+([.,:;])', r'\1', title)
    identifiers = doc.get('identifier', [])
    arxiv = next((value.removeprefix('arXiv:') for value in identifiers
                  if isinstance(value, str) and value.startswith('arXiv:')), '')
    return {
        'bibcode': bibcode,
        'year': int(doc['year']),
        'title': title,
        'journal': JOURNAL_LABELS.get(journal, journal),
        'volume': doc.get('volume', ''),
        'page': (doc.get('page') or [''])[0],
        'byline': display_authors,
        'author_position': position,
        'arxiv': arxiv,
        'citation_count': count,
    }


def fetch(token):
    if not token or not token.strip():
        raise ValueError('Set ADS_API_TOKEN or ADS_DEV_KEY')
    docs = []
    total = None
    while total is None or len(docs) < total:
        params = urlencode({'q': QUERY, 'fl': FIELDS, 'rows': 200,
                            'start': len(docs), 'sort': 'date desc, bibcode desc'})
        request = Request('https://api.adsabs.harvard.edu/v1/search/query?' + params,
                          headers={'Authorization': 'Bearer ' + token.strip(),
                                   'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)['response']
        page = payload['docs']
        found = payload['numFound']
        if (type(found) is not int or found < 1 or
                (total is not None and found != total) or
                not isinstance(page, list) or not page):
            raise ValueError('ADS returned an incomplete publication list')
        total = found
        docs.extend(page)
        if len(docs) > total:
            raise ValueError('ADS returned too many publication records')
    records = [clean_record(doc) for doc in docs]
    if len({record['bibcode'] for record in records}) != len(records):
        raise ValueError('ADS returned duplicate journal records')
    return records


def refresh(token, destination=CACHE):
    records = fetch(token)
    data = {'source': 'NASA ADS',
            'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'query': QUERY, 'records': records}
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
    print(f"Updated {len(result['records'])} ADS journal articles.")
