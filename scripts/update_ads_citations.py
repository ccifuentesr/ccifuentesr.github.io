#!/usr/bin/env python3
"""Refresh the public ADS citation cache; credentials stay in the environment."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'assets/data/ads-citations.json'
BIBCODES = (
    '2025A&A...693A.228C',
    '2020A&A...642A.115C',
    '2021RNAAS...5..129C',
    '2023PhDT.........1C',
    '2026A&A...713A.272L',
    '2025A&A...694A..15B',
    '2024A&A...692A.206C',
    '2024A&A...689A.302G',
    '2022A&A...665A.120C',
    '2021A&A...656A.162M',
    '2019ApJ...887..261M',
    '2019A&A...625A..68S',
)


def validate_response(payload):
    """Reject missing/partial/invalid results rather than replacing good counts."""
    docs = payload['response']['docs']
    records = {}
    for doc in docs:
        bibcode = doc['bibcode']
        count = doc['citation_count']
        if bibcode not in BIBCODES or bibcode in records:
            raise ValueError('Unexpected or duplicate ADS record')
        if type(count) is not int or count < 0:
            raise ValueError('Invalid citation count')
        records[bibcode] = {'citation_count': count}
    if set(records) != set(BIBCODES):
        raise ValueError('ADS returned incomplete results')
    return {bibcode: records[bibcode] for bibcode in BIBCODES}


def refresh(token, destination=CACHE):
    if not token or not token.strip():
        raise ValueError('Set ADS_API_TOKEN before running this script')
    params = urlencode({
        'q': ' OR '.join(f'bibcode:"{bibcode}"' for bibcode in BIBCODES),
        'fl': 'bibcode,citation_count',
        'rows': len(BIBCODES),
    })
    request = Request(
        'https://api.adsabs.harvard.edu/v1/search/query?' + params,
        headers={'Authorization': 'Bearer ' + token.strip(), 'Accept': 'application/json'},
    )
    with urlopen(request, timeout=30) as response:
        records = validate_response(json.load(response))
    cache = {
        'source': 'NASA ADS',
        'updated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'records': records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(cache, indent=2) + '\n', encoding='utf-8')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return cache


if __name__ == '__main__':
    try:
        result = refresh(os.environ.get('ADS_API_TOKEN', ''))
    except HTTPError as error:
        print(f'ADS returned HTTP {error.code}; previous cache kept.', file=sys.stderr)
        sys.exit(1)
    except (URLError, TimeoutError, ValueError, KeyError, TypeError, OSError):
        print('ADS update failed; check the token, connection and response. Previous cache kept.', file=sys.stderr)
        sys.exit(1)
    print(f"Updated {len(result['records'])} ADS citation counts.")
