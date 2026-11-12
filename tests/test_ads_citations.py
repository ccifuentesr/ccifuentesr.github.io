import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

spec = importlib.util.spec_from_file_location('ads_updater', Path(__file__).resolve().parents[1] / 'scripts/update_ads_citations.py')
ads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ads)


def payload(count=0):
    return {'response': {'docs': [{'bibcode': code, 'citation_count': count} for code in ads.BIBCODES]}}


class CitationCacheTests(unittest.TestCase):
    def test_zero_is_a_valid_count(self):
        result = ads.validate_response(payload(0))
        self.assertEqual(len(result), len(ads.BIBCODES))
        self.assertTrue(all(record['citation_count'] == 0 for record in result.values()))

    def test_incomplete_response_preserves_cache(self):
        data = payload()
        data['response']['docs'].pop()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'counts.json'
            cache.write_text('previous cache')
            with patch.object(ads, 'urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
                with self.assertRaises(ValueError):
                    ads.refresh('test-token', cache)
            self.assertEqual(cache.read_text(), 'previous cache')

    def test_network_failure_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'counts.json'
            cache.write_text('previous cache')
            with patch.object(ads, 'urlopen', side_effect=URLError('offline')):
                with self.assertRaises(URLError):
                    ads.refresh('test-token', cache)
            self.assertEqual(cache.read_text(), 'previous cache')

    def test_invalid_counts_rejected(self):
        for value in (-1, True, '4', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ads.validate_response(payload(value))

    def test_success_stores_only_public_data(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'counts.json'
            with patch.object(ads, 'urlopen', return_value=io.BytesIO(json.dumps(payload(5)).encode())):
                ads.refresh('test-token', cache)
            saved = json.loads(cache.read_text())
            self.assertEqual(len(saved['records']), len(ads.BIBCODES))
            self.assertEqual(saved['source'], 'NASA ADS')
            self.assertNotIn('test-token', cache.read_text())
            self.assertFalse(cache.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
