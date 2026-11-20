import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ads_publications', Path(__file__).resolve().parents[1] / 'scripts/update_ads_publications.py')
ads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ads)


def record(position, bibcode='2026A&A...713A.272L'):
    authors = [f'Author {index}, A.' for index in range(1, 12)]
    authors[position - 1] = 'Cifuentes, C.'
    return {'bibcode': bibcode, 'bibstem': ['A&A'], 'doctype': 'article',
            'property': ['REFEREED'], 'author': authors,
            'title': ['A journal article'], 'year': '2026', 'volume': '713',
            'page': ['A272'], 'identifier': ['arXiv:2607.19756'],
            'citation_count': 0}


class PublicationCacheTests(unittest.TestCase):
    def test_author_cutoff_and_zero_citations(self):
        seventh = ads.clean_record(record(7))
        eighth = ads.clean_record(record(8))
        self.assertEqual(seventh['byline'][-2:], ['Cifuentes', 'et al.'])
        self.assertEqual(eighth['byline'], ['Author 1', 'et al. (incl. Cifuentes)'])
        self.assertEqual(seventh['citation_count'], 0)
        self.assertEqual(seventh['arxiv'], '2607.19756')

    def test_ads_title_markup_is_plain_text(self):
        item = record(2)
        item['title'] = ['The <SUP>7</SUP>Li &amp; stellar sample']
        self.assertEqual(ads.clean_record(item)['title'], 'The 7Li & stellar sample')

    def test_incomplete_result_preserves_cache(self):
        incomplete = {'response': {'numFound': 2, 'docs': [record(2)]}}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'publications.json'
            destination.write_text('previous cache')
            with patch.object(ads, 'urlopen', return_value=io.BytesIO(json.dumps(incomplete).encode())):
                with self.assertRaises(ValueError):
                    ads.refresh('test-token', destination)
            self.assertEqual(destination.read_text(), 'previous cache')

    def test_non_journal_record_is_rejected(self):
        invalid = record(2)
        invalid['doctype'] = 'thesis'
        with self.assertRaises(ValueError):
            ads.clean_record(invalid)


if __name__ == '__main__':
    unittest.main()
