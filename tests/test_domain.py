import unittest
from app.resources import capacity_at
from app.gate import departure_manifest
from app.yard import scan_intervals
from app.store import load_case


class YardTests(unittest.TestCase):
    def test_loan_moves_but_does_not_create_workers(self):
        case = load_case('standard')
        self.assertEqual(capacity_at(case, 40), {'sort': 1, 'load': 2})
        self.assertEqual(capacity_at(case, 65), {'sort': 2, 'load': 1})

    def test_missing_scan_is_preserved(self):
        case = load_case('edge')
        interval = scan_intervals(case['jobs'][0], 140)['sort']
        self.assertFalse(interval['complete'])
        self.assertIsNone(interval['end'])

    def test_release_deduplicates_jobs(self):
        case = load_case('standard')
        case['departures'].append(dict(case['departures'][0]))
        with self.assertRaises(ValueError):
            departure_manifest(case)
