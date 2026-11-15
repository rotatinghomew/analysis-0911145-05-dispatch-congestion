import copy
import json
import math
import unittest

from app.cohort import analyse
from app.study_store import load_case


def flat_case(cohorts=(2, 3, 4)):
    """一个自造小例：入库代次固定，库龄 1 与 2，period = cohort + age。"""
    cells = []
    for cohort in cohorts:
        for age in (1, 2):
            cells.append({
                'id': 'c%d-a%d' % (cohort, age),
                'cohort': cohort,
                'age': age,
                'period': cohort + age,
                'grade': 'B',
                'offered': 100,
                'discounted': 10,
                'known_at': cohort + age,
                'heldout': False,
            })
    return {
        'kind': 'cohorts',
        'as_of_period': 10,
        'age_unit': 'week',
        'origin': '2026-08-03',
        'external_anchors': [],
        'cells': cells,
        '_snapshot': 'flat',
    }


class IdentificationTests(unittest.TestCase):
    def test_three_linear_trends_are_rank_deficient(self):
        result = analyse(load_case('standard'))
        fact = next(row for row in result['diagnostics']
                    if row['kind'] == 'trend_allocation_not_identified')
        self.assertEqual(fact['three_trend_rank'], fact['three_trend_columns'] - 1)
        self.assertEqual(fact['null_direction'], {'age': 1, 'cohort': 1, 'period': -1})

    def test_estimable_contrasts_are_pairwise_combinations(self):
        result = analyse(load_case('standard'))
        expected = {'within_cohort', 'cross_cohort_same_age', 'within_period_age'}
        self.assertEqual({row['id'] for row in result['estimable']}, expected)
        for row in result['estimable']:
            self.assertIn('std_error', row)
            self.assertIn('cluster_std_error', row)
            self.assertLessEqual(row['confidence_interval'][0], row['confidence_interval'][1])

    def test_estimates_match_alternative_centre_choice(self):
        """可估对比不随库龄/代次中心化改变。"""
        case = load_case('standard')
        first = analyse(case)
        shifted = copy.deepcopy(case)
        for cell in shifted['cells']:
            cell['age'] += 7
            cell['period'] += 7
        shifted['as_of_period'] += 7
        for anchor in shifted['external_anchors']:
            anchor['cohort_pair'] = [value + 7 for value in anchor['cohort_pair']]
        second = analyse(shifted)
        first_rows = {(row['scope'], row['id']): row['estimate'] for row in first['estimable']}
        second_rows = {(row['scope'], row['id']): row['estimate'] for row in second['estimable']}
        self.assertEqual(set(first_rows), set(second_rows))
        for key in first_rows:
            self.assertAlmostEqual(first_rows[key], second_rows[key], places=8)

    def test_ridge_style_solution_would_not_identify_trends(self):
        """秩亏设计不能用伪逆把三方趋势当作唯一解：拟合仍只报告两条可估方向。"""
        result = analyse(load_case('standard'))
        scope = next(row for row in result['scopes'] if row['scope'] == 'combined_grade_adjusted')
        self.assertEqual(scope['rank'], 4)
        self.assertEqual(scope['df'], len(scope['estimable']) * 0 + scope['rows'] - scope['rank'])


class NormalizationTests(unittest.TestCase):
    def test_every_scenario_preserves_estimable_combinations(self):
        for name in ('standard', 'edge'):
            result = analyse(load_case(name))
            for scope in result['scopes']:
                alpha = scope['readout']['data_supported_contrasts']['within_cohort']['estimate']
                gamma = scope['readout']['data_supported_contrasts']['cross_cohort_same_age']['estimate']
                for scenario in scope['scenarios']:
                    trends = {row['trend']: row['estimate'] for row in scenario['decomposition']}
                    self.assertAlmostEqual(trends['age'] + trends['period'], alpha, places=9)
                    self.assertAlmostEqual(trends['cohort'] + trends['period'], gamma, places=9)

    def test_normalizations_are_labelled_as_assumptions(self):
        result = analyse(load_case('standard'))
        normalizations = [row for row in result['constraints'] if row['kind'] == 'normalization']
        self.assertTrue(normalizations)
        for scenario in result['sensitivity']['scenarios']:
            if scenario['kind'] == 'normalization':
                self.assertFalse(scenario['data_supported'])

    def test_flip_is_reported_when_sign_depends_on_assumption(self):
        result = analyse(load_case('standard'))
        flipped = {row['trend'] for row in result['conclusions']['flips']}
        self.assertIn('period', flipped)
        for row in result['conclusions']['flips']:
            values = [value for value in row['estimates'].values() if abs(value) > 1e-12]
            self.assertGreater(len({(value > 0) - (value < 0) for value in values}), 1)


class AnchorTests(unittest.TestCase):
    def test_external_constraint_only_cites_supplied_anchor(self):
        result = analyse(load_case('standard'))
        external = [row for row in result['constraints'] if row['kind'] == 'external']
        self.assertEqual([row['anchor'] for row in external], ['pilot'])
        self.assertEqual(result['snapshot'], load_case('standard')['_snapshot'])

    def test_edge_case_has_no_external_constraints(self):
        result = analyse(load_case('edge'))
        self.assertFalse([row for row in result['constraints'] if row['kind'] == 'external'])
        self.assertFalse([row for row in result['sensitivity']['scenarios']
                          if row['kind'] == 'external'])

    def test_anchor_override_is_used_and_validated(self):
        case = load_case('standard')
        replaced = analyse(case, anchors=[{'id': 'alt', 'cohort_pair': [2, 3],
                                           'log_odds_difference': [-0.5, -0.4],
                                           'source': 'other_trial'}])
        external = [row for row in replaced['constraints'] if row['kind'] == 'external']
        self.assertEqual([row['anchor'] for row in external], ['alt'])
        self.assertTrue(any(row['id'].startswith('anchor:alt') for row in replaced['sensitivity']['scenarios']))
        with self.assertRaises(ValueError):
            analyse(case, anchors=[{'id': 'bad', 'cohort_pair': [2, 4],
                                    'log_odds_difference': [0, 1], 'source': 'x'}])
        with self.assertRaises(ValueError):
            analyse(case, anchors=[{'id': 'bad', 'cohort_pair': [2, 3],
                                    'log_odds_difference': [1, 0], 'source': 'x'}])
        empty = analyse(case, anchors=[])
        self.assertFalse([row for row in empty['constraints'] if row['kind'] == 'external'])

    def test_anchor_narrowing_is_limited_to_comparable_quantity(self):
        result = analyse(load_case('standard'))
        narrowed = result['conclusions']['anchor_narrowing']['narrowed_estimands']
        self.assertTrue(narrowed)
        for row in narrowed:
            self.assertEqual(row['estimand'], 'raw_cohort_pair_difference')
            self.assertLess(row['width_after'], row['width_before'])
        not_narrowed = result['conclusions']['anchor_narrowing']['not_narrowed']
        self.assertIn('cross_cohort_same_age_grade_adjusted', not_narrowed)
        self.assertIn('three_way_trend_allocation', not_narrowed)

    def test_anchor_scenario_declares_grade_composition(self):
        result = analyse(load_case('standard'))
        external = [row for row in result['sensitivity']['scenarios'] if row['kind'] == 'external']
        self.assertTrue(external)
        for row in external:
            self.assertIn('跨等级', row['caveat'])


class VisibilityTests(unittest.TestCase):
    def test_late_recorded_cells_are_excluded(self):
        case = load_case('standard')
        late = copy.deepcopy(case)
        for cell in late['cells']:
            if cell['id'] == 'c0a1':
                cell['known_at'] = case['as_of_period'] + 1
                cell['discounted'] = cell['offered']
        result = analyse(late)
        self.assertEqual(result['scope_definition']['late_cells_excluded'], 1)
        self.assertEqual(result['scope_definition']['visible_cells'],
                         len([cell for cell in case['cells'] if cell['known_at'] <= case['as_of_period']]) - 1)
        self.assertTrue(any(row['kind'] == 'late_recorded_excluded' for row in result['diagnostics']))
        baseline = analyse(case)
        self.assertNotEqual(result['estimable'], baseline['estimable'])

    def test_heldout_cells_never_enter_fit(self):
        case = load_case('standard')
        changed = copy.deepcopy(case)
        for cell in changed['cells']:
            if cell['heldout']:
                cell['discounted'] = 0
        result = analyse(changed)
        baseline = analyse(case)
        for left, right in zip(result['estimable'], baseline['estimable']):
            self.assertEqual(left['estimate'], right['estimate'])

    def test_validation_uses_complete_cohorts_only(self):
        result = analyse(load_case('standard'))
        holdout = [row for row in result['validation'] if row['scheme'] == 'complete_cohort']
        self.assertTrue(holdout)
        for row in holdout:
            self.assertTrue(row['all_cohorts_complete'])

    def test_partial_holdout_is_flagged_and_skipped(self):
        case = load_case('standard')
        partial = copy.deepcopy(case)
        for cell in partial['cells']:
            if cell['cohort'] == 6 and cell['age'] == 1:
                cell['heldout'] = False
        result = analyse(partial)
        flagged = next(row for row in result['diagnostics']
                       if row['kind'] == 'heldout_not_complete_cohort')
        self.assertEqual(flagged['cohorts'], [6])
        holdout = [row for row in result['validation'] if row['scheme'] == 'complete_cohort']
        for row in holdout:
            self.assertNotIn(6, row.get('heldout_cohorts', []))


class GradeScopeTests(unittest.TestCase):
    def test_grade_scopes_are_separate(self):
        result = analyse(load_case('standard'))
        scopes = {row['scope'] for row in result['scopes']}
        self.assertEqual(scopes, {'A', 'B', 'combined_grade_adjusted'})
        statements = [row['statement'] for row in result['constraints']
                      if row['id'] == 'grade_scope_separate']
        self.assertTrue(statements)

    def test_grade_nesting_is_reported(self):
        result = analyse(load_case('standard'))
        kinds = {row['kind'] for row in result['diagnostics']}
        self.assertIn('grade_nested_in_cohort', kinds)
        self.assertIn('grade_adjustment_relies_on_linear_cohort', kinds)

    def test_grade_interaction_is_tested(self):
        result = analyse(load_case('standard'))
        interaction = next(row for row in result['constraints']
                           if row['id'] == 'grade_interaction_test')
        self.assertEqual(interaction['status'], 'tested')
        self.assertTrue(0.0 <= interaction['p_value'] <= 1.0)


class OutcomeTests(unittest.TestCase):
    def test_outcome_is_share_not_revenue(self):
        result = analyse(load_case('standard'))
        self.assertIn('discounted / offered', result['scope_definition']['outcome'])
        self.assertEqual(result['scope_definition']['not_used'], '各批总销售额')
        for dimension in ('age', 'cohort', 'period'):
            for row in result['descriptive'][dimension]:
                self.assertLessEqual(row['discounted'], row['offered'])
                self.assertAlmostEqual(row['discount_share'],
                                       row['discounted'] / row['offered'], places=12)

    def test_flat_case_has_zero_slopes(self):
        result = analyse(flat_case())
        scope = result['scopes'][0]
        contrasts = scope['readout']['data_supported_contrasts']
        for name in ('within_cohort', 'cross_cohort_same_age', 'within_period_age'):
            self.assertAlmostEqual(contrasts[name]['estimate'], 0.0, places=6)

    def test_monotone_age_effect_is_recovered(self):
        case = flat_case(cohorts=range(2, 12))
        for cell in case['cells']:
            cell['discounted'] = 10 * cell['age'] + 5
        result = analyse(case)
        within = result['scopes'][0]['readout']['data_supported_contrasts']['within_cohort']
        exact = math.log(25 / 75) - math.log(15 / 85)
        self.assertAlmostEqual(within['estimate'], exact, places=4)
        self.assertTrue(within['excludes_zero'])


class SerializationTests(unittest.TestCase):
    def test_finite_numbers_and_null_instead_of_nan(self):
        for name in ('standard', 'edge'):
            result = analyse(load_case(name))
            text = json.dumps(result, allow_nan=False)
            self.assertNotIn('NaN', text)
            self.assertNotIn('Infinity', text)
            for row in result['estimable']:
                self.assertIsInstance(row['estimate'], float)
                self.assertTrue(math.isfinite(row['estimate']))

    def test_snapshot_matches_loaded_bytes(self):
        for name in ('standard', 'edge'):
            case = load_case(name)
            self.assertEqual(analyse(case)['snapshot'], case['_snapshot'])

    def test_edge_visibility_and_identifiability_are_stated(self):
        result = analyse(load_case('edge'))
        kinds = {row['kind'] for row in result['diagnostics']}
        self.assertIn('curvature_not_identified', kinds)
        self.assertIn('no_external_anchors', kinds)
        self.assertTrue(all(row['status'] in ('not_identified', 'tested')
                            for scope in result['scopes'] for row in scope['curvature']))


if __name__ == '__main__':
    unittest.main()
