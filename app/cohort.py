"""库龄、入库代次与日历周的联合分析。

分析对象是每个时间单元的折价箱数 discounted 占待售箱数 offered 的比例，
用二项 logit 建模，不比较各批总销售额。

cohort + age = period 是资料中的确定关系，因此三个线性趋势不能分别识别。
本模块只把真正可估的方向对比当作数据结论；三方趋势分摊只出现在明确标注的
规范化或外部假设下，并单独列出会翻转的结论。
"""

import math

import numpy as np
from scipy import stats

from app.study import validate, visible_cells

CONFIDENCE = 0.95
Z_CRITICAL = 1.959963984540054
RANK_TOLERANCE = 1e-9
CURVATURE_ALPHA = 0.05


# ---------------------------------------------------------------------------
# 序列化与基础数值


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, int) or value is None or isinstance(value, str):
        return value
    return str(value)


def _rank(matrix):
    if matrix.size == 0:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0:
        return 0
    return int(np.sum(singular > singular[0] * RANK_TOLERANCE))


def _critical(df):
    if df is None or df <= 0:
        return Z_CRITICAL
    return float(stats.t.ppf(0.5 + CONFIDENCE / 2, df))


def _matrix(rows, columns):
    return np.column_stack([column(rows) for _, column in columns])


def _values(rows, field):
    return np.array([row[field] for row in rows], dtype=float)


# ---------------------------------------------------------------------------
# 二项 logit：拟二项离散度 + 按入库代次聚类


def _expit(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _fit(design, discounted, offered, clusters):
    beta = np.zeros(design.shape[1])
    converged = False
    iterations = 0
    for iterations in range(200):
        eta = np.clip(design @ beta, -30.0, 30.0)
        mu = _expit(eta)
        weight = np.maximum(offered * mu * (1.0 - mu), 1e-10)
        working = eta + (discounted - offered * mu) / weight
        rooted = np.sqrt(weight)
        updated, *_ = np.linalg.lstsq(design * rooted[:, None], working * rooted, rcond=None)
        if np.max(np.abs(updated - beta)) < 1e-11:
            beta = updated
            converged = True
            break
        beta = updated
    eta = np.clip(design @ beta, -30.0, 30.0)
    mu = _expit(eta)
    weight = np.maximum(offered * mu * (1.0 - mu), 1e-12)
    information = design.T @ (weight[:, None] * design)
    rank = _rank(information)
    bread = np.linalg.pinv(information)
    pearson = float(np.sum((discounted - offered * mu) ** 2 / weight))
    df = len(discounted) - rank
    # 拟二项离散度不小于 1：欠离散不缩小二项标准误，只放大过度离散。
    dispersion = max(pearson / df, 1.0) if df > 0 else 1.0
    deviance = float(2.0 * np.sum(
        np.where(discounted > 0, discounted * np.log(discounted / np.maximum(offered * mu, 1e-300)), 0.0)
        + np.where(offered - discounted > 0,
                   (offered - discounted) * np.log((offered - discounted) / np.maximum(offered * (1.0 - mu), 1e-300)), 0.0)
    ))
    groups = sorted(set(clusters))
    meat = np.zeros_like(information)
    for group in groups:
        index = [i for i, value in enumerate(clusters) if value == group]
        score = design[index].T @ (discounted[index] - offered[index] * mu[index])
        meat += np.outer(score, score)
    # 聚类数不多于参数个数时，聚类协方差没有意义，宁可留空也不报出接近零的假精确。
    if len(groups) > 1 and df > 0 and len(groups) > rank:
        adjust = (len(groups) / (len(groups) - 1.0)) * ((len(discounted) - 1.0) / df)
        cluster_covariance = bread @ meat @ bread * adjust * dispersion
    else:
        cluster_covariance = np.full_like(information, np.nan)
    return {
        'beta': beta,
        'bread': bread,
        'model_covariance': bread * dispersion,
        'cluster_covariance': cluster_covariance,
        'rank': rank,
        'df': df,
        'dispersion': dispersion,
        'pearson': pearson,
        'deviance': deviance,
        'mu': mu,
        'converged': converged,
        'iterations': iterations + 1,
        'clusters': len(groups),
    }


def _predict(design, beta):
    return _expit(design @ beta)


# ---------------------------------------------------------------------------
# 设计


def _centers(rows):
    scale = {}
    for field in ('age', 'cohort', 'period'):
        values = _values(rows, field)
        scale[field] = float(np.std(values)) or 1.0
    return {
        'age': float(np.mean(_values(rows, 'age'))),
        'cohort': float(np.mean(_values(rows, 'cohort'))),
        'period': float(np.mean(_values(rows, 'period'))),
        'scale': scale,
    }


def _grade_indicator(rows):
    return np.array([1.0 if row['grade'] == 'A' else 0.0 for row in rows], dtype=float)


def _base_columns(rows, centers, pooled):
    columns = [
        ('intercept', lambda data: np.ones(len(data))),
        ('age', lambda data: _values(data, 'age') - centers['age']),
        ('cohort', lambda data: _values(data, 'cohort') - centers['cohort']),
    ]
    if pooled:
        columns.append(('grade_A', lambda data: _grade_indicator(data)))
    return columns


def _fit_scope(rows, pooled):
    centers = _centers(rows)
    columns = _base_columns(rows, centers, pooled)
    design = _matrix(rows, columns)
    fit = _fit(design, _values(rows, 'discounted'), _values(rows, 'offered'), [row['cohort'] for row in rows])
    fit.update({
        'design': design,
        'names': [name for name, _ in columns],
        'centers': centers,
        'rows': rows,
        'discounted': _values(rows, 'discounted'),
        'offered': _values(rows, 'offered'),
    })
    return fit


def _contrast(fit, weights, identifier, description):
    weights = np.asarray(weights, dtype=float)
    estimate = float(weights @ fit['beta'])
    variance = float(weights @ fit['model_covariance'] @ weights)
    standard_error = math.sqrt(max(variance, 0.0))
    if np.all(np.isfinite(fit['cluster_covariance'])):
        cluster_variance = float(weights @ fit['cluster_covariance'] @ weights)
        cluster_se = math.sqrt(max(cluster_variance, 0.0))
    else:
        cluster_se = None
    critical = _critical(fit['df'])
    record = {
        'id': identifier,
        'contrast': description,
        'estimate': estimate,
        'std_error': standard_error,
        'confidence_interval': [estimate - critical * standard_error, estimate + critical * standard_error],
        'cluster_std_error': cluster_se,
        'cluster_confidence_interval': (
            [estimate - critical * cluster_se, estimate + critical * cluster_se] if cluster_se is not None else None),
        'cluster_interval_available': cluster_se is not None,
        'cluster_interval_note': (None if cluster_se is not None else
                                  '入库代次数不多于参数个数，聚类协方差不可靠，明确留空而不是报出假精确。'),
        'df': fit['df'],
        'dispersion': fit['dispersion'],
        'uncertainty': ('二项标准误乘拟二项离散度；聚类标准误按入库代次聚类。'
                        '同一箱可在不同时点再次待售，两种标准误都可能偏乐观。'),
    }
    record['excludes_zero'] = bool(
        record['confidence_interval'][0] > 0 or record['confidence_interval'][1] < 0)
    return record


def _estimable(fit, pooled):
    names = fit['names']
    age_index = names.index('age')
    cohort_index = names.index('cohort')
    alpha = np.zeros(len(names))
    alpha[age_index] = 1.0
    gamma = np.zeros(len(names))
    gamma[cohort_index] = 1.0
    records = [
        _contrast(fit, alpha, 'within_cohort',
                  '同一入库代次库龄增加一周（日历同时前进一周）的 log-odds 变化；合并了库龄与日历效应。'),
        _contrast(fit, gamma, 'cross_cohort_same_age',
                  '同一库龄下较晚代次相对较早代次（日历同时前进一周）的 log-odds 变化；合并了代次与日历效应。'),
        _contrast(fit, alpha - gamma, 'within_period_age',
                  '同一日历周内较大库龄（较早代次）相对较小库龄的 log-odds 变化；合并了库龄与代次效应。'),
    ]
    return records, alpha, gamma


def _curvature(fit, rows, dimension):
    levels = sorted({row[dimension] for row in rows})
    if len(levels) < 3:
        return {'dimension': dimension, 'status': 'not_identified', 'levels': len(levels),
                'reason': ('该方向只有 ' + str(len(levels)) +
                           ' 个不同取值，二次项与线性项完全共线，曲率不可识别。')}
    base = fit['design']
    centered = (_values(rows, dimension) - fit['centers'][dimension]) / fit['centers']['scale'][dimension]
    candidate = centered ** 2
    residual = candidate - base @ np.linalg.lstsq(base, candidate, rcond=None)[0]
    norm = float(np.linalg.norm(residual))
    scale = float(np.linalg.norm(candidate - np.mean(candidate)))
    if norm <= 1e-7 * max(scale, 1e-12):
        return {'dimension': dimension, 'status': 'not_identified', 'levels': len(levels),
                'reason': '该方向的二次项与现有线性项共线，没有额外的非线性自由度。'}
    design = np.column_stack([base, residual / norm])
    trial = _fit(design, fit['discounted'], fit['offered'], [row['cohort'] for row in rows])
    if trial['rank'] < design.shape[1]:
        return {'dimension': dimension, 'status': 'not_identified', 'levels': len(levels),
                'reason': '加入曲率列后设计仍秩亏，不能把该系数归给单一方向。'}
    difference = max(fit['deviance'] - trial['deviance'], 0.0)
    f_value = difference / max(fit['dispersion'], 1e-12)
    p_value = float(1.0 - stats.f.cdf(f_value, 1, max(fit['df'], 1)))
    coefficient_se = math.sqrt(max(float(trial['bread'][-1, -1]) * trial['dispersion'], 0.0))
    return {
        'dimension': dimension,
        'status': 'tested',
        'coefficient': float(trial['beta'][-1]),
        'std_error': coefficient_se,
        'deviance_difference': float(difference),
        'F': float(f_value),
        'p_value': p_value,
        'supported_in_sample': bool(p_value < CURVATURE_ALPHA),
        'note': '在已控制线性库龄与代次后，检验该方向额外的曲率；拟二项离散度已缩放。',
    }


# ---------------------------------------------------------------------------
# 外部锚点


def _grade_by_cohort(case):
    grades = {}
    for cell in case['cells']:
        grades.setdefault(cell['cohort'], set()).add(cell['grade'])
    return grades


def _anchor_records(case, anchors):
    supplied = case.get('external_anchors', []) if anchors is None else anchors
    if not isinstance(supplied, list):
        raise ValueError('anchors 必须是列表')
    grades = _grade_by_cohort(case)
    records = []
    for raw in supplied:
        if not isinstance(raw, dict):
            if anchors is not None:
                raise ValueError('锚点必须是对象')
            records.append({'id': None, 'usable': False, 'reason': '锚点不是对象。'})
            continue
        identifier = raw.get('id')
        pair = raw.get('cohort_pair')
        interval = raw.get('log_odds_difference')
        record = {
            'id': identifier,
            'source': raw.get('source'),
            'cohort_pair': pair,
            'interval': interval,
            'usable': False,
            'reason': None,
        }
        problem = None
        if not isinstance(pair, list) or len(pair) != 2 or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in pair):
            problem = '代次对必须是两个整数。'
        elif pair[0] == pair[1]:
            problem = '代次对相同，差异不可识别。'
        elif abs(pair[0] - pair[1]) != 1:
            problem = '锚点跨越不止一个入库周，无法直接对应单一线性斜率。'
        elif not isinstance(interval, list) or len(interval) != 2 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in interval):
            problem = '差异区间必须是两个有限数值。'
        elif interval[0] > interval[1]:
            problem = '差异区间下界大于上界。'
        if problem:
            if anchors is not None:
                raise ValueError('锚点 ' + str(identifier) + ' 不可用：' + problem)
            record['reason'] = problem
            records.append(record)
            continue
        record['usable'] = True
        record['crosses_grade'] = grades.get(pair[0], set()) != grades.get(pair[1], set())
        record['commensurability'] = (
            'includes_grade_composition' if record['crosses_grade'] else 'grade_internal')
        if record['crosses_grade']:
            record['reason'] = ('该代次对跨越不同等级，锚点差异含等级构成；'
                                '不能直接对应等级调整后的代次斜率。')
        records.append(record)
    return records


def _anchor_check(case, anchor, rows):
    """把锚点与同一代次对的原始差异比较，并说明收窄了什么、没收敛什么。

    锚点描述的是代次对之间未控制等级的 log-odds 差异，因此只与同口径的
    原始差异可比。等级调整后的代次斜率是另一个量，锚点不能直接收窄它。
    """
    pair = anchor['cohort_pair']
    subset = [row for row in rows if row['cohort'] in pair]
    if len({row['cohort'] for row in subset}) < 2:
        return {'anchor': anchor['id'], 'comparable': False,
                'reason': '数据中没有同时包含这两个代次的可见单元。'}
    ages = _values(subset, 'age')
    indicator = np.array([1.0 if row['cohort'] == pair[1] else 0.0 for row in subset])
    design = np.column_stack([np.ones(len(subset)), ages - float(np.mean(ages)), indicator])
    fit = _fit(design, _values(subset, 'discounted'), _values(subset, 'offered'),
               [row['cohort'] for row in subset])
    index = 2
    estimate = float(fit['beta'][index])
    standard_error = math.sqrt(max(float(fit['model_covariance'][index, index]), 0.0))
    critical = _critical(max(fit['df'], 1))
    interval = [estimate - critical * standard_error, estimate + critical * standard_error]
    low, high = anchor['interval']
    overlap = [max(interval[0], low), min(interval[1], high)]
    consistent = overlap[0] <= overlap[1]
    return {
        'anchor': anchor['id'],
        'cohort_pair': pair,
        'estimand': 'raw_cohort_pair_log_odds_difference_adjusted_for_age_only',
        'observed_difference': estimate,
        'std_error': standard_error,
        'confidence_interval': interval,
        'anchor_interval': [low, high],
        'overlap': overlap if consistent else None,
        'consistent': bool(consistent),
        'width_before': interval[1] - interval[0],
        'width_after': (overlap[1] - overlap[0]) if consistent else None,
        'narrowed': bool(consistent and (overlap[1] - overlap[0]) < (interval[1] - interval[0]) - 1e-12),
        'narrows': ('raw_cohort_pair_difference' if consistent and
                    (overlap[1] - overlap[0]) < (interval[1] - interval[0]) - 1e-12 else None),
        'does_not_narrow': [
            'cross_cohort_same_age_grade_adjusted',
            'within_cohort',
            'within_period_age',
            'three_way_trend_allocation',
        ],
        'reason': ('锚点与同代次对、仅控制库龄的原始差异同口径，可以收窄这一项；'
                   '等级调整后的代次斜率、库龄内变化与三方分摊都不因此收窄。'),
    }


# ---------------------------------------------------------------------------
# 三方趋势分摊：规范化与外部假设


def _variance(covariance, coefficients):
    return float(np.asarray(coefficients) @ covariance @ np.asarray(coefficients))


def _trend_records(alpha, gamma, covariance, fixed_trend, fixed_value):
    """在一个规范化下给出 (A, C, P)。

    固定方向记为假设值（SE 为 0）；另外两个方向由可估的 alpha = A + P
    与 gamma = C + P 解出，并保留其标准误。
    """
    if fixed_trend == 'cohort':
        table = {'age': alpha - gamma + fixed_value, 'cohort': fixed_value,
                 'period': gamma - fixed_value}
        coefficients = {'age': (1.0, -1.0), 'cohort': (0.0, 0.0), 'period': (0.0, 1.0)}
    elif fixed_trend == 'period':
        table = {'age': alpha, 'cohort': gamma, 'period': fixed_value}
        coefficients = {'age': (1.0, 0.0), 'cohort': (0.0, 1.0), 'period': (0.0, 0.0)}
    elif fixed_trend == 'age':
        table = {'age': fixed_value, 'cohort': gamma - alpha, 'period': alpha}
        coefficients = {'age': (0.0, 0.0), 'cohort': (-1.0, 1.0), 'period': (1.0, 0.0)}
    else:
        raise ValueError('未知的规范化方向')
    records = []
    for name in ('age', 'cohort', 'period'):
        assumed = name == fixed_trend
        records.append({
            'trend': name,
            'estimate': float(table[name]),
            'std_error': 0.0 if assumed else math.sqrt(max(
                _variance(covariance, coefficients[name]), 0.0)),
            'assumed': bool(assumed),
        })
    return records


def _scenarios(alpha, gamma, covariance, anchors):
    scenarios = []
    for identifier, statement, fixed_trend, fixed_value in (
            ('cohort_flat', 'C = 0：假设入库代次本身没有线性趋势', 'cohort', 0.0),
            ('period_flat', 'P = 0：假设日历周本身没有线性趋势', 'period', 0.0),
            ('age_flat', 'A = 0：假设库龄本身没有线性趋势', 'age', 0.0)):
        scenarios.append({
            'id': identifier,
            'kind': 'normalization',
            'fixed_trend': fixed_trend,
            'constraint': statement,
            'decomposition': _trend_records(alpha, gamma, covariance, fixed_trend, fixed_value),
            'data_supported': False,
            'interpretation': '规范化只是把不可识别的三方趋势固定下来；换一条等价规范化会得到不同分摊。',
        })
    for anchor in anchors:
        if not anchor.get('usable'):
            continue
        low, high = anchor['interval']
        for suffix, fixed_value in (('low', low), ('mid', (low + high) / 2.0), ('high', high)):
            scenarios.append({
                'id': 'anchor:' + str(anchor['id']) + ':' + suffix,
                'kind': 'external',
                'anchor': anchor['id'],
                'fixed_trend': 'cohort',
                'commensurability': anchor.get('commensurability'),
                'constraint': ('外部锚点把代次趋势 C 取为 ' + format(fixed_value, '.6g') +
                               '；锚点区间说明这不是精确真值。'),
                'decomposition': _trend_records(alpha, gamma, covariance, 'cohort', fixed_value),
                'data_supported': False,
                'interpretation': '这是外部假设带来的分摊，不是本地数据识别的结果。',
                'caveat': ('该代次对跨等级，把锚点差异当作 C 会同时吸收等级构成差异；'
                           '因此这个分摊依赖“等级差异可忽略或已被线性代次趋势吸收”的额外假设。'
                           if anchor.get('crosses_grade') else
                           '锚点与代次趋势同等级口径；分摊仍只是外部假设，不是数据识别。'),
            })
    return scenarios


def _same_estimand(favourable):
    estimates = [row['estimate'] for row in favourable.values() if row]
    return bool(len(estimates) == 2 and abs(estimates[0] - estimates[1]) < 1e-9)


def _readout(alpha, gamma, records, scenarios, curvature, validation):
    by_id = {record['id']: record for record in records}
    def interval(identifier):
        record = by_id[identifier]
        return {
            'estimate': record['estimate'],
            'interval': record['confidence_interval'],
            'cluster_interval': record['cluster_confidence_interval'],
            'cluster_interval_available': record['cluster_interval_available'],
            'excludes_zero': record['excludes_zero'],
        }
    stories = []
    for scenario in scenarios:
        by_trend = {row['trend']: row for row in scenario['decomposition']}
        cohort = by_trend['cohort']['estimate']
        period = by_trend['period']['estimate']
        stories.append({
            'scenario': scenario['id'],
            'kind': scenario['kind'],
            'newer_cohorts_more_durable': bool(cohort < 0),
            'calendar_explains_lower_discount': bool(period < 0),
            'both': bool(cohort < 0 and period < 0),
        })
    manager = [row['scenario'] for row in stories if row['newer_cohorts_more_durable']]
    sales = [row['scenario'] for row in stories if row['calendar_explains_lower_discount']]
    both = [row['scenario'] for row in stories if row['both']]
    # 每个说法最有利的规范化：仓储要 C<0（取 P=0）；销售要 P<0（取 C=0）。
    # 两者各自落在最有利规范化下，量仍然是同一条可估的 gamma，符号自然相同。
    favourable = {}
    for story, identifier, trend in (
            ('manager_newer_cohorts_more_durable', 'period_flat', 'cohort'),
            ('sales_calendar_promotion', 'cohort_flat', 'period')):
        scenario = next((row for row in scenarios if row['id'] == identifier), None)
        if scenario is None:
            favourable[story] = None
            continue
        row = next(item for item in scenario['decomposition'] if item['trend'] == trend)
        critical = _critical(by_id['cross_cohort_same_age']['df'])
        supported = bool(abs(row['estimate']) > critical * row['std_error'])
        favourable[story] = {
            'normalization': identifier,
            'estimate': row['estimate'],
            'std_error': row['std_error'],
            'sign_matches_claim': bool((trend == 'cohort' and row['estimate'] < 0) or
                                       (trend == 'period' and row['estimate'] < 0)),
            'supported_under_favourable_normalization': supported,
            'detail': ('最有利的规范化把该方向恰好等同于可估的 cross_cohort 对比；'
                       '即使如此，估计为 ' + format(row['estimate'], '.4f') +
                       '，标准误 ' + format(row['std_error'], '.4f') + '。'),
        }
    # 两种说法同时成立意味着 C<0 且 P<0，于是 gamma = C+P < 0；gamma 可估，可直接检验。
    joint = interval('cross_cohort_same_age')
    joint.update({
        'requirement': '仓储负责人(代次 C<0) 与销售(日历 P<0) 同时成立，必须 gamma = C+P < 0。',
        'supported': bool(joint['excludes_zero'] and joint['estimate'] < 0),
        'interpretation': ('联合说法对应可估的 cross_cohort 对比；若该区间不排除零或方向相反，'
                           '就不能同时接受两种解释。单个说法本身仍不可由数据分摊。'),
    })
    curvature_supported = [row['dimension'] for row in curvature if row.get('supported_in_sample')]
    out_of_sample = [row['scope'] for row in validation
                     if row.get('curvature_supported_out_of_sample')]
    return {
        'data_supported_contrasts': {
            'within_cohort': interval('within_cohort'),
            'cross_cohort_same_age': interval('cross_cohort_same_age'),
            'within_period_age': interval('within_period_age'),
        },
        'joint_claim': joint,
        'curvature': {
            'supported_in_sample': curvature_supported,
            'supported_out_of_sample': out_of_sample,
            'detail': '样本内显著才列为候选曲率；只有在完整代次留出后仍改善才称为数据支持。',
        },
        'stories': {
            'manager_newer_cohorts_more_durable': manager,
            'sales_calendar_promotion': sales,
            'both_stories_true': both,
            'mutually_exclusive': not both,
            'under_most_favourable_normalization': favourable,
            'claims_share_same_estimand': _same_estimand(favourable),
            'shared_estimand_note': ('两种说法各自最有利的规范化都落到同一条可估对比 gamma = C+P；'
                                     '因此数据既不能支持其中一个而不支持另一个，也不能把它们分开。'),
            'detail': ('仓储负责人需要代次趋势 C<0，销售需要日历趋势 P<0。'
                       '两者能否同时成立只由规范化或锚点决定；可估数据只裁决它们的联合含义 gamma。'),
        },
        'not_allocable': {
            'detail': ('库龄、代次、日历各自的线性趋势不能由数据分摊；'
                       '只有 within_cohort、cross_cohort_same_age、within_period_age 可估。'),
            'null_direction': {'age': 1, 'cohort': 1, 'period': -1},
        },
    }


# ---------------------------------------------------------------------------
# 验证：完整代次留出与逐代次轮换


def _heldout_groups(case, heldout_rows):
    all_by_cohort = {}
    for cell in case['cells']:
        all_by_cohort.setdefault(cell['cohort'], []).append(cell)
    heldout_cohorts = sorted({row['cohort'] for row in heldout_rows})
    records = {}
    usable = []
    for cohort in heldout_cohorts:
        cells = all_by_cohort.get(cohort, [])
        complete_heldout = all(cell['heldout'] for cell in cells)
        complete_visible = all(cell['known_at'] <= case['as_of_period'] for cell in cells)
        records[str(cohort)] = {
            'cells': len(cells),
            'all_heldout': complete_heldout,
            'all_visible': complete_visible,
        }
        if complete_heldout and complete_visible:
            usable.append(cohort)
    return heldout_cohorts, records, usable


def _binomial_deviance(observed, offered, predicted):
    predicted = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    expected = offered * predicted
    expected_miss = offered * (1.0 - predicted)
    hit = np.where(observed > 0, observed * np.log(np.where(observed > 0, observed / expected, 1.0)), 0.0)
    miss = np.where(offered - observed > 0,
                    (offered - observed) * np.log(np.where(offered - observed > 0,
                                                           (offered - observed) / expected_miss, 1.0)), 0.0)
    return float(2.0 * np.sum(hit + miss))


def _design_with_terms(rows, centers, pooled, terms):
    columns = _base_columns(rows, centers, pooled)
    for term, field in (('age_quadratic', 'age'), ('cohort_quadratic', 'cohort'),
                        ('period_quadratic', 'period')):
        if term in terms:
            columns.append((term, lambda data, field=field: (
                (_values(data, field) - centers[field]) / centers['scale'][field]) ** 2))
    return _matrix(rows, columns)


def _quadratic_fit(rows, pooled):
    fit = _fit_scope(rows, pooled)
    design = fit['design']
    terms = []
    for term, field in (('age_quadratic', 'age'), ('cohort_quadratic', 'cohort'),
                        ('period_quadratic', 'period')):
        candidate = ((_values(rows, field) - fit['centers'][field]) / fit['centers']['scale'][field]) ** 2
        if _rank(np.column_stack([design, candidate])) > _rank(design):
            design = np.column_stack([design, candidate])
            terms.append(term)
    trial = _fit(design, fit['discounted'], fit['offered'], [row['cohort'] for row in rows])
    trial.update({'names': fit['names'] + terms, 'centers': fit['centers'],
                  'terms': terms, 'base': fit, 'design': design})
    return trial


def _cohort_holdout(case, rows, heldout_rows, pooled, label):
    cohorts, records, usable = _heldout_groups(case, heldout_rows)
    test = [row for row in heldout_rows if row['cohort'] in usable]
    excluded = [cohort for cohort in cohorts if cohort not in usable]
    if not test:
        return {'scope': label, 'scheme': 'complete_cohort', 'heldout_cohorts': [],
                'excluded_cohorts': excluded,
                'cohort_complete': records, 'all_cohorts_complete': False,
                'comparisons': [], 'note': '没有完整且可见的留出代次，无法做代次留出验证。'}
    linear = _fit_scope(rows, pooled)
    quadratic = _quadratic_fit(rows, pooled)
    specs = [('linear_apc', [], linear), ('quadratic_apc', quadratic['terms'], quadratic)]
    if pooled:
        age_only = _fit_scope(rows, False)
        specs.append(('linear_no_grade_term', [], age_only))
    comparisons = []
    for name, terms, fit in specs:
        design = _design_with_terms(test, linear['centers'], pooled and name != 'linear_no_grade_term',
                                     terms if name == 'quadratic_apc' else [])
        if design.shape[1] != fit['beta'].size:
            continue
        predicted = _predict(design, fit['beta'])
        observed = _values(test, 'discounted')
        offered = _values(test, 'offered')
        comparisons.append({
            'specification': name,
            'terms': list(terms),
            'heldout_deviance': _binomial_deviance(observed, offered, predicted),
            'heldout_pearson': float(np.sum((observed - offered * predicted) ** 2
                                            / np.maximum(offered * predicted * (1.0 - predicted), 1e-12))),
            'predicted_discounted': float(np.sum(offered * predicted)),
            'observed_discounted': float(np.sum(observed)),
            'train_deviance': float(fit['deviance']),
            'train_df': int(fit['df']),
        })
    best = min(comparisons, key=lambda row: row['heldout_deviance']) if comparisons else None
    linear_row = next((row for row in comparisons if row['specification'] == 'linear_apc'), None)
    curvature_supported = bool(
        best and best['specification'] == 'quadratic_apc' and linear_row
        and best['heldout_deviance'] + 1.0 < linear_row['heldout_deviance'])
    return {
        'scope': label,
        'scheme': 'complete_cohort',
        'heldout_cohorts': [cohort for cohort in cohorts if cohort in usable],
        'excluded_cohorts': excluded,
        'cohort_complete': records,
        'all_cohorts_complete': all(record['all_heldout'] and record['all_visible']
                                    for record in records.values()),
        'comparisons': comparisons,
        'best_specification': best['specification'] if best else None,
        'curvature_supported_out_of_sample': curvature_supported,
        'note': ('每个留出代次的全部单元都排除在拟合之外；曲率只有在留出后仍改善才算数据支持。'),
    }


def _leave_one_cohort_out(rows, pooled, label):
    cohorts = sorted({row['cohort'] for row in rows})
    if len(cohorts) < 3 or len(rows) < 6:
        return {'scope': label, 'scheme': 'leave_one_cohort_out', 'folds': [],
                'note': '可估代次太少，无法留出完整代次做轮换验证。'}
    folds = []
    for cohort in cohorts:
        train = [row for row in rows if row['cohort'] != cohort]
        test = [row for row in rows if row['cohort'] == cohort]
        linear = _fit_scope(train, pooled)
        quadratic = _quadratic_fit(train, pooled)
        design_linear = _design_with_terms(test, linear['centers'], pooled, [])
        design_quadratic = _design_with_terms(test, quadratic['centers'], pooled, quadratic['terms'])
        observed = _values(test, 'discounted')
        offered = _values(test, 'offered')
        predicted_linear = _predict(design_linear, linear['beta'])
        predicted_quadratic = _predict(design_quadratic, quadratic['beta'])
        folds.append({
            'cohort': cohort,
            'cells': len(test),
            'linear_deviance': _binomial_deviance(observed, offered, predicted_linear),
            'quadratic_deviance': _binomial_deviance(observed, offered, predicted_quadratic),
            'linear_expected_discounted': float(np.sum(offered * predicted_linear)),
            'quadratic_expected_discounted': float(np.sum(offered * predicted_quadratic)),
            'observed_discounted': float(np.sum(observed)),
        })
    linear_total = sum(fold['linear_deviance'] for fold in folds)
    quadratic_total = sum(fold['quadratic_deviance'] for fold in folds)
    return {
        'scope': label,
        'scheme': 'leave_one_cohort_out',
        'folds': folds,
        'linear_deviance': linear_total,
        'quadratic_deviance': quadratic_total,
        'curvature_supported_out_of_sample': bool(quadratic_total + 1.0 < linear_total),
        'note': '每个折都留出完整入库代次；曲率只在轮换后仍改善才算数据支持。',
    }


# ---------------------------------------------------------------------------
# 描述统计


def _wilson(discounted, offered):
    if offered == 0:
        return [None, None]
    proportion = discounted / offered
    z = Z_CRITICAL
    denominator = 1.0 + z * z / offered
    center = (proportion + z * z / (2.0 * offered)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / offered + z * z / (4.0 * offered * offered)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _descriptive(rows):
    output = {}
    for dimension in ('age', 'cohort', 'period'):
        groups = {}
        for row in rows:
            entry = groups.setdefault(row[dimension], {'offered': 0, 'discounted': 0, 'cells': 0})
            entry['offered'] += row['offered']
            entry['discounted'] += row['discounted']
            entry['cells'] += 1
        output[dimension] = []
        for level in sorted(groups):
            entry = groups[level]
            output[dimension].append({
                'level': level,
                'cells': entry['cells'],
                'offered': entry['offered'],
                'discounted': entry['discounted'],
                'discount_share': entry['discounted'] / entry['offered'] if entry['offered'] else None,
                'wilson_interval': _wilson(entry['discounted'], entry['offered']),
            })
    return output


# ---------------------------------------------------------------------------
# 等级交互


def _grade_interaction(rows, pooled):
    grades = sorted({row['grade'] for row in rows})
    if len(grades) < 2:
        return {'status': 'not_applicable', 'detail': '只有一个等级，无需检验等级交互。'}
    base = pooled['design']
    indicator = _grade_indicator(rows)
    age = (_values(rows, 'age') - pooled['centers']['age'])
    cohort = (_values(rows, 'cohort') - pooled['centers']['cohort'])
    trial_design = np.column_stack([base, indicator * age, indicator * cohort])
    trial = _fit(trial_design, pooled['discounted'], pooled['offered'],
                 [row['cohort'] for row in rows])
    added = trial['rank'] - pooled['rank']
    if added <= 0:
        return {'status': 'not_identified', 'detail': '等级交互与现有列共线，无法检验。'}
    difference = max(pooled['deviance'] - trial['deviance'], 0.0)
    f_value = (difference / added) / max(pooled['dispersion'], 1e-12)
    p_value = float(1.0 - stats.f.cdf(f_value, added, max(pooled['df'], 1)))
    return {'status': 'tested', 'F': float(f_value), 'p_value': p_value,
            'detail': '若交互不显著，合并口径的方向斜率可共享；显著则分等级结论不能合并。'}


# ---------------------------------------------------------------------------
# 入口


def _verdict(scope):
    readout = scope['readout']
    contrasts = readout['data_supported_contrasts']
    joint = readout['joint_claim']
    within = contrasts['within_cohort']
    cross = contrasts['cross_cohort_same_age']
    if within['excludes_zero'] and within['estimate'] > 0:
        aging = ('同一入库代次库龄每增加一周，折价 log-odds 上升约 ' +
                 format(within['estimate'], '.3f') + '（区间不含零）；这是可估的联合效应。')
    elif within['excludes_zero']:
        aging = ('同一入库代次库龄每增加一周，折价 log-odds 下降约 ' +
                 format(within['estimate'], '.3f') + '（区间不含零）。')
    else:
        aging = '同一入库代次的库龄变化区间含零，数据不足以确认方向。'
    if cross['excludes_zero']:
        cohort_line = ('同一库龄下较晚代次与较早代次的差异区间不含零（' +
                       format(cross['estimate'], '.3f') + '），但该对比合并了代次与日历，'
                       '不能单独归给“新苹果更耐卖”。')
    else:
        cohort_line = ('同一库龄下较晚代次对较早代次的差异区间含零（' +
                       format(cross['estimate'], '.3f') + '），没有可估证据支持“新入库的苹果更耐卖”。')
    return {
        'warehouse_claim': {
            'claim': '新入库的苹果更耐卖（代次趋势 C<0）。',
            'status': 'not_identified' if not joint['supported'] else 'consistent_with_joint_claim',
            'detail': ('代次趋势本身不可识别；其可估的联合含义 C+P 见 cross_cohort 对比。'),
        },
        'sales_claim': {
            'claim': '近期折价减少只是促销周过去（日历趋势 P<0）。',
            'status': 'not_identified',
            'detail': ('日历趋势本身不可识别；P 的符号随规范化（例如 C=0 与 A=0）改变，'
                       '数据不裁决这一点。'),
        },
        'joint_claim': joint,
        'aging_evidence': aging,
        'cohort_evidence': cohort_line,
        'aging_supported': bool(within['excludes_zero']),
        'cohort_supported': bool(cross['excludes_zero']),
        'summary': _verdict_summary(within['excludes_zero'], cross['excludes_zero']),
    }


def _verdict_summary(aging_supported, cohort_supported):
    supported = []
    if aging_supported:
        supported.append('同一入库代次内的库龄变化')
    if cohort_supported:
        supported.append('同一库龄下的代次对比（仍合并日历效应）')
    if supported:
        opening = '数据支持：' + '、'.join(supported) + '。'
    else:
        opening = '数据没有给出区间不含零的方向证据。'
    return (opening + '仓储负责人与销售各自的说法都需要不可识别的三方趋势分摊，'
            '因此都不能称为数据结论；可估的只是它们的联合含义 cross_cohort 对比。')


def analyse(case, anchors=None):
    cells = validate(case)
    visible = visible_cells(case)
    as_of = case['as_of_period']
    late = [cell for cell in cells.values() if cell['known_at'] > as_of]
    grade_counts = {}
    for cell in visible:
        grade_counts[cell['grade']] = grade_counts.get(cell['grade'], 0) + 1
    fit_rows = [cell for cell in visible if not cell['heldout']]
    heldout_rows = [cell for cell in visible if cell['heldout']]
    anchor_records = _anchor_records(case, anchors)

    diagnostics = []
    if not visible:
        diagnostics.append({'kind': 'no_visible_data',
                            'detail': 'as_of_period 前没有可见时间单元。'})
    if late:
        diagnostics.append({'kind': 'late_recorded_excluded',
                            'detail': 'known_at 晚于 as_of_period 的折价按原有可见口径排除。',
                            'cells': len(late)})
    if not case.get('external_anchors') and anchors is None:
        diagnostics.append({'kind': 'no_external_anchors',
                            'detail': '未提供独立仓储试验锚点，不能给出外部假设下的分摊。'})
    if heldout_rows:
        by_cohort = {}
        for cell in cells.values():
            by_cohort.setdefault(cell['cohort'], []).append(cell)
        partial = sorted({
            row['cohort'] for row in heldout_rows
            if not all(cell['heldout'] for cell in by_cohort[row['cohort']])
        })
        if partial:
            diagnostics.append({
                'kind': 'heldout_not_complete_cohort',
                'cohorts': partial,
                'detail': ('这些代次只有部分单元标记留出；验证不使用它们，'
                           '以免把同代次信息泄漏进留出评估。'),
            })

    scopes = []

    def estimable_scope(rows, pooled):
        if not rows:
            return False
        columns = _base_columns(rows, _centers(rows), pooled)
        return _rank(_matrix(rows, columns)) >= len(columns)

    if visible:
        grades = sorted(grade_counts)
        if len(grades) > 1:
            for grade in grades:
                rows = [cell for cell in fit_rows if cell['grade'] == grade]
                held = [cell for cell in heldout_rows if cell['grade'] == grade]
                if estimable_scope(rows, False):
                    scopes.append((grade, rows, held, False))
                else:
                    diagnostics.append({'kind': 'scope_not_estimable', 'scope': 'grade:' + grade,
                                        'detail': '该等级可见单元不足以识别线性库龄与代次对比。'})
            if estimable_scope(fit_rows, True):
                scopes.append(('combined_grade_adjusted', fit_rows, heldout_rows, True))
            elif fit_rows:
                diagnostics.append({'kind': 'scope_not_estimable', 'scope': 'combined_grade_adjusted',
                                    'detail': '合并口径可见单元不足以识别线性库龄与代次对比。'})
        elif fit_rows:
            if estimable_scope(fit_rows, False):
                scopes.append((grades[0], fit_rows, heldout_rows, False))
            else:
                diagnostics.append({'kind': 'scope_not_estimable', 'scope': grades[0],
                                    'detail': '可见单元不足以识别线性库龄与代次对比。'})
        if not fit_rows and visible:
            diagnostics.append({'kind': 'no_fit_cells',
                                'detail': '所有可见单元都被留出，无法拟合；只能看描述统计。'})

    estimable = []
    sensitivity = {'scenarios': [], 'flips': [], 'anchor_checks': []}
    validation = []
    per_scope = []
    for label, rows, held, pooled in scopes:
        fit = _fit_scope(rows, pooled)
        records, alpha_vector, gamma_vector = _estimable(fit, pooled)
        for record in records:
            record['scope'] = label
        estimable.extend(records)
        alpha = float(alpha_vector @ fit['beta'])
        gamma = float(gamma_vector @ fit['beta'])
        covariance = np.array([
            [float(alpha_vector @ fit['model_covariance'] @ alpha_vector),
             float(alpha_vector @ fit['model_covariance'] @ gamma_vector)],
            [float(gamma_vector @ fit['model_covariance'] @ alpha_vector),
             float(gamma_vector @ fit['model_covariance'] @ gamma_vector)],
        ])
        curvature = [_curvature(fit, rows, dimension) for dimension in ('age', 'cohort', 'period')]
        scope_anchors = [anchor for anchor in anchor_records
                         if anchor.get('usable') and (pooled or not anchor.get('crosses_grade'))]
        scenarios = _scenarios(alpha, gamma, covariance, scope_anchors)
        for scenario in scenarios:
            scenario['scope'] = label
        sensitivity['scenarios'].extend(scenarios)
        if pooled or scope_anchors:
            for anchor in scope_anchors:
                sensitivity['anchor_checks'].append({'scope': label, **_anchor_check(case, anchor, rows)})
        scope_validation = []
        if held:
            scope_validation.append(_cohort_holdout(case, rows, held, pooled, label))
        scope_validation.append(_leave_one_cohort_out(rows, pooled, label))
        validation.extend(scope_validation)
        readout = _readout(alpha, gamma, records, scenarios, curvature, scope_validation)
        per_scope.append({
            'scope': label,
            'pooled_with_grade': pooled,
            'rows': len(rows),
            'grade_counts': {grade: sum(1 for row in rows if row['grade'] == grade)
                             for grade in sorted({row['grade'] for row in rows})},
            'rank': int(fit['rank']),
            'df': int(fit['df']),
            'dispersion': float(fit['dispersion']),
            'deviance': float(fit['deviance']),
            'clusters': int(fit['clusters']),
            'converged': bool(fit['converged']),
            'iterations': int(fit['iterations']),
            'estimable': records,
            'estimand_map': {
                'within_cohort': 'A + P',
                'cross_cohort_same_age': 'C + P',
                'within_period_age': 'A - C',
                'note': 'A/C/P 为库龄、代次、日历各自的线性趋势；两两组合可估，单独趋势不可估。',
            },
            'curvature': curvature,
            'scenarios': scenarios,
            'readout': readout,
        })
        for row in curvature:
            if row['status'] == 'not_identified':
                diagnostics.append({'kind': 'curvature_not_identified', 'scope': label,
                                    'dimension': row['dimension'], 'detail': row['reason']})
        if fit['df'] <= 0:
            diagnostics.append({'kind': 'saturated_scope', 'scope': label,
                                'detail': '该范围没有剩余自由度，不确定性无法由数据估计。'})

    # 三线性趋势的秩亏事实
    full_rank = None
    columns = None
    if per_scope:
        full_columns = [
            ('intercept', lambda data: np.ones(len(data))),
            ('age', lambda data: _values(data, 'age')),
            ('cohort', lambda data: _values(data, 'cohort')),
            ('period', lambda data: _values(data, 'period')),
        ]
        full_rank = _rank(_matrix(fit_rows, full_columns))
        columns = len(full_columns)
    if per_scope:
        diagnostics.append({
            'kind': 'trend_allocation_not_identified',
            'detail': ('cohort + age = period 使三列线性趋势秩亏：三个方向对比可估，'
                       '但库龄、代次、日历各自的线性趋势不能分摊。'),
            'three_trend_rank': full_rank,
            'three_trend_columns': columns,
            'null_direction': {'age': 1, 'cohort': 1, 'period': -1},
        })

    # 等级嵌套事实
    grades_by_cohort = _grade_by_cohort(case)
    nested = bool(grades_by_cohort) and all(len(values) == 1 for values in grades_by_cohort.values())
    if nested:
        diagnostics.append({
            'kind': 'grade_nested_in_cohort',
            'detail': ('每个入库代次只含一个等级；相邻代次必然跨等级，'
                       '代次锚点无法与等级调整后的代次斜率直接比较。'),
        })
        diagnostics.append({
            'kind': 'grade_adjustment_relies_on_linear_cohort',
            'detail': ('等级嵌套在入库代次中，只有把代次趋势限制为线性时，'
                       '等级截距才与代次趋势分开；若代次用自由虚拟变量，等级效应不可识别。'),
        })

    # 翻转：按范围分别比较真正估计出来的趋势；固定为规范值的条目不算翻转
    for scope in per_scope:
        for trend in ('age', 'cohort', 'period'):
            estimates = {}
            for scenario in scope['scenarios']:
                for row in scenario['decomposition']:
                    if row['trend'] == trend and not row['assumed']:
                        estimates[scenario['id']] = row['estimate']
            signs = {(value > 0) - (value < 0) for value in estimates.values() if abs(value) > 1e-12}
            if len(signs) > 1:
                causes = sorted({('external_anchor' if str(identifier).startswith('anchor:')
                                  else 'normalization')
                                 for identifier in estimates})
                sensitivity['flips'].append({
                    'scope': scope['scope'],
                    'trend': trend,
                    'causes': causes,
                    'estimates': estimates,
                    'detail': ('仅因' + ('外部锚点取值' if causes == ['external_anchor']
                                        else '规范化或外部锚点') +
                               '不同，该趋势的符号就发生变化；不是数据识别的结论。'),
                })

    # 外部约束：只引用实际提供且可用的锚点
    constraints = []
    constraints.append({
        'kind': 'substantive',
        'id': 'linear_additive_apc',
        'statement': '库龄与代次以 log-odds 线性、可加方式进入；曲率需单独检验。',
    })
    constraints.append({
        'kind': 'substantive',
        'id': 'grade_scope_separate',
        'statement': '不同等级经营范围不同：分等级拟合；合并口径只加等级截距，并检验等级交互。',
    })
    if fit_rows and len({row['grade'] for row in fit_rows}) > 1:
        pooled_fit = _fit_scope(fit_rows, True)
        interaction = _grade_interaction(fit_rows, pooled_fit)
        constraints.append({'kind': 'substantive', 'id': 'grade_interaction_test',
                            'statement': '合并口径假设各等级方向斜率相同；用交互检验检查该假设。',
                            **interaction})
    seen_anchors = set()
    for anchor in anchor_records:
        if not anchor.get('usable') or anchor['id'] in seen_anchors:
            continue
        seen_anchors.add(anchor['id'])
        constraints.append({
            'kind': 'external',
            'id': 'anchor:' + str(anchor['id']),
            'anchor': anchor['id'],
            'source': anchor.get('source'),
            'statement': '独立仓储试验给出代次 log-odds 差异区间，作为弱约束而非精确真值。',
            'interval': anchor['interval'],
            'cohort_pair': anchor['cohort_pair'],
            'commensurability': anchor.get('commensurability'),
        })
    constraints.append({
        'kind': 'normalization',
        'id': 'three_trend_normalizations',
        'statement': '把三方线性趋势分摊开需要 C=0、P=0 或 A=0 之类的规范化；它们互相等价于不同假设。',
    })
    constraints.append({
        'kind': 'normalization',
        'id': 'coefficient_scale',
        'statement': '方向斜率以每个库龄/代次/日历周为单位；中心化只影响截距，不影响可估对比。',
    })

    # 可估性总结：哪些数据结论、哪些外部假设、哪些不能分摊
    combined = next((scope for scope in per_scope if scope['scope'] == 'combined_grade_adjusted'), None)
    narrowing = {
        'narrowed_estimands': [
            {'scope': row['scope'], 'anchor': row['anchor'], 'estimand': row['narrows'],
             'width_before': row['width_before'], 'width_after': row['width_after']}
            for row in sensitivity['anchor_checks'] if row.get('narrowed')
        ],
        'not_narrowed': [
            'cross_cohort_same_age_grade_adjusted',
            'within_cohort',
            'within_period_age',
            'three_way_trend_allocation',
        ],
        'detail': ('弱约束只收窄与锚点同口径的原始代次对差异；'
                   '可估的方向对比、曲率与三方分摊都不因此获得识别。'),
    }
    sensitivity['anchor_narrowing'] = narrowing
    conclusions = {
        'verdict': _verdict(combined) if combined else None,
        'data_supported_contrasts': combined['readout']['data_supported_contrasts'] if combined else None,
        'data_supported_joint_claim': combined['readout']['joint_claim'] if combined else None,
        'curvature': combined['readout']['curvature'] if combined else None,
        'external_assumption': [row['id'] for row in sensitivity['scenarios'] if row['kind'] == 'external'],
        'anchor_narrowing': narrowing,
        'not_allocable': combined['readout']['not_allocable'] if combined else None,
        'flips': sensitivity['flips'],
    }

    return _plain({
        'snapshot': case['_snapshot'],
        'as_of_period': as_of,
        'origin': case['origin'],
        'age_unit': case['age_unit'],
        'scope_definition': {
            'visible_cells': len(visible),
            'late_cells_excluded': len(late),
            'fit_cells': len(fit_rows),
            'heldout_cells': len(heldout_rows),
            'grade_counts': grade_counts,
            'outcome': 'discounted / offered（折价箱数占待售箱数）',
            'not_used': '各批总销售额',
        },
        'descriptive': _descriptive(visible),
        'descriptive_scope': 'as_of_period 可见单元（含留出代次），只作描述，不进入拟合。',
        'estimable': estimable,
        'constraints': constraints,
        'sensitivity': sensitivity,
        'validation': validation,
        'diagnostics': diagnostics,
        'scopes': per_scope,
        'conclusions': conclusions,
    })
