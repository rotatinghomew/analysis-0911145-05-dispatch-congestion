from collections import defaultdict
from app.resources import capacity_at
from app.gate import departure_manifest


def scan_intervals(job, observation_end):
    intervals = {}
    for station in ('sort', 'load'):
        start = job['scans'].get(station + '_start')
        end = job['scans'].get(station + '_end')
        if start is not None and end is not None and start > end:
            raise ValueError('negative observed duration')
        intervals[station] = {'start': start, 'end': end, 'complete': start is not None and end is not None,
                              'observation_end': observation_end}
    return intervals


def preview(case):
    shifts = defaultdict(lambda: {'jobs': 0, 'complete_sort': 0, 'sort_minutes': 0})
    for job in case['jobs']:
        group = shifts[job['shift']]
        group['jobs'] += 1
        interval = scan_intervals(job, case['observation_end'])['sort']
        if interval['complete']:
            group['complete_sort'] += 1
            group['sort_minutes'] += interval['end'] - interval['start']
    return {'snapshot': case['_snapshot'], 'shifts': dict(shifts),
            'capacity_at_40': capacity_at(case, 40), 'departures': departure_manifest(case),
            'missing_scan_is_not_zero': True}
