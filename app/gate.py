from app.store import unique


def departure_manifest(case):
    jobs = unique(case['jobs'], 'id')
    seen = set()
    result = []
    for departure in case['departures']:
        keys = [(departure['shift'], identifier) for identifier in departure['released_jobs']]
        if len(set(keys)) != len(keys) or any(key in seen for key in keys):
            raise ValueError('job released twice')
        for identifier in departure['released_jobs']:
            if jobs[identifier]['shift'] != departure['shift']:
                raise ValueError('cross-shift manifest')
        seen.update(keys)
        result.append({**departure, 'actual_units': sum(jobs[identifier]['units'] for identifier in departure['released_jobs'])})
    return result
