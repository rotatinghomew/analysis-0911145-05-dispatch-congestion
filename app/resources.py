def capacity_at(case, minute):
    assignment = dict(case['base_assignment'])
    total = sum(row['capacity'] for row in case['resource_calendar'] if row['start'] <= minute < row['end'])
    for loan in case['loans']:
        if loan['start'] <= minute < loan['end']:
            assignment[loan['from']] -= loan['workers']
            assignment[loan['to']] += loan['workers']
    if any(value < 0 for value in assignment.values()) or sum(assignment.values()) > total:
        raise ValueError('resource assignment exceeds available crew')
    return assignment
