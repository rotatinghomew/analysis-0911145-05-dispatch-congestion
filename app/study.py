from app.study_schema import rows_by_id, positive_integer


def validate(case):
    cells = rows_by_id(case['cells'])
    for cell in cells.values():
        if cell['cohort'] + cell['age'] != cell['period']:
            raise ValueError('库龄、入库代次与日历周不一致')
        positive_integer(cell['offered'])
        if not isinstance(cell['discounted'], int) or not 0 <= cell['discounted'] <= cell['offered']:
            raise ValueError('折价箱数超出待售范围')
    return cells


def visible_cells(case):
    return [cell for cell in validate(case).values() if cell['known_at'] <= case['as_of_period']]


def design_rows(case):
    cells = visible_cells(case)
    columns = sorted({f'{dimension}:{cell[dimension]}' for cell in cells
                      for dimension in ('age', 'cohort', 'period')})
    return columns, [[int(column in {f'{dimension}:{cell[dimension]}'
                   for dimension in ('age', 'cohort', 'period')}) for column in columns] for cell in cells]


def preview(case):
    cells = visible_cells(case)
    columns, design = design_rows(case)
    counts = {}
    for cell in cells:
        entry = counts.setdefault(cell['period'], {'offered': 0, 'discounted': 0})
        entry['offered'] += cell['offered']
        entry['discounted'] += cell['discounted']
    return {'snapshot': case['_snapshot'], 'calendar_counts': counts,
            'design_columns': columns, 'design_rows': len(design),
            'anchors': case['external_anchors'], 'descriptive_only': True}
