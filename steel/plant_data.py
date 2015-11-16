""" Steel plant case data, and dump it as json.
"""

import json
import xlrd

# whether to aggregate the units from the first three stages
AGGREGATE_UNITS = False

group2heats = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12],
               4: [13, 14, 15, 16, 17], 5: [18, 19, 20], 6: [21, 22, 23, 24]}

equip2mw = {'EAF': 85, 'AOD': 2, 'LF': 2, 'CC1': 7, 'CC2': 7}
energy_price = [23.0000, 23.1400, 24.6900, 24.450, 24.4600, 24.290,
                24.8700, 30.1600, 33.3900, 34.5200, 35.4900, 32.5400,
                32.3100, 29.4300, 27.2000, 26.2900, 25.9000, 27.1900,
                34.9300, 45.2000, 35.4700, 33.4800, 32.1000, 27.9300]

equip2process_time = dict()
book = xlrd.open_workbook('data/test.xls')
sheet = book.sheet_by_name('ProcessingTime')
for col in range(2, 10):
    unit = sheet.cell(0, col).value.encode('ascii')
    equip2process_time[unit] = dict()
    for row in range(1, 13):
        heat1 = int(sheet.cell(row, 0).value)
        heat2 = int(sheet.cell(row, 1).value)
        for heat in range(heat1, heat2 + 1):
            duration = sheet.cell(row, col).value
            equip2process_time[unit][heat] = duration

stage2units = {'1': {'EAF': 2}, '2': {'AOD': 2}, '3': {'LF': 2}, '4': {'CC1': 1, 'CC2': 1}}

trans_time = {'TR_S1': 10, 'TR_S2': 4, 'TR_S3': 10}
trans_time_max = {'TR_S1': 240, 'TR_S2': 240, 'TR_S3': 120}
setup_time = {'CC1': 70, 'CC2': 50}

if AGGREGATE_UNITS:
    equip2process_time['EAF'] = equip2process_time.pop('EAF1')
    equip2process_time.pop('EAF2')
    equip2process_time['AOD'] = equip2process_time.pop('AOD1')
    equip2process_time.pop('AOD2')
    equip2process_time['LF'] = equip2process_time.pop('LF1')
    equip2process_time.pop('LF2')
    model = 'rtn1'
else:
    trans_time = {'EAF1-AOD1': 10, 'EAF1-AOD2': 20, 'EAF2-AOD1': 20, 'EAF2-AOD2': 10,
                  'AOD1-LF1': 4, 'AOD1-LF2': 20, 'AOD2-LF1': 20, 'AOD2-LF2': 8,
                  'LF1-CC1': 10, 'LF1-CC2': 20, 'LF2-CC1': 10, 'LF2-CC2': 15}
    trans_time_max = {'EAF1-AOD1': 240, 'EAF1-AOD2': 240, 'EAF2-AOD1': 240, 'EAF2-AOD2': 240,
                      'AOD1-LF1': 240, 'AOD1-LF2': 240, 'AOD2-LF1': 240, 'AOD2-LF2': 240,
                      'LF1-CC1': 120, 'LF1-CC2': 120, 'LF2-CC1': 120, 'LF2-CC2': 120}
    stage2units = {'1': {'EAF1': 1, 'EAF2': 1}, '2': {'AOD1': 1, 'AOD2':1},
                   '3': {'LF1': 1, 'LF2': 1}, '4': {'CC1': 1, 'CC2': 1}}
    model = 'rtn2'

test_case = {'equip2mw': equip2mw, 'group2heats': group2heats, 'energy_price': energy_price,
             'equip2process_time': equip2process_time, 'stage2units': stage2units,
             'trans_time': trans_time, 'trans_time_max': trans_time_max,
             'setup_time': setup_time}

json.dump(test_case, open('data/plant-1 %s.json' % model, 'w+'), indent=2)
print 'dump to', 'data/plant-1 %s.json' % model


