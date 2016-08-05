"""
Simulate scheduling with a variety of prices.

"""

import datetime
import os

import json
import logging
import numpy as np
import pandas as pd
import cv2

import steel_util
from build_plant_rtn import PlantRtnBuilder, PlantRtn2Builder
from build_opt_model import OptModelBuilder, OptModelBuilderRTN2
from solve_schedule import solve_cplex

# setup log
log = logging.getLogger('change_price')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s %(module)s %(levelname)s %(lineno)d: - %(message)s'))
fh = logging.FileHandler('change_price.log')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s %(module)s %(levelname)s %(lineno)d: - %(message)s'))
fh.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(ch)
log.addHandler(fh)


def run_rtn(case, options, work_dir):
    if case['model'] == 'rtn1':
        steel_rtn = PlantRtnBuilder(case)
        opt_math_model = OptModelBuilder(steel_rtn, options)
    elif case['model'] == 'rtn2':
        steel_rtn = PlantRtn2Builder(case)
        opt_math_model = OptModelBuilderRTN2(steel_rtn, options)
    else:
        opt_math_model = None

    result = solve_cplex(opt_math_model)

    if result['status'] in [101, 102, 107]:
        log.info('%-40s obj %.1f group %d CPU %d rel_gap %.4f relax_obj %.1f status %d'
                 % (case['doc'], result['obj'], case['group_num'], result['cpu_time'],
                    result['rel_gap'], result['best_relaxed_obj'], result['status']))
        xx = result['xx'].toarray()
        yy = result['yy'].toarray()
        steel_util.dump_schedule(case, xx.reshape(-1).tolist(), yy.reshape(-1).tolist(), work_dir)
    else:
        log.error('%s No solution after %d s status %d' % (case['doc'], result['cpu_time'], result['status']))


def simulate(group_num, rtn_t0=15, plant='plant3', model='rtn1', price=[10.0]*24, price_id='0',
             work_dir='/', run_clex=True):
    test_case = json.load(open('data/%s_%s.json' % (plant, model), 'r'))

    test_case['energy_price'] = price

    test_case['model'] = model
    test_case['rtn_t0'] = rtn_t0
    test_case['group_num'] = group_num
    test_case['case'] = plant
    doc = '%s_%s_G%d_t%d_p%s' % (plant, model, group_num, rtn_t0, price_id)
    test_case['doc'] = doc
    test_case['price_id'] = price_id

    for key in test_case['group2heats'].keys():
        if int(key) > test_case['group_num']:
            del test_case['group2heats'][key]

    if run_clex:
        run_rtn(test_case, None, work_dir)

    # steel_util.check_model(test_case, None, ref_name='rtn2_G1_t15_case2')
    steel_util.draw_schedule(test_case, work_dir)


def load_eem_prices(size=None):
    df = pd.read_csv('data/price_eem2016.csv')
    assert len(df) % 24 == 0

    df['date'] = pd.to_datetime(df['date'])

    mask = (df['date'] >= datetime.datetime(2015, 4, 28)) & (df['date'] < datetime.datetime(2016, 1, 1))
    # mask = (df['date'] >= pd.to_datetime('20150101')) & (df['date'] < pd.to_datetime('20160101'))
    df = df.loc[mask]

    n_days = len(df)/24
    price_list = df['price'].values.reshape((n_days, 24)).tolist()
    date_list = [df['date'].iloc[24*i].strftime('%Y%m%d') for i in range(n_days)]
    if size:
        price_list = price_list[:size]
        date_list = date_list[:size]
    return price_list, date_list


def run_simulations(price_list, date_list):
    """run simulations and record results"""
    for group in GROUPS:
        work_dir = WORK_DIR_BASE + 'G%d/' % group
        for i in range(len(prices)):
            simulate(group_num=group, rtn_t0=15, model='rtn2', price=price_list[i], price_id=date_list[i],
                     work_dir=work_dir, run_clex=True)


def write_video(date_list):
    prefix = 'schedule_plant3_rtn2_G'
    for group in GROUPS:
        video = None
        for date in date_list:
            file_png = '%s/G%d/%s%d_t15_p%s.png' % (WORK_DIR_BASE, group, prefix, group, date)
            if not os.path.isfile(file_png):
                log.warn('Not exist %s' % file_png)
                continue
            img = cv2.imread(file_png)
            if not video:
                height, width, layers = img.shape
                file_avi = '%sschedule_G%d.avi' % (WORK_DIR_BASE, group)
                log.info('Write video to %s' % file_avi)
                fourcc = cv2.cv.CV_FOURCC(*'XVID')
                video = cv2.VideoWriter(file_avi, fourcc, 2, (width, height))
            video.write(img)
        if video:
            video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    log.info(str(datetime.datetime.now()))

    WORK_DIR_BASE = '../../../../Downloads/data/steel_price/'
    GROUPS = [3]

    prices, dates = load_eem_prices()

    run_simulations(prices, dates)

    write_video(dates)

