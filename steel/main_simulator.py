"""

__author__ = xxxzhang
"""

import datetime
import copy

import json
import logging
import numpy as np

import solve_schedule_bb
import steel_util
from build_plant_rtn import PlantRtnBuilder, PlantRtn2Builder
from build_opt_model import OptModelBuilder, OptModelBuilderRTN2
from solve_schedule import solve_cplex

# setup log
log = logging.getLogger('steel')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(module)s %(levelname)s %(lineno)d: - %(message)s'))
fh = logging.FileHandler('steel.log')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s %(module)s %(levelname)s %(lineno)d: - %(message)s'))
fh.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(ch)
log.addHandler(fh)

DIR_TEMP = 'tmp/'


def run_rtn(case, options):
    if case['model'] == 'rtn1':
        steel_rtn = PlantRtnBuilder(case)
        opt_math_model = OptModelBuilder(steel_rtn, options)
    elif case['model'] == 'rtn2':
        steel_rtn = PlantRtn2Builder(case)
        opt_math_model = OptModelBuilderRTN2(steel_rtn, options)
    else:
        opt_math_model = None

    # log.info('task %d resource %d binary %d total %d constraint %d T %d'
    #          % (opt_math_model.num_tasks, opt_math_model.num_resources,
    #             opt_math_model.num_x, (opt_math_model.num_y+opt_math_model.num_x),
    #             opt_math_model.num_con, opt_math_model.num_t))

    result = solve_cplex(opt_math_model)
    # result = solve_schedule_bb.SteelBranchBound(steel_rtn, opt_math_model).solve_bb()

    if result['status'] in [101, 102, 107]:
        # print json.dumps(result, indent=2)
        log.info('%-40s obj %.1f group %d CPU %d rel_gap %.4f relax_obj %.1f status %d'
                 % (case['doc'], result['obj'], case['group_num'], result['cpu_time'],
                    result['rel_gap'], result['best_relaxed_obj'], result['status']))
        xx = result['xx'].toarray()
        np.savetxt('%sxx_%s.txt' % (DIR_TEMP, case['doc']), xx)
        yy = result['yy'].toarray()
        np.savetxt('%syy_%s.txt' % (DIR_TEMP, case['doc']), yy)
    else:
        log.error('%s No solution after %d s status %d' % (case['doc'], result['cpu_time'], result['status']))


def simulate(group_num, rtn_t0=15, plant='plant1', model='rtn2', suffix='', heu=False, acc=False, heu_all=False):
    test_case = json.load(open('data/%s_%s.json' % (plant, model), 'r'))

    # test_case['energy_price'] = [10, 20] + [h*h+5 for h in range(21, -1, -1)]
    # json.dump(test_case, open('data/a2_rtn2.json', 'w+'), indent=2)

    test_case['model'] = model
    test_case['rtn_t0'] = rtn_t0
    test_case['group_num'] = group_num
    test_case['case'] = plant
    doc = '%s_%s_G%d_t%d' % (plant, model, group_num, rtn_t0)

    opt_options = {'heuristic_eaf': heu,
                   'obj_f_type': 'MAKE_SPAN',
                   # 'impose_group_order_stages': [1],
                   'heuristic_all': heu_all,
                   'accurate_wait': acc}

    if 'accurate_wait' in opt_options and opt_options['accurate_wait']:
        doc += '_acc'
        opt_options['task_order'] = True
    if 'heuristic_eaf' in opt_options and opt_options['heuristic_eaf']:
        doc += '_heu'
    if 'heuristic_all' in opt_options and opt_options['heuristic_all']:
        doc += '_all'
    if bool_heat_trick:
        doc += '_trick'
    if 'obj_f_type' in opt_options and opt_options['obj_f_type'] == 'MAKE_SPAN':
        doc = 'span_%s' % doc
    test_case['doc'] = doc + suffix

    # log.info('Opt options ...')
    # log.info(json.dumps(opt_options, indent=2))

    for key in test_case['group2heats'].keys():
        if int(key) > test_case['group_num']:
            del test_case['group2heats'][key]

    original_process_time = copy.deepcopy(test_case['equip2process_time'])
    if bool_heat_trick:
        t_step = 1
        for g, heats in test_case['group2heats'].items():
            d_t = t_step*len(heats)
            for heat in heats:
                d_t -= t_step
                for u in test_case['equip2process_time'].keys():
                    test_case['equip2process_time'][u][str(heat)] -= d_t

    if bool_solve_opt:
        run_rtn(test_case, opt_options)

    if bool_check_model:
        steel_util.check_model(test_case, opt_options, ref_name='rtn2_G1_t15_case2')

    if bool_display:
        if bool_heat_trick:
            test_case['equip2process_time'] = original_process_time
        steel_util.draw_schedule(test_case, opt_options)

if __name__ == "__main__":
    """ Simulations with various models. """

    bool_solve_opt = True
    bool_display = True
    bool_check_model = False

    # todo set all transfer tasks to be continuous variables?
    # todo tune the price curve here

    log.info(str(datetime.datetime.now()))
    for t in [15]:
        for group in [3, 4, 5, 6]:
            for acc_wait in [True]:
                for heu_eaf in [False]:
                    for bool_heat_trick in [False]:
                        for heu_all_stages in [False]:
                            simulate(group, rtn_t0=t, heu=heu_eaf, acc=acc_wait, model='rtn2', heu_all=heu_all_stages, plant='plant2')