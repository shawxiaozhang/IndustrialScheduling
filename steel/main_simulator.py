"""

__author__ = xxxzhang
"""

import json
import logging

import solve_schedule_bb
import steel_util
from build_plant_rtn import PlantRtnBuilder
from build_opt_model import OptModelBuilder
from solve_schedule import solve_cplex

# setup log
log = logging.getLogger('steel')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(module)s %(levelname)s %(lineno)d: - %(message)s'))
fh = logging.FileHandler('steel.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(module)s %(levelname)s %(lineno)d: - %(message)s'))
log.addHandler(ch)
# log.addHandler(fh)


if __name__ == "__main__":
    group_num = 2
    rtn_t0 = 15

    test_case = json.load(open('data/plant-1 rtn1.json', 'r'))
    test_case['rtn_t0'] = 15
    for key in test_case['group2heats'].keys():
        if int(key) > group_num:
            del test_case['group2heats'][key]

    steel_rtn = PlantRtnBuilder(test_case, 15)
    opt_math_model = OptModelBuilder(steel_rtn)
    # SteelHelper.SteelHelper.check_model(steel_rtn, opt_math_model)

    result = solve_cplex(opt_math_model)

    # result = solve_schedule_bb.SteelBranchBound(steel_rtn, opt_math_model).solve_bb()

    # todo reduce lp-solve-num, look at lp infeasibility
    # todo lp num, consider lead task for all stages ...
    if result is not None:
        # print json.dumps(result, indent=2)
        print 'obj %.1f group %d' % (result['obj'], group_num)
        xx = [0]*opt_math_model.num_x
        for col in result['xx'].col:
            xx[col] = 1.0
        steel_util.SteelHelper().draw_schedule(xx, steel_rtn, '')
    else:
        print 'No solution.'