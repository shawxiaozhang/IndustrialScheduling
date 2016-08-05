""" Solve optimal scheduling of steel plants by tailored branch and bound.
__author__ = xxxzhang
"""

from Queue import PriorityQueue
import math
import sys

import numpy as np
import time
import matplotlib.pyplot as plt
from steel_util import SteelHelper, ProbPara
import json
import logging
from scipy import sparse

import solve_schedule
import util_search_schedule

BB_OBJ_INTEGER_EPSILON = 1e-6
BB_VAR_INTEGER_EPSILON = 1e-6
BB_MAX_CPU = 7200
BB_BRANCH_SWITCH_THRESHOLD = 4
LEAD_FOLLOWER_EACH_STAGE = 'EACH STAGE'
LEAD_FOLLOWER_ALL_STAGES = 'ALL STAGES'
log = logging.getLogger('steel')


class SteelBranchBound():
    def __init__(self, steel_rtn, opt_model, branch_option=LEAD_FOLLOWER_ALL_STAGES):
        # todo reduce lp-solve-num, look at lp infeasibility
        # todo lp num, consider lead task for all stages ...
        self.opt_model = opt_model
        self.steel_rtn = steel_rtn
        self.cplex_lp_prob = solve_schedule.convert_to_cplex(opt_model)

        self.q_relax = PriorityQueue()
        self.q_integer = PriorityQueue()
        self.best_int_obj = sys.maxint
        self.branch_option = branch_option
        if branch_option == LEAD_FOLLOWER_EACH_STAGE:
            self.follower_offset = self.get_follower_offset_each_stage(steel_rtn)
        elif branch_option == LEAD_FOLLOWER_ALL_STAGES:
            self.follower_offset = self.get_follower_offset_all_stages(steel_rtn)
        self.group2caster = self._assign_caster()
        self.lead2follower = dict()

        self.count_lp = 0
        self.count_bb_node = 0
        self.obj_int_iter = dict()
        self.obj_relax_iter = dict()

    def solve_bb(self):
        t1 = time.time()

        # time range [) for task to start
        task_start_rng = dict()
        for task in range(1, self.steel_rtn.num_tasks+1):
            task_start_rng[task] = (-1, -1)
        initial_result = self.solve_lp(task_start_rng)
        self.check_solution(initial_result['obj'], initial_result['xx'], initial_result['yy'], task_start_rng)

        schedule_guess = util_search_schedule.find_feasible_task_rng(self.steel_rtn)
        result_guess = self.solve_lp(schedule_guess)
        if result_guess is not None:
            self.check_solution(result_guess['obj'], result_guess['xx'], result_guess['yy'], schedule_guess)
        else:
            log.error('No relaxation solution.')
            # return None

        while not self.q_relax.empty():
            if time.time() - t1 > BB_MAX_CPU:
                break
            best_relax = self.q_relax.get()
            node = {'obj': best_relax[0], 'start_rng': best_relax[2]}
            self.obj_relax_iter[self.count_lp] = best_relax[0]
            if self.best_int_obj - node['obj'] < BB_OBJ_INTEGER_EPSILON:
                log.info('b+b terminates as upper lower bounds are close %.3f v.s. %.3f' % (self.best_int_obj, node['obj']))
                break
            else:
                for node_i in self.branch(node):
                    result = self.solve_lp(node_i['start_rng'])
                    if result is None:
                        log.debug('Infeasible LP start_rng %s' % node_i['start_rng'])
                        continue
                    self.check_solution(result['obj'], result['xx'], result['yy'], node_i['start_rng'])

        cpu_time = time.time() - t1
        log.info('best integer objective %.6f' % self.best_int_obj)
        log.info('compute time: %.3f s' % cpu_time)
        log.info('lp solve num = %d' % self.count_lp)
        log.info('generated bb node num = %d' % self.count_bb_node)
        log.info('q_relax remained node num = %d' % self.q_relax.qsize())
        self.draw_iter()
        # with open(case_name+'_iter.json','w+') as f:
        #     json.dump({'q1':self.debug_q, 'q2_int':self.debug_q_int},f, indent=2)
        return self.q_integer.get()[2]

    def check_solution(self, obj, xx, yy, start_rng):
        compare = (np.absolute(xx) < BB_VAR_INTEGER_EPSILON) | (np.absolute([x-1 for x in xx]) < BB_VAR_INTEGER_EPSILON)
        is_integer = compare.all()
        if is_integer:
            self.q_integer.put((obj, self.count_lp, {'xx': sparse.coo_matrix(xx), 'yy': sparse.coo_matrix(yy),
                                                     'start_rng': start_rng, 'obj': obj}))
            self.best_int_obj = min(obj, self.best_int_obj)
            self.obj_int_iter[self.count_lp] = self.best_int_obj
        else:
            self.q_relax.put((obj, self.count_lp, start_rng))

        # todo add a better rounding method
        if self.count_bb_node > 500 and self.count_bb_node % 500 == 1:
            log.warn('count_bb_node = %d' % self.count_bb_node)
            log.warn('q_relax.qsize = %d' % self.q_relax.qsize())
        if self.count_bb_node > 10 ** 10:
            log.warn('count_bb_node = %d' % self.count_bb_node)
            log.warn('too long : queue size %d' % self.q_relax.qsize())
            self.q_relax = PriorityQueue()
            return True

        log.debug('%d int:%s obj %.1f start_rnt: %s' % (self.count_lp, is_integer, obj, json.dumps(start_rng)))

        return is_integer

    def solve_lp(self, task_start_range):
        x_up = self.get_x_up(task_start_range, self.opt_model, self.steel_rtn)
        self.cplex_lp_prob.variables.set_upper_bounds(zip(range(len(x_up)), x_up))
        self.cplex_lp_prob.solve()
        self.count_lp += 1
        status = self.cplex_lp_prob.solution.get_status()
        if status == self.cplex_lp_prob.solution.status.optimal:
            return {'obj': self.cplex_lp_prob.solution.get_objective_value(),
                    'xx': self.cplex_lp_prob.solution.get_values()[0:self.opt_model.num_x],
                    'yy': self.cplex_lp_prob.solution.get_values()[self.opt_model.num_x:]}
        else:
            # log.error('LP solve status %s' % status)
            return None

    def branch(self, node):
        # todo add more branch options
        return self._branch_by_lead_task(node)

    def _branch_by_lead_task(self, node):
        """Branch by lead tasks.
        Consider the start time ranges for lead task and its followers all-together in one bb node.
        Lead task: the first heat in each group.
        """
        start_rng = node['start_rng']
        # if any task hasn't been considered, then consider this task
        if self.branch_option == LEAD_FOLLOWER_EACH_STAGE:
            for stage in [1, 2, 3]:
                task_cat = self.steel_rtn.stage2units[str(stage)].keys()[0]
                for group, heats in self.steel_rtn.group2heats.items():
                    task1 = self.steel_rtn.tasks[task_cat][heats[0] - 1]
                    if start_rng[task1][0] < 0:
                        split_task = [task1]
                        for idx in range(1, len(heats)):
                            task_i = self.steel_rtn.tasks[task_cat][heats[idx] - 1]
                            split_task.append(task_i)
                        self.lead2follower[task1] = split_task
                        return self._branch_node_batch_tasks(node, split_task)
        elif self.branch_option == LEAD_FOLLOWER_ALL_STAGES:
            task_type = 'EAF'
            for group_, heats in self.steel_rtn.group2heats.items():
                task_lead = self.steel_rtn.tasks[task_type][heats[0] - 1]
                if start_rng[task_lead][0] < 0:
                    task_array = []
                    for stage in [1, 2, 3]:
                        task_type = self.steel_rtn.stage2units[str(stage)].keys()[0]
                        for idx in range(len(heats)):
                            task_i = self.steel_rtn.tasks[task_type][heats[idx] - 1]
                            task_array.append(task_i)
                    self.lead2follower[task_lead] = task_array
                    return self._branch_node_batch_tasks(node, task_array)
        # caster has been assigned ahead
        for group in range(self.steel_rtn.num_groups):
            caster = self.group2caster[group+1]
            task = self.opt_model.steel_rtn.tasks[caster][group]
            if start_rng[task][0] < 0:
                self.lead2follower[task] = task
                return self._branch_node_batch_tasks(node, task)
        # all leading tasks have been considered, then narrow their start time range
        [split_task, rng_max] = self._widest_range(self.lead2follower.keys(), start_rng)
        # narrow the leader task or narrow all tasks
        if rng_max > BB_BRANCH_SWITCH_THRESHOLD:
            return self._branch_node_batch_tasks(node, self.lead2follower[split_task])
        else:
            [split_task, rng_max] = self._widest_range(range(1, self.steel_rtn.num_tasks+1), start_rng)
            if rng_max > 1:
                return self._branch_node_batch_tasks(node, split_task)
            else:
                log.error('[warn] No Freedom' + str(start_rng))
                return []

    def _branch_node_batch_tasks(self, node, split_task):
        # todo delete isinstance checking
        if isinstance(split_task, list):
            task1 = split_task[0]
        else:
            task1 = split_task

        start_rng = node['start_rng']
        if start_rng[task1][0] >= 0 and start_rng[task1][1] <= self.opt_model.num_t:
            time_pair = start_rng[task1]
        else:
            time_pair = (0, self.opt_model.num_t)
        # branch into two nodes
        start_rng_1 = start_rng.copy()
        start_rng_2 = start_rng.copy()
        middle = int(math.floor(sum(time_pair) / 2))
        start_rng_1[task1] = (time_pair[0], middle)
        start_rng_2[task1] = (middle, time_pair[1])
        # also restrict the start time ranges for the follower tasks
        if isinstance(split_task, list):
            for task_i in split_task[1:]:
                offset = self.follower_offset[task_i]
                start_rng_1[task_i] = (start_rng_1[task1][0] + offset[0], start_rng_1[task1][1] + offset[1])
                start_rng_2[task_i] = (start_rng_2[task1][0] + offset[0], start_rng_2[task1][1] + offset[1])

        self.count_bb_node += 1
        node_1 = {'obj': node['obj'], 'count': self.count_bb_node, 'start_rng': start_rng_1}
        self.count_bb_node += 1
        node_2 = {'obj': node['obj'], 'count': self.count_bb_node, 'start_rng': start_rng_2}
        return [node_1, node_2]

    def draw_iter(self):
        plt.figure()
        obj_int = [0]*(self.count_lp+1)
        obj_relax = [0]*(self.count_lp+1)
        best_int = max(self.obj_int_iter.values())
        best_relax = min(self.obj_relax_iter.values())
        for i in range(self.count_lp+1):
            if i in self.obj_int_iter:
                best_int = self.obj_int_iter[i]
            if i in self.obj_relax_iter:
                best_relax = self.obj_relax_iter[i]
            obj_int[i] = best_int
            obj_relax[i] = best_relax
        plt.plot(range(self.count_lp+1), obj_int)
        plt.plot(range(self.count_lp+1), obj_relax)
        plt.show()

    @staticmethod
    def get_follower_offset_each_stage(steel_rtn):
        """Get the time offsets between lead task and its followers.
        Lead task: the first heat in each group.
        Follower task: the other heats in each group.

        Lead and follower tasks are in the same process stage.
        For RTN1.
        """
        offset = dict()
        for stage in [1, 2, 3]:
            for group, heats in steel_rtn.group2heats.items():
                task_cat = steel_rtn.stage2units[str(stage)].keys()[0]
                num_equip = steel_rtn.stage2units[str(stage)].values()[0]
                equip_offset = [0] * num_equip
                delays = 0
                for idx in range(len(heats)):
                    heat = heats[idx]
                    task = steel_rtn.tasks[task_cat][heat - 1]
                    equip_id = idx % num_equip
                    if idx == 0:
                        offset[task] = (0, 0)
                        delays += steel_rtn.task_duration[task]
                    elif idx < num_equip:
                        offset[task] = (0, delays)
                        delays += steel_rtn.task_duration[task]
                    elif idx >= num_equip:
                        task_pre = steel_rtn.tasks[task_cat][heat - 1 - num_equip]
                        task_pre_len = steel_rtn.task_duration[task_pre]
                        start = offset[task_pre][0] + task_pre_len
                        end = offset[task_pre][1] + task_pre_len
                        offset[task] = (start, end)
                    equip_offset[equip_id] += steel_rtn.task_duration[task]
        return offset

    @staticmethod
    def get_follower_offset_all_stages(steel_rtn):
        """Get the time offsets between lead task and its followers.
        Lead task: the first heat in each group.
        Follower task: the other heats in each group.

        Lead task is the first heat in the first process stage.
        The other heats in all the first three stages are the followers.
        For RTN1.
        """
        offset = SteelBranchBound.get_follower_offset_each_stage(steel_rtn)
        for stage in [2, 3]:
            task_cat = steel_rtn.stage2units[str(stage)].keys()[0]
            for group, heats in steel_rtn.group2heats.items():
                for idx in range(len(heats)):
                    heat = heats[idx]
                    task_process = steel_rtn.tasks[task_cat][heat - 1]
                    if task_process in offset:
                        del offset[task_process]
                    task_trans = task_process - steel_rtn.num_heats
                    min_trans_time = steel_rtn.task_duration[task_trans]
                    max_trans_time = int(math.ceil(steel_rtn.time_trans_max['TR_S%d' % (stage-1)]/steel_rtn.rtn_t0))
                    # todo a smaller max_trans_time reduces the branches
                    max_trans_time = min_trans_time
                    task_pre_stage = steel_rtn.tasks[steel_rtn.stage2units[str(stage-1)].keys()[0]][heat - 1]
                    offset[task_process] = (offset[task_pre_stage][0] + min_trans_time,
                                            offset[task_pre_stage][1] + max_trans_time)
        return offset

    @staticmethod
    def get_x_up(start_rng, opt_model, steel_rtn):
        x_up = [1]*opt_model.num_x
        for task_cat, task_list in steel_rtn.tasks.items():
            for task in task_list:
                if task not in start_rng or start_rng[task] == (-1, -1):
                    continue
                for t in range(opt_model.num_t):
                    if t not in range(start_rng[task][0], start_rng[task][1]):
                        x_up[opt_model.pos_x_task_t(task, t)] = 0
                # restrict its parallel tasks
                if 'CC' in task_cat:
                    g_idx = task - steel_rtn.tasks[task_cat][0]
                    for caster in steel_rtn.stage2units['4'].keys():
                        if caster == task_cat:
                            continue
                        cast_task = steel_rtn.tasks[caster][g_idx]
                        for t_ in range(steel_rtn.num_t):
                            x_up[opt_model.pos_x_task_t(cast_task, t_)] = 0
        return x_up

    @staticmethod
    def _widest_range(considered_tasks, task_start_rng):
        split_task = -1
        width_max = -1
        for task in considered_tasks:
            if task_start_rng[task][0] < 0:
                continue
            width = task_start_rng[task][1] - task_start_rng[task][0]
            if width_max < width:
                width_max = width
                split_task = task
        return [split_task, width_max]

    @staticmethod
    def _assign_caster():
        # todo a better caster-group assignment method
        group2caster = {1: 'CC2', 2: 'CC2', 3: 'CC2', 4: 'CC1', 5: 'CC2', 6: 'CC1'}
        return group2caster


