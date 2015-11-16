__author__ = 'suzhou'

from Queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np
import logging
log = logging.getLogger('steel')


def check_model(plant, model):
    schedule = SteelHelper.find_feasible_schedule_new(plant)
    xx, yy = SteelHelper.parse_schedule(schedule, model, plant)
    print 'assert y up bounds'
    exceptions = SteelHelper.assert_vector(yy, model.y_up, '<=')

    print 'assert y lo bounds'
    exceptions = SteelHelper.assert_vector(yy, model.y_lo, '>=')

    print 'assert task execution'
    SteelHelper.assert_vector(model.con_exe_x.dot(xx), 1.0, '==')

    print 'assert waiting time'
    SteelHelper.assert_vector(model.con_wait_y.dot(yy), model.con_wait_b, '<')

    # print 'EN calculation'
    # SteelHelper.assert_vector(model.con_en_x.dot(xx) + model.con_en_y.dot(yy), 0.0, '~=')

    print 'assert res balance'
    exceptions = SteelHelper.assert_vector(model.con_res_x.dot(xx) + model.con_res_y.dot(yy), model.con_res_b, '==')
    for j, val1, val2 in exceptions:
        res, t = model.inv_pos_y(j)
        print 'idx %d res %s t %d : %s %s' % (j, res, t, '%.1f(=%.1f+%.1f)' % (val1, model.con_res_x.dot(xx)[j], model.con_res_y.dot(yy)[j]), val2)
        #     todo need reverse mapping here


def assert_vector(vector1, vector2, sign):
    if isinstance(vector2, float) or isinstance(vector2, int):
        vector2 = [vector2]*len(vector1)
    if sign == '~=':
        compare = [vector1[i] != vector2[i] for i in range(len(vector1))]
    elif sign == '==':
        compare = [vector1[i] == vector2[i] for i in range(len(vector1))]
    elif sign == '<=':
        compare = [vector1[i] <= vector2[i] for i in range(len(vector1))]
    elif sign == '>=':
        compare = [vector1[i] >= vector2[i] for i in range(len(vector1))]
    else:
        compare = []
    exceptions = []
    for j in range(len(compare)):
        if not compare[j]:
            exceptions.append((j, vector1[j], vector2[j]))
            # print 'Fail @ %d %s %s' % (j, str(vector1[j]), str(vector2[j]))
    return exceptions


class ProbPara():
    def __init__(self, opt_math):
        # basic setup
        self.epsilon_bool = 1e-6
        self.epsilon_obj = 1e-4
        self.num_heats = opt_math.steel_rtn.num_heats
        self.num_groups = opt_math.steel_rtn.num_groups
        self.num_t = opt_math.steel_rtn.num_t
        self.num_x = opt_math.num_x
        self.num_y = opt_math.num_y
        self.num_tasks = opt_math.steel_rtn.num_tasks
        self.unit2num = opt_math.steel_rtn.get_unit2num()
        self.heat_sequence = opt_math.steel_rtn.heat_sequence
        self.main_process = opt_math.steel_rtn.main_process
        # self.sorted_t_list = self.get_sort_t_list()


class SteelHelper():
    epsilon_bool = 1e-4

    @staticmethod
    def draw_schedule(xx, steel_rtn, case_name):

        import json
        schedule = []
        for task in range(1, steel_rtn.num_tasks+1):
            for t_ in range(steel_rtn.num_t):
                val = xx[SteelHelper.cal_varpos_taskres_t(steel_rtn.num_t, task,t_)]
                if abs(val-1) < SteelHelper.epsilon_bool:
                    schedule.append((task, t_))
        json.dump({'schedule': schedule, 'xx': xx}, open(case_name + ' sol.json', 'w+'))

        color_map = {1:'b', 2:'g', 3: 'r', 4: 'c', 5:'m', 6:'y'}
        num_unit = 0
        for stage, units in steel_rtn.stage2units.items():
            for unit, itsnum in units.items():
                num_unit += itsnum

        occupy = [[0 for tt in range(steel_rtn.num_t)] for uu in range(num_unit+1)]

        unit2y = {'EAF':[8,7], 'AOD':[6,5], 'LF':[4,3], 'CC1':[2],'CC2':[1]}

        for task_type, task_list in steel_rtn.tasks.items():
            if 'TR_' in task_type:
                continue
            for group, heats in steel_rtn.group2heats.items():
                group = int(group)
                if 'CC' in task_type:
                    task_idx = [group-1]
                else:
                    task_idx = [heat-1 for heat in heats]
                for idx in task_idx:
                    task = task_list[idx]
                    for t_ in range(steel_rtn.num_t):
                        if abs(xx[SteelHelper.cal_varpos_taskres_t(steel_rtn.num_t, task, t_)] - 1) < 1e-5:
                            duration = steel_rtn.task_duration[task]
                            x_cod = range(t_, t_ + duration)
                            if sum([occupy[unit2y[task_type][0]][j] for j in x_cod]) == 0:
                                y_level = unit2y[task_type][0]
                            else:
                                y_level = unit2y[task_type][1]
                            for t1 in x_cod:
                                occupy[y_level][t1] += 1
                            body_x = x_cod
                            tail_x = []
                            for digit in range(1, 6):
                                tail_x.append(x_cod[-1] + float(digit)/10)
                            plt.plot(body_x, [y_level]*len(body_x), linewidth=15.0, color=color_map[group])
                            plt.plot(tail_x, [y_level]*len(tail_x), linewidth=8.0, color=color_map[group])
                            # plt.text(body_x[int(1.0/3*len(body_x))], y_level, r"%d" % (idx+1), backgroundcolor='white')
                            plt.text(max(1, body_x[0]), y_level, r"%d" % (idx+1), backgroundcolor='white')
                            # print task_type, idx+1, '[', t_+1, t_+duration, ']'
        plt.axes().set_aspect(5.0)
        frame1 = plt.gca()
        # x_ticks = [60/steel_rtn.stage2units*h for h in range(0,25)]
        # if steel_rtn.stage2units == 10:
        #     x_ticks = range(0,24*60/steel_rtn.stage2units+1, 10)
        # frame1.axes.get_xaxis().set_ticks(x_ticks)
        # plt.xlabel('Time slot [%d min]' % steel_rtn.stage2units)
        frame1.axes.get_xaxis().set_ticks([60/steel_rtn.rtn_t0*h for h in range(0, 25)])
        frame1.axes.get_xaxis().set_ticklabels(['%d' % h for h in range(0,25)])
        plt.xlabel('Hour')
        frame1.axes.get_yaxis().set_ticks([1,2,3,4,5,6,7,8])
        frame1.axes.get_yaxis().set_ticklabels(['CC2','CC1','LF2','LF1','AOD2','AOD1','EAF2','EAF1'])
        plt.xlim(0, steel_rtn.num_t)
        plt.ylim(0, num_unit+0.5)
        plt.grid(True)
        plt.show()
        plt.savefig(case_name + ' schedule')
        plt.close()

    @staticmethod
    def draw_iter(q1, q2_int, case_name):
        plt.figure(figsize=(10.24, 3.5))
        plt.plot(q1)
        for i in range(len(q2_int)):
            if q2_int[i] == float('inf'):
                q2_int[i] = 1.5 * q1[-1]
        plt.plot(q2_int)
        # plt.axes().set_aspect(0.07)
        frame1 = plt.gca()
        plt.xlabel('Iteration')
        plt.ylabel('Obj Value [$]')
        plt.grid(True)
        # plt.show()
        plt.savefig(case_name + ' iter')
        plt.close()

    @staticmethod
    def print_xx():
        print 'xx'

    @staticmethod
    def log_format_xx(xx, num_t):
        i = 0
        while i + num_t <= len(xx):
            if sum(xx[i:i + num_t]) != num_t:
                log.info(str(int(i / num_t)) + ':' + str(xx[i:i + num_t]))
            i += num_t

    @staticmethod
    def find_widest_task(considered_tasks, task_start_rng):
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
    def is_integer(xx):
        if xx == -1:
            return False
        # if node[0] == float('inf'):
        # return False
        # xx = node[1]
        for i in range(len(xx)):
            val = xx[i]
            if not (abs(val) < SteelHelper.epsilon_bool or abs(val - 1) < SteelHelper.epsilon_bool):
                return False
        return True

    @staticmethod
    def cal_varpos_taskres_t(num_t, task_or_res, t_):
        '''match between variable position and resource/task'''
        assert 0 <= t_ < num_t
        assert 1 <= task_or_res
        return (task_or_res-1)*num_t + t_

    @staticmethod
    def cal_taskres_t_varPos(varPos, num_t):
        return divmod(varPos + num_t, num_t)

    @staticmethod
    def get_x_upper_bound(task_time, num_x, num_t, rtn_tasks, rtn_casters):
        # restrictions upper bounds on this node
        x_upbound = [1] * num_x
        for task_type, tasklist in rtn_tasks.items():
            for task in tasklist:
                if task_time[task] == (-1, -1):
                    continue
                # todo elif task_time[task][0] > self.task_start_latest[task] and task_time[task][1] <= self.task_start_earlist[task]:
                else:
                    # restrict 0 for the outside
                    for t_ in range(0, num_t):
                        if t_ in range(task_time[task][0], task_time[task][1]):
                            continue
                        # the task
                        x_upbound[SteelHelper.cal_varpos_taskres_t(num_t, task, t_)] = 0
                        # todo the transfer task that follows
                        # x_upbound[self.optmath.cal_varpos_taskres_t(task+self.num_heats,t_ + transfer_duration)] = 0
                    # restrict 0 for parallel tasks
                    if 'CC' in task_type:
                        group_ = task - rtn_tasks[task_type][0]
                        for caster in rtn_casters:
                            if caster == task_type:
                                continue
                            cast_task = rtn_tasks[caster][group_]
                            for t_ in range(num_t):
                                x_upbound[SteelHelper.cal_varpos_taskres_t(num_t, cast_task, t_)] = 0
        return x_upbound


    @staticmethod
    def find_feasible_schedule(para, rtn):
        """find a feasible schedule"""
        # todo: do not work for G6
        # todo or, expand the time horizon and assign a bigM price for t>96
        # todo the following is simply a greedy alg
        # todo offset to the most cheapst hours
        # todo need test for 3EAF,2AOD,4LF,1CC

        rtn_casters = rtn.steel_rtn.casters

        task_start = [-100] + [-1] * para.num_tasks
        heat_ready_q2 = PriorityQueue()

        # get caster start asap by arranging heat priority
        # group_time_q = PriorityQueue()
        # for group_ in range(object.num_groups):
        # total_slot = 0
        # for heat in object.optmath.steel_rtn.group2heats[group_+1]:
        #         for s in object.process_sequence:
        #             total_slot += object.optmath.steel_rtn.task_length[object.optmath.steel_rtn.tasks[s][heat-1]]
        #     total_slot += max([object.optmath.steel_rtn.task_cleanup_length[object.optmath.steel_rtn.tasks[unit][group_]] for unit in casters])
        #     group_time_q.put_nowait((total_slot,group_+1))
        # equip_count = 0
        # num_eaf = object.optmath.steel_rtn.stage2units['1']['EAF']
        # while not group_time_q.empty():
        #     (total_slot, group) = group_time_q.get_nowait()
        #     for heat in object.optmath.steel_rtn.group2heats[group]:
        #         heat_ready_q2.put_nowait((math.floor(equip_count/num_eaf),heat-1))
        #         equip_count += 1

        for heat_ in range(para.num_heats):
            heat_ready_q2.put_nowait((0, heat_))
        for seq in [0, 2, 4]:
            heat_ready_q = heat_ready_q2
            heat_ready_q2 = PriorityQueue()
            task_type = para.heat_sequence[seq]
            equip_time = [0] * para.unit2num[task_type]
            equip_id = 0
            while not heat_ready_q.empty():
                (ready_t, heat_) = heat_ready_q.get_nowait()
                task = rtn.steel_rtn.tasks[task_type][heat_]
                equip_time[equip_id] = max(equip_time[equip_id], ready_t)
                task_start[task] = equip_time[equip_id]
                equip_time[equip_id] += rtn.steel_rtn.task_length[task]
                trans_time = rtn.steel_rtn.task_length[task + para.num_heats]
                heat_ready_q2.put_nowait((equip_time[equip_id] + trans_time, heat_))
                equip_id = (equip_id + 1) % len(equip_time)
        # casting
        heat_ready_list = [-1] * para.num_heats
        while not heat_ready_q2.empty():
            (read_t, heat_) = heat_ready_q2.get_nowait()
            heat_ready_list[heat_] = read_t
        group_ready_t = dict()
        caster_time = [0] * len(rtn_casters)
        for caster in rtn_casters:
            group_ready_t[caster] = PriorityQueue()
        # group ready time
        for group_ in range(para.num_groups):
            heats = rtn.steel_rtn.group2heats[group_ + 1]
            for caster in rtn_casters:
                read_t = [heat_ready_list[heat - 1] - rtn.steel_rtn.cast_heat_rel_slot[caster][heat] for heat
                          in heats]
                group_ready_t[caster].put_nowait((max(read_t), group_))
        scheduled_groups = []
        while len(scheduled_groups) < para.num_groups:
            caster_id = np.argmin(caster_time)
            (read_t, group_) = group_ready_t[rtn_casters[caster_id]].get_nowait()
            while group_ in scheduled_groups:
                (read_t, group_) = group_ready_t[rtn_casters[caster_id]].get_nowait()
            scheduled_groups.append(group_)
            schedule_time = max(read_t, caster_time[caster_id])
            schedule_task = rtn.steel_rtn.tasks[rtn_casters[caster_id]][group_]
            caster_time[caster_id] = schedule_time + rtn.steel_rtn.task_cleanup_length[schedule_task]
            task_start[schedule_task] = schedule_time

        task_time = [(-100, -100)] + [(-1, -1)] * rtn.steel_rtn.num_tasks  # task counts from 1
        for task in range(1, para.num_tasks + 1):
            if task_start[task] < 0:
                continue
            task_time[task] = (task_start[task], task_start[task] + 1)
        return task_time


    @staticmethod
    def cal_task_start_range(task, task_start_earliest, task_start_latest):
        return range(task_start_earliest[task], task_start_latest[task] + 1)

    @staticmethod
    def cal_alltask_start_bounds(para, opt_math_model):
        """
        # calculate up/low bounds [] for starting times
        # start time for tasks, -1 means not assigned - the corresponding x_Nit is relaxed
        """
        # todo need test
        # todo consider the casting here
        task_start_earliest = [-100] + [-1] * opt_math_model.steel_rtn.num_tasks
        task_start_latest = [-100] + [-1] * opt_math_model.steel_rtn.num_tasks
        for group in range(1, para.num_groups + 1):
            for heat in opt_math_model.steel_rtn.group2heats[group]:
                EAF = opt_math_model.steel_rtn.tasks['EAF'][heat - 1]
                EA = opt_math_model.steel_rtn.tasks['Trans_EA'][heat - 1]
                AOD = opt_math_model.steel_rtn.tasks['AOD'][heat - 1]
                AL = opt_math_model.steel_rtn.tasks['Trans_AL'][heat - 1]
                LF = opt_math_model.steel_rtn.tasks['LF'][heat - 1]
                LC = opt_math_model.steel_rtn.tasks['Trans_LC'][heat - 1]
                task_start_earliest[EAF] = 0
                task_start_earliest[EA] = sum([opt_math_model.steel_rtn.task_length[i] for i in [EAF]])
                task_start_earliest[AOD] = sum([opt_math_model.steel_rtn.task_length[i] for i in [EAF, EA]])
                task_start_earliest[AL] = sum([opt_math_model.steel_rtn.task_length[i] for i in [EAF, EA, AOD]])
                task_start_earliest[LF] = sum([opt_math_model.steel_rtn.task_length[i] for i in [EAF, EA, AOD, AL]])
                task_start_earliest[LC] = sum([opt_math_model.steel_rtn.task_length[i] for i in [EAF, EA, AOD, AL, LF]])
                task_start_latest[EAF] = para.num_t - sum(
                    [opt_math_model.steel_rtn.task_length[i] for i in [EAF, EA, AOD, AL, LF, LC]])
                task_start_latest[EA] = para.num_t - sum(
                    [opt_math_model.steel_rtn.task_length[i] for i in [EA, AOD, AL, LF, LC]])
                task_start_latest[AOD] = para.num_t - sum(
                    [opt_math_model.steel_rtn.task_length[i] for i in [AOD, AL, LF, LC]])
                task_start_latest[AL] = para.num_t - sum(
                    [opt_math_model.steel_rtn.task_length[i] for i in [AL, LF, LC]])
                task_start_latest[LF] = para.num_t - sum([opt_math_model.steel_rtn.task_length[i] for i in [LF, LC]])
                task_start_latest[LC] = para.num_t - sum([opt_math_model.steel_rtn.task_length[i] for i in [LC]])
                if heat == opt_math_model.steel_rtn.group2heats[group][0]:  # if first heat in group
                    for cast_task in [opt_math_model.steel_rtn.tasks[caster][group - 1] for caster in
                                      opt_math_model.steel_rtn.stage2units['4'].keys()]:
                        task_start_earliest[cast_task] = sum(
                            [opt_math_model.steel_rtn.task_length[i] for i in [EAF, EA, AOD, AL, LF, LC]])
                        task_start_latest[cast_task] = para.num_t - opt_math_model.steel_rtn.task_length[cast_task]
        return task_start_earliest, task_start_latest

