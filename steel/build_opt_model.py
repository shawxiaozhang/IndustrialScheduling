""" Builds the optimal scheduling model for steel plant.

Build the matrix blocks for the scheduling optimization.
    blocks divided according to:
    variable    -   binary variable x for tasks, and continuous variable y for resources
    constraint  -   different types of constraints: resource balance, task execution, ...
"""

import math

from scipy.sparse import coo_matrix


class OptModelBuilder():
    def __init__(self, steel_rtn, options=None):
        # todo delete this
        self.steel_rtn = steel_rtn

        self.num_t = steel_rtn.num_t
        self.num_tasks = steel_rtn.num_tasks
        self.num_resources = steel_rtn.num_resources

        self.num_x = steel_rtn.num_tasks * steel_rtn.num_t
        self.num_y = steel_rtn.num_resources * steel_rtn.num_t

        # objective function
        self.obj_x, self.obj_y = self.build_obj_f(steel_rtn, options)

        # continuous variables bounds
        self.y_lo, self.y_up = self.build_res_bounds(steel_rtn)

        # binary variables bounds
        if options and 'heuristic_all' in options and options['heuristic_all']:
            self.x_lo, self.x_up = self.build_binary_bounds2(options)
        else:
            self.x_lo, self.x_up = self.build_binary_bounds(options)

        # resource balance constraints
        self.con_res_x, self.con_res_y, self.con_res_b = self.build_con_res(steel_rtn)

        # energy consumption constraints
        self.con_en_x, self.con_en_y, self.con_en_b = self.build_con_en(steel_rtn)

        # task execution constraints
        self.con_exe_x, self.con_exe_b = self.build_con_task(steel_rtn)

        # max waiting time constraints
        self.con_wait_y, self.con_wait_b = self.build_con_wait(steel_rtn)

        self.num_con = self.con_res_x.shape[0] + self.con_en_y.shape[0] + self.con_exe_x.shape[0] + self.con_wait_y.shape[0]

        # task execution sequence chain
        if options and 'task_order' in options and options['task_order']:
            self.con_exe_order_x, self.con_exe_order_b = self.build_con_exe_order(steel_rtn, options)
            self.num_con += self.con_exe_order_x.shape[0]

    def pos_x_task_t(self, task, t):
        assert 0 <= t < self.num_t
        assert 1 <= task <= self.num_tasks
        return (task - 1) * self.num_t + t

    def pos_y_res_t(self, res, t):
        assert 0 <= t < self.num_t
        assert 1 <= res <= self.num_resources
        return (res - 1) * self.num_t + t

    def inv_pos_x(self, pos):
        task, t = divmod(pos, self.num_t)
        return task+1, t

    def inv_pos_y(self, pos):
        res, t = divmod(pos, self.num_t)
        return res+1, t

    def build_obj_f(self, steel_rtn, options):
        if options is not None and 'obj_f_type' in options:
            if options['obj_f_type'] == 'MAKE_SPAN':
                return self._obj_make_span(steel_rtn)
        return self._obj_energy_cost(steel_rtn)

    def _obj_energy_cost(self, steel_rtn):
        f_x = [0] * steel_rtn.num_t * steel_rtn.num_tasks  # binary variable part
        f_y = [0] * steel_rtn.num_t * steel_rtn.num_resources  # continuous variable part
        for t in range(steel_rtn.num_t):
            pos = self.pos_y_res_t(steel_rtn.resources['EN'][0], t)
            f_y[pos] = steel_rtn.price_energy[t]
        return [f_x, f_y]

    def _obj_make_span(self, steel_rtn):
        f_x = [0] * steel_rtn.num_t * steel_rtn.num_tasks  # binary variable part
        f_y = [0] * steel_rtn.num_t * steel_rtn.num_resources  # continuous variable part
        for task_cat, task_list in steel_rtn.tasks.items():
            for task in task_list:
                for t_ in range(steel_rtn.num_t):
                    pos = self.pos_x_task_t(task, t_)
                    f_x[pos] = t_
        return [f_x, f_y]

    def build_binary_bounds(self, options):
        if options is not None:
            if 'heuristic_eaf' in options and options['heuristic_eaf']:
                # heuristic restriction on starting time for EAF task
                # todo may need a better ordered group chain
                group_chain = range(1, self.steel_rtn.num_groups+1)
                # group_chain = [3, 4, 5, 2, 1]   # for G5
                casters = self.steel_rtn.stage2units['4'].keys()
                num_caster = sum(self.steel_rtn.stage2units['4'].values())
                caster_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][g]] for u in casters])
                                   for g in range(self.steel_rtn.num_groups)]
                eafs = self.steel_rtn.stage2units['1'].keys()
                num_eaf = sum(self.steel_rtn.stage2units['1'].values())
                eaf_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][h]] for u in eafs])
                                for h in range(self.steel_rtn.num_heats)]
                lfs = self.steel_rtn.stage2units['3'].keys()
                num_lf = sum(self.steel_rtn.stage2units['3'].values())
                lf_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][h]] for u in lfs])
                               for h in range(self.steel_rtn.num_heats)]
                aods = self.steel_rtn.stage2units['2'].keys()
                num_aod = sum(self.steel_rtn.stage2units['2'].values())
                aod_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][h]] for u in aods])
                                for h in range(self.steel_rtn.num_heats)]
                tr_min_time = [0]*3
                if 'TR_S1' in self.steel_rtn.tasks:
                    # model RTN1
                    tr_min_time[0] = self.steel_rtn.task_duration[self.steel_rtn.tasks['TR_S1'][0]]
                    tr_min_time[1] = self.steel_rtn.task_duration[self.steel_rtn.tasks['TR_S2'][0]]
                    tr_min_time[2] = self.steel_rtn.task_duration[self.steel_rtn.tasks['TR_S3'][0]]
                else:
                    # model RTN2
                    for stage in [1, 2, 3]:
                        transfers = ['TR_%s_%s' % (u1, u2) for u1 in self.steel_rtn.stage2units[str(stage)].keys()
                                     for u2 in self.steel_rtn.stage2units[str(stage+1)].keys()]
                        tr_min_time[stage-1] = min([self.steel_rtn.task_duration[self.steel_rtn.tasks[tr][0]]
                                                    for tr in transfers])
                heat2times = dict()
                for i in range(len(group_chain)):
                    group = group_chain[i]
                    num_sub_groups = int((len(group_chain)-i-1)/num_caster)*num_caster
                    sub_groups = [group_chain[-j-1] for j in range(num_sub_groups)]
                    t2_base = self.num_t-1
                    for j in range(0, num_sub_groups, num_caster):
                        t2_base -= min(caster_min_time[-j-num_caster-1:-j-1])
                    t2_base -= tr_min_time[2]
                    heat_1 = self.steel_rtn.group2heats[str(group)][0]
                    t2_base -= lf_min_time[heat_1-1]
                    t2_base -= tr_min_time[1]
                    t2_base -= aod_min_time[heat_1-1]
                    t2_base -= tr_min_time[0]
                    pre_groups = [group_chain[j] for j in range(i)]
                    pre_heats = [h for g in pre_groups for h in self.steel_rtn.group2heats[str(g)]]
                    pre_eaf_times = [eaf_min_time[h-1] for h in pre_heats]
                    pre_eaf_times = sorted(pre_eaf_times)
                    num_pre_heats = int(len(pre_heats)/num_eaf)*num_eaf
                    t1_base = 0
                    for j in range(0, num_pre_heats, num_eaf):
                        t1_base += min(pre_eaf_times[j:j+num_eaf])
                    for heat in self.steel_rtn.group2heats[str(group)]:
                        t2 = t2_base
                        t1 = t1_base
                        heat2times[heat] = (t1, t2)
                print heat2times
                # set upper bounds
                x_up = [1]*self.num_x
                for heat, (t1, t2) in heat2times.items():
                    for task in [self.steel_rtn.tasks[u][heat-1] for u in eafs]:
                        for t in range(t1) + range(t2, self.num_t):
                            x_up[self.pos_x_task_t(task, t)] = 0
                return [0]*self.num_x, x_up
            if 'operation_time_slots' in options:
                x_up = [0]*self.num_x
                for t in options['operation_time_slots']:
                    for task in range(1, self.num_tasks+1):
                        x_up[self.pos_x_task_t(task, t)] = 1
                return [0]*self.num_x, x_up
        return [0]*self.num_x, [1]*self.num_x

    def build_binary_bounds2(self, options):
        if options is not None:
            if 'heuristic_eaf' in options and options['heuristic_eaf']:
                # heuristic restriction on starting time for all stages' task
                group_chain = range(1, self.steel_rtn.num_groups+1)
                casters = self.steel_rtn.stage2units['4'].keys()
                num_caster = sum(self.steel_rtn.stage2units['4'].values())
                caster_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][group]] for u in casters])
                                   for group in range(self.steel_rtn.num_groups)]
                eafs = self.steel_rtn.stage2units['1'].keys()
                num_eaf = sum(self.steel_rtn.stage2units['1'].values())
                eaf_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][h]] for u in eafs])
                                for h in range(self.steel_rtn.num_heats)]
                lfs = self.steel_rtn.stage2units['3'].keys()
                num_lf = sum(self.steel_rtn.stage2units['3'].values())
                lf_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][h]] for u in lfs])
                               for h in range(self.steel_rtn.num_heats)]
                aods = self.steel_rtn.stage2units['2'].keys()
                num_aod = sum(self.steel_rtn.stage2units['2'].values())
                aod_min_time = [min([self.steel_rtn.task_duration[self.steel_rtn.tasks[u][h]] for u in aods])
                                for h in range(self.steel_rtn.num_heats)]
                process_min_time = {'s1': eaf_min_time, 's2': aod_min_time, 's3': lf_min_time, 's4': caster_min_time}
                stage_units_num = {'s1': num_eaf, 's2': num_aod, 's3': num_lf, 's4': num_caster}
                tr_min_time = dict()
                if 'TR_S1' in self.steel_rtn.tasks:
                    # model RTN1
                    tr_min_time['s1'] = self.steel_rtn.task_duration[self.steel_rtn.tasks['TR_S1'][0]]
                    tr_min_time['s2'] = self.steel_rtn.task_duration[self.steel_rtn.tasks['TR_S2'][0]]
                    tr_min_time['s3'] = self.steel_rtn.task_duration[self.steel_rtn.tasks['TR_S3'][0]]
                else:
                    # model RTN2
                    for stage in [1, 2, 3]:
                        transfers = ['TR_%s_%s' % (u1, u2) for u1 in self.steel_rtn.stage2units[str(stage)].keys()
                                     for u2 in self.steel_rtn.stage2units[str(stage+1)].keys()]
                        tr_min_time['s%d' % stage] = min([self.steel_rtn.task_duration[self.steel_rtn.tasks[tr][0]]
                                                         for tr in transfers])
                stage2heat2times = dict()
                for stage in [1, 2, 3, 4]:
                    stage2heat2times['%d' % stage] = dict()
                    for i in range(len(group_chain)):
                        group = group_chain[i]
                        pre_groups = [group_chain[j] for j in range(i)]
                        pre_heats = [h for gg in pre_groups for h in self.steel_rtn.group2heats[str(gg)]]
                        sub_groups = [group_chain[j] for j in range(i+1, len(group_chain))]
                        sub_heats = [h for gg in sub_groups for h in self.steel_rtn.group2heats[str(gg)]]
                        t2 = self.num_t-1
                        t1 = 0
                        # caster applies to all t2
                        for j in range(0, len(sub_groups)+1-num_caster, num_caster):
                            t2 -= min(caster_min_time[-j-num_caster-1:-j-1])
                        for s in range(stage, 4):
                            t2 -= tr_min_time['s%d' % s]
                            t2 -= process_min_time['s%d' % s][self.steel_rtn.group2heats[str(group)][-1]-1]
                        # eaf applies to all t1
                        for j in range(0, len(pre_heats)+1-num_eaf, num_eaf):
                            t1 += min(eaf_min_time[j:j+num_eaf])
                        for s in range(1, stage+1):
                            if 1 < s:
                                t1 += tr_min_time['s%d' % (s-1)]
                            if 1 < s < 4:
                                t1 += process_min_time['s%d' % s][self.steel_rtn.group2heats[str(group)][0]-1]
                        if stage != 4:
                            for heat in self.steel_rtn.group2heats[str(group)]:
                                stage2heat2times['%d' % stage][heat] = (t1, t2)
                        else:
                            stage2heat2times['%d' % stage][group] = (t1, t2)
                for stage, heat2times in stage2heat2times.items():
                    print stage, heat2times
                # set upper bounds
                x_up = [1]*self.num_x
                for s, heat2times in stage2heat2times.items():
                    for heat, (t1, t2) in heat2times.items():
                        assert t1 < t2
                        for task in [self.steel_rtn.tasks[u][heat-1] for u in self.steel_rtn.stage2units[s].keys()]:
                            for t in range(t1) + range(t2, self.num_t):
                                x_up[self.pos_x_task_t(task, t)] = 0
                return [0]*self.num_x, x_up
            if 'operation_time_slots' in options:
                x_up = [0]*self.num_x
                for t in options['operation_time_slots']:
                    for task in range(1, self.num_tasks+1):
                        x_up[self.pos_x_task_t(task, t)] = 1
                return [0]*self.num_x, x_up
        return [0]*self.num_x, [1]*self.num_x

    def build_res_bounds(self, steel_rtn):
        # default bounds, as all heats resources are bounded by 0,1
        y_lo = [0] * (steel_rtn.num_resources * steel_rtn.num_t)
        y_up = [1] * (steel_rtn.num_resources * steel_rtn.num_t)
        # equipment resources
        for stage, units in steel_rtn.stage2units.items():
            for unit, num in units.items():
                res_idx = steel_rtn.resources[unit][0]
                for t_ in range(steel_rtn.num_t):
                    y_up[self.pos_y_res_t(res_idx, t_)] = num
        # heat resources
        for res_cat, res_list in steel_rtn.resources.items():
            # todo unify RTN1 and RTN2
            if 'H_FINAL' in res_cat or 'H_A_S4' in res_cat:  # final products should be available at last time slot
                for res_idx in res_list:
                    y_lo[self.pos_y_res_t(res_idx, steel_rtn.num_t - 1)] = 1
            elif 'H_A_' in res_cat:  # intermediate products transfer immediately
                for res_idx in res_list:
                    for t_ in range(steel_rtn.num_t):
                        y_up[self.pos_y_res_t(res_idx, t_)] = 0
        # energy
        res_idx = steel_rtn.resources['EN'][0]
        for t_ in range(steel_rtn.num_t):
            # needs a little extra space
            y_up[self.pos_y_res_t(res_idx, t_)] = steel_rtn.maxPower * steel_rtn.rtn_t0 / 60 * 1.0001
        return [y_lo, y_up]

    def build_con_res(self, steel_rtn):
        res_list = [res for cat, cat_res in steel_rtn.resources.items() if cat != 'EN' for res in cat_res]
        task_list = [task for cat, cat_task in steel_rtn.tasks.items() for task in cat_task]
        row_num = steel_rtn.num_t * len(res_list)

        x_vals = []
        x_rows = []
        x_cols = []
        y_vals = []
        y_rows = []
        y_cols = []
        b_rhs = [0] * row_num

        row_ = 0
        for res in res_list:
            for t_ in range(0, steel_rtn.num_t):
                y_rows.append(row_)
                y_cols.append(self.pos_y_res_t(res, t_))
                y_vals.append(-1)
                if t_ >= 1:
                    y_rows.append(row_)
                    y_cols.append(self.pos_y_res_t(res, t_ - 1))
                    y_vals.append(1)
                else:
                    b_rhs[row_] += - steel_rtn.cal_initial_resource(res)
                for task in task_list:
                    if task not in steel_rtn.rtn_profile[res]:
                        continue
                    duration = len(steel_rtn.rtn_profile[res][task]) - 1
                    for theta in range(0, min(duration, t_) + 1):
                        x_rows.append(row_)
                        x_cols.append(self.pos_x_task_t(task, t_ - theta))
                        x_vals.append(steel_rtn.rtn_profile[res][task][theta])
                row_ += 1
        assert(row_num-1 == max(x_rows))
        return [coo_matrix((x_vals, (x_rows, x_cols)), shape=(row_num, self.num_x)),
                coo_matrix((y_vals, (y_rows, y_cols)), shape=(row_num, self.num_y)),
                b_rhs]

    def build_con_en(self, steel_rtn):
        """constraints on energy summation"""
        res = steel_rtn.resources['EN'][0]
        row_num = steel_rtn.num_t
        x_vals = []
        x_cols = []
        x_rows = []
        for row_t in range(steel_rtn.num_t):
            for task_cat, task_list in steel_rtn.tasks.items():
                if 'TR' in task_cat:  # transportation has no energy consumption
                    continue
                for task in task_list:
                    duration = steel_rtn.task_duration[task]
                    for theta in range(0, min(duration, row_t) + 1):
                        x_rows.append(row_t)
                        x_cols.append(self.pos_x_task_t(task, row_t - theta))
                        x_vals.append(steel_rtn.rtn_profile[res][task][theta])
        y_vals = []
        y_rows = []
        y_cols = []
        for row_t in range(steel_rtn.num_t):
            y_rows.append(row_t)
            y_cols.append(self.pos_y_res_t(res, row_t))
            y_vals.append(-1)
        b_rhs = [0]*row_num
        assert(row_num-1 == max(x_rows))
        return [coo_matrix((x_vals, (x_rows, x_cols)), shape=(row_num, self.num_x)),
                coo_matrix((y_vals, (y_rows, y_cols)), shape=(row_num, self.num_y)),
                b_rhs]

    def build_con_wait(self, steel_rtn):
        y_vals = []
        y_rows = []
        y_cols = []
        b_rhs = []
        row = 0
        for res_cat, resources in steel_rtn.resources.items():
            if 'H_B_' not in res_cat:
                continue
            max_wait_time = steel_rtn.time_trans_max['TR_S%d' % (int(res_cat[-1]) - 1)]
            min_tran_time = steel_rtn.time_trans['TR_S%d' % (int(res_cat[-1]) - 1)]
            for res in resources:   # each heat a constraint
                b_rhs.append(math.ceil((max_wait_time-min_tran_time)/steel_rtn.rtn_t0))
                for t in range(steel_rtn.num_t):
                    y_rows.append(row)
                    y_cols.append(self.pos_y_res_t(res, t))
                    y_vals.append(1)
                row += 1
        row_num = row
        assert(row_num-1 == max(y_rows))
        assert(row_num == len(b_rhs))
        return [coo_matrix((y_vals, (y_rows, y_cols)), shape=(row_num, self.num_y)), b_rhs]

    def build_con_task(self, steel_rtn):
        """constraints for task execution"""
        # works for RTN1
        heat2tasks = [task for task_cat, task_list in steel_rtn.tasks.items()
                      if 'CC' not in task_cat for task in task_list]
        group2tasks = dict()
        for group in range(1, steel_rtn.num_groups + 1):
            group2tasks[group] = [steel_rtn.tasks[caster][group - 1] for caster in
                                  steel_rtn.stage2units['4'].keys()]
        x_vals = []
        x_cols = []
        x_rows = []
        row = 0
        # general task, execute once for each heat
        for task in heat2tasks:
            for t in range(steel_rtn.num_t):
                x_rows.append(row)
                x_cols.append(self.pos_x_task_t(task, t))
                x_vals.append(1)
            row += 1
        # casting task, execute once for each group
        for group in range(1, steel_rtn.num_groups + 1):  # todo, range of groups
            for task in group2tasks[group]:
                for t in range(steel_rtn.num_t):
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task, t))
                    x_vals.append(1)
            row += 1
        row_num = row
        b_rhs = [1] * row_num
        assert(row_num-1 == max(x_rows))
        return [coo_matrix((x_vals, (x_rows, x_cols)), shape=(row_num, self.num_x)),
                b_rhs]

    def build_con_exe_order(self, steel_rtn, options):
        max_wait_task_ordering = []
        # accurate transfer waiting time constraint for intermediate product
        if 'accurate_wait' in options and options['accurate_wait']:
            for stage in range(1, 3):   # todo consider cast stage
                for unit2 in steel_rtn.stage2units[str(stage+1)].keys():
                    max_wait_slots = steel_rtn.time_trans_max['TR_S%s' % stage]/steel_rtn.rtn_t0
                    for heat in range(steel_rtn.num_heats):
                        task2 = steel_rtn.tasks[unit2][heat]
                        transfers = ['TR_%s_%s' % (unit1, unit2) for unit1 in steel_rtn.stage2units[str(stage)].keys()]
                        task1s = [steel_rtn.tasks[transfer][heat] for transfer in transfers]
                        # todo consider the group task in casting stage
                        max_wait_task_ordering.append((task1s, task2, max_wait_slots))
        ordered_tasks = []
        if 'impose_group_order_stages' in options and len(options['impose_group_order_stages']) > 0:
            for stage in options['impose_group_order_stages']:
                units = steel_rtn.stage2units[str(stage)]
                # todo to manually set the group ordering
                for group in range(1, steel_rtn.num_groups):
                    heat_a = steel_rtn.group2heats[str(group)][-1]-1
                    heat_b = steel_rtn.group2heats[str(group+1)][0]-1
                    ordered_tasks.append(([steel_rtn.tasks[unit][heat_a] for unit in units],
                                          [steel_rtn.tasks[unit][heat_b] for unit in units]))
        tasks_order_pair = []
        if 'lock_heats_stage' in options and len(options['lock_heats_stage']) > 0:
            extra_slots = options['lock_heats_stage_extra_slots']
            for stage in options['lock_heats_stage']:
                for heats in steel_rtn.group2heats.values():
                    for heat in heats[:-1]:
                        task1s = [steel_rtn.tasks[u][heat-1] for u in steel_rtn.stage2units[str(stage)].keys()]
                        task2s = [steel_rtn.tasks[u][heat] for u in steel_rtn.stage2units[str(stage)].keys()]
                        max_delay_slots = max([steel_rtn.task_duration[task] for task in task1s]) + extra_slots
                        tasks_order_pair.append((task1s, task2s, max_delay_slots))
        # model ordering in constraints
        x_vals = []
        x_cols = []
        x_rows = []
        row = 0
        b_rhs = []
        # max wait constraints
        for (task1s, task2, max_wait_slots) in max_wait_task_ordering:
            # t2 <= t1 + delay
            for t in range(self.num_t):
                x_rows.append(row)
                x_cols.append(self.pos_x_task_t(task2, t))
                x_vals.append(t)
                for task1 in task1s:
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task1, t))
                    x_vals.append(-max_wait_slots-t)
            b_rhs.append(0)
            row += 1
        for (task1s, task2s) in ordered_tasks:
            # t1 <= t2
            for t in range(self.num_t):
                for task1 in task1s:
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task1, t))
                    x_vals.append(t)
                for task2 in task2s:
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task2, t))
                    x_vals.append(-t)
            b_rhs.append(0)
            row += 1
        # general task process order imposing
        for (task1s, task2s, max_delay_slots) in tasks_order_pair:
            # t2 - t1 <= delay
            for t in range(self.num_t):
                for task2 in task2s:
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task2, t))
                    x_vals.append(t)
                for task1 in task1s:
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task1, t))
                    x_vals.append(-t)
            b_rhs.append(max_delay_slots)
            row += 1
        row_num = row
        if row_num > 0:
            assert(row_num-1 == max(x_rows))
        return [coo_matrix((x_vals, (x_rows, x_cols)), shape=(row_num, self.num_x)),
                b_rhs]


class OptModelBuilderRTN2(OptModelBuilder):
    def build_con_wait(self, steel_rtn):
        # todo to accurately model the waiting time, can we add another resource as a time counter?
        y_vals = []
        y_rows = []
        y_cols = []
        b_rhs = []
        row = 0
        for stage in [2, 3, 4]:
            units = steel_rtn.stage2units[str(stage)].keys()
            units1 = steel_rtn.stage2units[str(stage-1)].keys()
            max_wait_time = steel_rtn.time_trans_max['TR_S%d' % (stage-1)]
            min_trans_time = min([steel_rtn.time_trans['%s-%s' % (u1, u)] for u in units for u1 in units1])
            for heat in range(steel_rtn.num_heats):
                b_rhs.append(math.ceil((max_wait_time-min_trans_time)/steel_rtn.rtn_t0))
                for res in [steel_rtn.resources['H_B_%s' % u][heat] for u in units]:
                    for t in range(steel_rtn.num_t):
                        y_rows.append(row)
                        y_cols.append(self.pos_y_res_t(res, t))
                        y_vals.append(1)
                row += 1
        row_num = row
        assert(row_num-1 == max(y_rows))
        assert(row_num == len(b_rhs))
        return [coo_matrix((y_vals, (y_rows, y_cols)), shape=(row_num, self.num_y)), b_rhs]

    def build_con_task(self, steel_rtn):
        """constraints for task execution"""
        print 'consider task execution constraint for transfer task'
        process2tasks = dict()
        # process of first three stages
        for heat in range(1, steel_rtn.num_heats + 1):
            for stage in range(1, steel_rtn.num_stage):
                process2tasks['S%d_H%d' % (stage, heat)] = \
                    [steel_rtn.tasks[unit][heat - 1] for unit in steel_rtn.stage2units[str(stage)].keys()]
        # process of casting stage
        for group in range(1, steel_rtn.num_groups + 1):
            process2tasks['S4_G%d' % group] = \
                [steel_rtn.tasks[caster][group - 1] for caster in steel_rtn.stage2units['4'].keys()]
        # process of transfer
        for heat in range(1, steel_rtn.num_heats + 1):
            for stage in range(1, steel_rtn.num_stage):
                transfer_cats = ['TR_%s_%s' % (unit, unit_2) for unit in steel_rtn.stage2units[str(stage)].keys()
                                 for unit_2 in steel_rtn.stage2units[str(stage+1)].keys()]
                process2tasks['TR_S%d_S%d_H%d' % (stage, stage+1, heat)] = \
                    [steel_rtn.tasks[transfer][heat - 1] for transfer in transfer_cats]
        # constraints: execution once over time
        x_vals = []
        x_cols = []
        x_rows = []
        row = 0
        # general task, execute once for each heat
        for process, tasks in process2tasks.items():
            for task in tasks:
                for t in range(steel_rtn.num_t):
                    x_rows.append(row)
                    x_cols.append(self.pos_x_task_t(task, t))
                    x_vals.append(1)
            row += 1
        row_num = row
        b_rhs = [1] * row_num
        assert(row_num-1 == max(x_rows))
        return [coo_matrix((x_vals, (x_rows, x_cols)), shape=(row_num, self.num_x)),
                b_rhs]
