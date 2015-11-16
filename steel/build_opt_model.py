""" Builds the optimal scheduling model for steel plant.

Build the matrix blocks for the scheduling optimization.
    blocks divided according to:
    variable    -   binary variable x for tasks, and continuous variable y for resources
    constraint  -   different types of constraints: resource balance, task execution, ...
"""

import math

from scipy.sparse import coo_matrix


class OptModelBuilder():
    def __init__(self, steel_rtn, obj_f_type='ENERGY_COST'):
        # todo delete this
        self.steel_rtn = steel_rtn

        self.num_t = steel_rtn.num_t
        self.num_tasks = steel_rtn.num_tasks
        self.num_resources = steel_rtn.num_resources

        self.num_x = steel_rtn.num_tasks * steel_rtn.num_t
        self.num_y = steel_rtn.num_resources * steel_rtn.num_t

        # objective function
        self.obj_x, self.obj_y = self.build_obj_f(steel_rtn, obj_f_type)

        # continuous variables bounds
        self.y_lo, self.y_up = self.build_res_bounds(steel_rtn)

        # resource balance constraints
        self.con_res_x, self.con_res_y, self.con_res_b = self.build_con_res(steel_rtn)

        # energy consumption constraints
        self.con_en_x, self.con_en_y, self.con_en_b = self.build_con_en(steel_rtn)

        # task execution constraints
        self.con_exe_x, self.con_exe_b = self.build_con_task(steel_rtn)

        # max waiting time constraints
        self.con_wait_y, self.con_wait_b = self.build_con_wait(steel_rtn)

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

    def build_obj_f(self, steel_rtn, obj_f_type):
        if obj_f_type == 'MAKE_SPAN':
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
            if 'H_A_S4' in res_cat:  # final products should be available at last time slot
                for res_idx in res_list:
                    y_lo[self.pos_y_res_t(res_idx, steel_rtn.num_t - 1)] = 1
            elif 'H_A_S' in res_cat:  # intermediate products transfer immediately
                for res_idx in res_list:
                    for t_ in range(steel_rtn.num_t):
                        y_up[self.pos_y_res_t(res_idx, t_)] = 0
        # energy
        res_idx = steel_rtn.resources['EN'][0]
        for t_ in range(steel_rtn.num_t):
            y_up[self.pos_y_res_t(res_idx, t_)] = steel_rtn.maxPower * steel_rtn.rtn_t0 / 6
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
            if 'H_B_S' not in res_cat:
                continue
            max_time = steel_rtn.time_trans_max['TR_S%d' % (int(res_cat[-1]) - 1)]
            for res in resources:
                b_rhs.append(math.ceil(max_time / steel_rtn.rtn_t0))
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
        for group in range(1, steel_rtn.num_groups + 1):  #todo, range of groups
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
