"""

__author__ = xxxzhang
"""

from Queue import PriorityQueue


def find_feasible_task_rng(plant):
    return _convert_schedule_to_task_rng(search_schedule(plant))


def find_feasible_xy(plant, model):
    return _convert_schedule_to_xy(search_schedule(plant), model, plant)


def search_schedule(plant):
    # todo AI scheduling alg can base on this: heat priorities in q, caster priorities
    # todo put index to the heats, delete heat+1, group+1
    # todo AI: find a fastest schedule, then cal the adjustment room in time, then shift the EAF tasks
    # todo AI: exchange the groups orders, see what happens with the final ECost
    # todo aim for a ChemE journal paper
    # currently, heat/group starting at 1, time starting at 0
    """Find a feasible schedule by matching heats and equipment units.

    Main Idea:
        Match available heats (intermediate products) to available equipment units.
        Maintain availability by priority queues (priority index: available time).
    """
    schedule = []
    q_t_h = PriorityQueue()   # ready time (priority), heat
    for heat in range(1, plant.num_heats+1):
        q_t_h.put((0, heat))
    for stage in [1, 2, 3]:
        q_t_u = PriorityQueue()     # ready time (priority), unit
        for unit, num in plant.stage2units[str(stage)].items():
            for u_id in range(num):
                q_t_u.put((0, unit, u_id))
        q_t_h_next = PriorityQueue()    # ready time of heat for next stage
        while not q_t_h.empty():
            t_heat, heat = q_t_h.get()
            t_unit, unit, u_id = q_t_u.get()
            t_start = max(t_heat, t_unit)
            task = plant.tasks[unit][heat-1]
            t_end_process = t_start + plant.task_duration[task]
            q_t_u.put((t_end_process, unit, u_id))
            # todo, RTN2
            transport = plant.tasks['TR_S%d' % stage][heat-1]
            t_end_trans = t_end_process + plant.task_duration[transport]
            q_t_h_next.put((t_end_trans, heat))
            schedule.append((stage, heat, unit, u_id, task, t_start, t_end_process, t_end_process))
            schedule.append((stage, heat, 'TR_S%d' % stage, 0, transport, t_end_process, t_end_trans, t_end_trans))
        q_t_h = q_t_h_next
    heat2time = dict()
    while not q_t_h.empty():
        t, heat = q_t_h.get()
        heat2time[heat] = t
    q_t_caster = PriorityQueue()     # ready time (priority), unit
    for unit, num in plant.stage2units['4'].items():
        for u_id in range(num):
            q_t_caster.put((0, unit, u_id))
    for group in range(1, plant.num_groups+1):
        t_caster, caster, caster_id = q_t_caster.get()
        t_group = max([heat2time[heat] - plant.heat_consume_time_in_group[caster][heat]
                       for heat in plant.group2heats[str(group)]])
        t_start = max(t_group, t_caster)
        task = plant.tasks[caster][group-1]
        t_end_process = t_start + plant.task_duration[task]
        t_cleanup = t_start + plant.task_cleanup_duration[task]
        schedule.append((4, group, caster, caster_id, task, t_start, t_end_process, t_cleanup))
        q_t_caster.put((t_cleanup, caster, caster_id))
    return schedule


def _convert_schedule_to_task_rng(schedule):
    """Convert the schedule to dict{task: starting time range}"""
    task_start_rng = dict()
    for record in schedule:
        # parse schedule entry
        task = record[4]
        t_start = record[5]     # process start time
        task_start_rng[task] = (t_start, t_start+1)
    return task_start_rng


def _convert_schedule_to_xy(schedule, model, plant):
    """Convert the schedule to xx,yy vector in RTN models."""
    xx = [0]*model.num_x
    yy = [0]*model.num_y
    for stage, unit2num in plant.stage2units.items():
        for unit, num in unit2num.items():
            unit = plant.resources[unit][0]
            for t in range(plant.num_t):
                yy[model.pos_y_res_t(unit, t)] = num
    for record in schedule:
        # parse schedule entry
        stage = record[0]
        heat = record[1]
        unit = record[2]
        unit_id = record[3]
        task = record[4]
        t_start = record[5]     # process start time
        t_end = record[6]       # process end time
        t_ready = record[7]     # equipment available time

        # task start time
        xx[model.pos_x_task_t(task, t_start)] = 1
        if 'TR_S' in unit:
            if stage < 4:
                for t in range(t_end, model.num_t):
                    yy[model.pos_y_res_t(plant.resources['H_B_S%d' % (stage+1)][heat-1], t)] += 1
        if unit in plant.equip2num:
            # equipment resource consumption
            for t in range(t_start, t_ready):
                yy[model.pos_y_res_t(plant.resources[unit][0], t)] -= 1
            # intermediate product consumption
            if 1 < stage < 4:
                for t in range(t_start, model.num_t):
                    yy[model.pos_y_res_t(plant.resources['H_B_S%d' % stage][heat-1], t)] -= 1
            # product interaction with casting procedure
            if stage == 4:
                group = record[1]
                for heat in plant.group2heats[str(group)]:
                    heat_consume = plant.heat_consume_time_in_group[unit][heat]
                    heat_generate = plant.heat_generate_time_in_group[unit][heat]
                    for t in range(t_start+heat_generate, model.num_t):
                        yy[model.pos_y_res_t(plant.resources['H_A_S4'][heat-1], t)] += 1
                    for t in range(t_start+heat_consume, model.num_t):
                        yy[model.pos_y_res_t(plant.resources['H_B_S4'][heat-1], t)] -= 1
    return xx, yy