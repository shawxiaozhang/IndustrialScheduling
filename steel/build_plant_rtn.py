""" Build the resource task network for steel plants.

Notations
    Resource Category
    EAF     -   equipment unit in the EAF stage (first stage)
    EAF1    -   (if multiple units in this stage) one equipment unit in the EAF stage
    H_A_S3  -   (RTN1) intermediate product after stage 3
    H_B_S4  -   (RTN1) intermediate product before stage 4
    H_A_LF2 -   (RTN2) intermediate product before unit LF2
    EN      -   electric energy
    Task Category
    EAF     -   process task by unit EAF, i.e. melting
    TR_S3   -   (RTN1) transportation task between stage 3 and stage 4
    TR_LF2_CC1    -   (RTN2) transportation task between units LF2 and CC1

__author__ = xxxzhang
"""

from collections import OrderedDict
import math

import numpy as np
import json
import logging

log = logging.getLogger('steel')


class PlantRtnBuilder():
    """ resources, tasks, and their interaction networks for steel plant scheduling
    """

    def __init__(self, plant, rtn_t0):
        self.rtn_t0 = rtn_t0  # time grid interval for RTN
        self.price_energy = []
        for price in plant['energy_price']:
            self.price_energy += [price] * (60 / self.rtn_t0)
        self.num_t = 24 * 60 / self.rtn_t0
        self.num_stage = len(plant['stage2units'].keys())
        self.num_groups = len(plant['group2heats'].keys())
        self.num_heats = 0
        for group, heats in plant['group2heats'].items():
            self.num_heats += len(heats)

        # basic information
        self.group2heats = plant['group2heats']
        self.stage2units = plant['stage2units']
        self.time_trans_max = plant['trans_time_max']
        self.time_setup = plant['setup_time']
        self.time_process = plant['equip2process_time']

        # resources and tasks
        # todo instead using arrays, using dicts in tasks/res
        [self.resources, self.num_resources] = self.build_resources(plant)
        [self.tasks, self.num_tasks] = self.build_tasks(plant)
        [self.task_duration, self.task_cleanup_duration] = self.cal_task_duration(plant)
        [self.rtn_profile, self.heat_consume_time_in_group, self.heat_generate_time_in_group] \
            = self.build_resource_task_profile(plant)

        # redundant information
        self.equip2num = {}
        for unit2num in self.stage2units.values():
            for unit, num in unit2num.items():
                self.equip2num[unit] = num
        self.maxPower = sum([float(plant['equip2mw'][unit])*num for unit, num in self.equip2num.items()])
        self.casters = [unit for unit, num in self.stage2units['4'].items()]
        self.heat_sequence = ['EAF', 'TR_EA', 'AOD', 'TR_AL', 'LF', 'TR_LC']
        self.main_process = ['EAF', 'AOD', 'LF', 'CC1', 'CC2']

        log.info('TaskCategory : (Task, Duration)')
        for task_type, task_list in self.tasks.items():
            log.info('%s : %s' % (task_type, ','.join('(%s,%d)' % (x, self.task_duration[x]) for x in task_list)))
        log.info('ResourceCategory : Resource')
        for res_cat, res_list in self.resources.items():
            log.info('%s : %s' % (res_cat, ','.join(str(x) for x in res_list)))
        # log.info('ResourceTaskInteraction')
        # log.info(json.dumps(self.rtn_profile, indent=2))

    def build_resources(self, plant):
        # todo, do you need the reverse mapping
        res_cat2idx = OrderedDict()
        r_idx = 1
        # equipment resources
        for stage in range(1, self.num_stage + 1):
            for unit in plant['stage2units'][str(stage)].keys():
                res_cat2idx[unit] = [r_idx]
                r_idx += 1
        # intermediate products and final products, A - after, B - before
        for stage in range(1, self.num_stage+1):
            res_cat2idx['H_A_S%s' % stage] = range(r_idx, r_idx + self.num_heats)
            r_idx += self.num_heats
            if int(stage) == 1:
                continue
            res_cat2idx['H_B_S%s' % stage] = range(r_idx, r_idx + self.num_heats)
            r_idx += self.num_heats
        # energy resource
        res_cat2idx['EN'] = [r_idx]
        res_num = r_idx
        return [res_cat2idx, res_num]

    def build_tasks(self, plant):
        tasks = OrderedDict()
        i_idx = 1
        for stage in range(1, self.num_stage + 1):
            for unit in plant['stage2units'][str(stage)].keys():
                if int(stage) == 4:
                    tasks[unit] = range(i_idx, i_idx + self.num_groups)
                    i_idx += self.num_groups
                else:
                    tasks[unit] = range(i_idx, i_idx + self.num_heats)
                    i_idx += self.num_heats
                    tasks['TR_S%s' % stage] = range(i_idx, i_idx + self.num_heats)
                    i_idx += self.num_heats
        task_num = i_idx - 1
        return [tasks, task_num]

    def cal_task_duration(self, plant):
        """calculate time slots duration of tasks"""
        task_duration = dict()
        task_cleanup_duration = dict()
        for heat in range(1, self.num_heats + 1):
            for task_type in ['EAF', 'AOD', 'LF']:
                task_duration[self.tasks[task_type][heat - 1]] = \
                    int(math.ceil(float(plant['equip2process_time'][task_type][str(heat)])/self.rtn_t0))
            for task_type in ['TR_S1', 'TR_S2', 'TR_S3']:
                task_duration[self.tasks[task_type][heat - 1]] = \
                    int(math.ceil(float(plant['trans_time'][task_type])/self.rtn_t0))
        for group in range(1, self.num_groups + 1):
            for task_type in ['CC1', 'CC2']:
                cast_time = [plant['equip2process_time'][task_type][str(heat)] for heat in self.group2heats[str(group)]]
                total_time = float(sum(cast_time))
                task_duration[self.tasks[task_type][group - 1]] = int(math.ceil(total_time/self.rtn_t0))
                total_time = float(sum(cast_time) + self.time_setup[task_type])
                task_cleanup_duration[self.tasks[task_type][group - 1]] = int(math.ceil(total_time/self.rtn_t0))
        return task_duration, task_cleanup_duration

    def build_resource_task_profile(self, plant):
        """calculate resource task profile, i.e. the interactions between resource and task"""
        rtn_profile = dict()
        for res_category, res_idxes in self.resources.items():
            for res_idx in res_idxes:
                rtn_profile[res_idx] = dict()
        # equipment usage
        for stage in range(1, self.num_stage + 1):
            for unit in plant['stage2units'][str(stage)].keys():
                res = self.resources[unit][0]
                for task in self.tasks[unit]:
                    rtn_profile[res][task] = [-1] + [0] * (self.task_duration[task] - 1) + [1]
                    if stage == 4:
                        rtn_profile[res][task] = [-1] + [0] * (self.task_cleanup_duration[task] - 1) + [1]
        # heat consumption and generation for the first three stages
        for heat_idx in range(0, self.num_heats):
            # process task generate intermediate heat H_A_S (after stage)
            for task_cat, res_cat in [('EAF', 'H_A_S1'), ('AOD', 'H_A_S2'), ('LF', 'H_A_S3')]:
                task = self.tasks[task_cat][heat_idx]
                resource = self.resources[res_cat][heat_idx]
                rtn_profile[resource][task] = [0] * self.task_duration[task] + [1]
            # process task consume intermediate heat H_B_S (before stage)
            for task_cat, res_cat in [('AOD', 'H_B_S2'), ('LF', 'H_B_S3')]:
                task = self.tasks[task_cat][heat_idx]
                resource = self.resources[res_cat][heat_idx]
                rtn_profile[resource][task] = [-1] + [0] * self.task_duration[task]
            # transfer task transports heat
            for task_cat, res_cat in [('TR_S1', ['H_A_S1', 'H_B_S2']),
                                      ('TR_S2', ['H_A_S2', 'H_B_S3']),
                                      ('TR_S3', ['H_A_S3', 'H_B_S4'])]:
                task = self.tasks[task_cat][heat_idx]
                resource1 = self.resources[res_cat[0]][heat_idx]
                rtn_profile[resource1][task] = [-1] + [0] * self.task_duration[task]
                resource2 = self.resources[res_cat[1]][heat_idx]
                rtn_profile[resource2][task] = [0] * self.task_duration[task] + [1]
        # group-heat consumption and generation
        heat_consume_time_in_group = dict()
        heat_generate_time_in_group = dict()
        for unit in plant['stage2units']['4']:
            heat_consume_time_in_group[unit] = dict()
            heat_generate_time_in_group[unit] = dict()
            for group, heats in self.group2heats.items():
                # todo because the task numbers are stored in an array, can we do a better mapping?
                task = self.tasks[unit][int(group)-1]
                duration = 0
                for heat in heats:
                    consume_time = int(math.floor(duration/self.rtn_t0))
                    resource = self.resources['H_B_S4'][heat - 1]
                    rtn_profile[resource][task] = [0] * (self.task_duration[task] + 1)
                    rtn_profile[resource][task][consume_time] = -1
                    duration += plant['equip2process_time'][unit][str(heat)]
                    generate_time = int(math.ceil(duration/self.rtn_t0))
                    resource = self.resources['H_A_S4'][heat - 1]
                    rtn_profile[resource][task] = [0] * (self.task_duration[task] + 1)
                    rtn_profile[resource][task][generate_time] = 1
                    heat_consume_time_in_group[unit][heat] = consume_time
                    heat_generate_time_in_group[unit][heat] = generate_time
        # energy usage
        for stage, units in plant['stage2units'].items():
            stage = int(stage)
            for unit in units.keys():
                norm_mw = float(plant['equip2mw'][unit])
                for task in self.tasks[unit]:
                    total_energy = 0
                    if stage < 4:
                        heat = task - self.tasks[unit][0] + 1
                        total_energy = norm_mw * plant['equip2process_time'][unit][str(heat)] / 60
                    elif stage == 4:
                        group = task - self.tasks[unit][0] + 1
                        for heat in self.group2heats[str(group)]:
                            total_energy += norm_mw * plant['equip2process_time'][unit][str(heat)] / 60
                    profile = [norm_mw * self.rtn_t0 / 60] * (self.task_duration[task] - 1)
                    profile = profile + [total_energy - sum(profile)] + [0]
                    rtn_profile[self.resources['EN'][0]][task] = profile
        return rtn_profile, heat_consume_time_in_group, heat_generate_time_in_group

    def cal_initial_resource(self, res):
        # todo, here needs reverse mapping
        # only equipment initial value not zero
        for stage, stage_units in self.stage2units.items():
            for unit in stage_units.keys():
                if res in self.resources[unit]:
                    return stage_units[unit]
        return 0

    # todo delete this
    def get_unit2num(self):
        unit2num = dict()
        for stage, units in self.stage2units.items():
            for key, value in units.items():
                unit2num[key] = value
        return unit2num

    def get_same_heats(self):
        """find heats which share same processing times"""
        process_time = np.zeros((self.num_heats, len(self.main_process)))
        for i in range(len(self.main_process)):
            for heat_ in range(self.num_heats):
                process_time[heat_, i] = self.time_process[self.main_process[i]][heat_ + 1]
        visited = []
        same_heat = dict()
        for row in range(self.num_heats):
            if row not in visited:
                visited.append(row)
                same_heat[row + 1] = [row + 1]
                for row2 in range(row + 1, self.num_heats):
                    if np.array_equal(process_time[row2], process_time[row]):
                        same_heat[row + 1].append(row2 + 1)
                        visited.append(row2)
        return same_heat




