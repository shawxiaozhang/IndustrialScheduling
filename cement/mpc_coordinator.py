"""
MPC Coordination for Hourly Operation.
B/R are given.
"""


import time
import itertools

import cplex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class CementPlant():
    def __init__(self, switch_max=100):
        self.num_crusher = 4
        self.crusher_power_mw = [2.0, 2.0, 2.0, 2.0]
        self.profit_mwh = 30.0
        self.max_mw = sum(self.crusher_power_mw)
        self.min_mw = 0.0

        self.con_switch_max = switch_max

        self.con_switch_period_s = 5*60
        self.con_e_period_s = 15*60


class EnergyStorage():
    def __init__(self):
        self.e_max_mwh = 1.0
        self.e_min_mwh = 0.0
        self.p_max_mw = 3.0
        self.p_min_mw = -3.0
        self.e_initial_mwh = 0.5


class MarketSignals():
    def __init__(self, agc, agc_t0_s):
        self.agc = agc
        self.agc_t0_s = agc_t0_s
        self.energy_price_mwh = [20]
        self.regulation_price_mw = [12]     # hourly price


class BuildOptModel():
    def __init__(self, cement, storage, signals):
        self.ts = 2         # time interval for storage, in seconds
        self.tc = 60        # time interval for cement, in seconds
        num_hour = len(signals.energy_price_mwh)
        self.num_tc = 60*60/self.tc*num_hour
        self.num_ts = 60*60/self.ts*num_hour
        self.num_x = cement.num_crusher*self.num_tc  # cement crusher status
        self.num_p = self.num_ts    # energy storage charging power
        self.num_e = self.num_ts    # energy storage level
        self.num_b = num_hour       # baseline for regulation provision
        self.num_r = num_hour       # regulation capacity
        self.num_var = self.num_x + self.num_p + self.num_e + self.num_b + self.num_r
        self.cement = cement
        self.storage = storage
        self.signals = signals
        self.forecast = None
        self.reg_MW = 0
        self.base_MW = 0

        self.mpc_obj_alpha = 10.0
        self.mpc_obj_beta = 10.0
        self.mpc_obj_gamma = 10.0

    def set_mpc(self, mpc):
        if mpc is None:
            mpc = dict()
            mpc['horizon'] = 1
            mpc_h = mpc['horizon']
            ratio = self.tc/self.ts
            # mpc in the units of p.u.*s, only used in the MPC control step
            mpc['agc_int'] = [[self.signals.agc_t0_s*sum(self.signals.agc[t1*ratio:(t1+1)*ratio]) for t1 in range(tc, tc+mpc_h)] for tc in range(self.num_tc)]
            mpc['agc_max'] = [[max(np.append(self.signals.agc[t1*ratio:(t1+1)*ratio], [0.0])) for t1 in range(tc, tc+mpc_h)] for tc in range(self.num_tc)]
            mpc['agc_min'] = [[min(np.append(self.signals.agc[t1*ratio:(t1+1)*ratio], [0.0])) for t1 in range(tc, tc+mpc_h)] for tc in range(self.num_tc)]
        else:
            if 'agc_max' not in mpc and 'horizon' in mpc:
                mpc['agc_max'] = [[1.0 for t1 in range(tc, tc+mpc['horizon'])] for tc in range(self.num_tc)]
                mpc['agc_min'] = [[-1.0 for t1 in range(tc, tc+mpc['horizon'])] for tc in range(self.num_tc)]
        self.forecast = mpc

    def mpc_control_ts(self, pred, para):
        h = len(pred)
        cement_p0 = para['cement_p0']
        storage_e0 = para['storage_e0']

        c_pos = lambda x: x
        se_pos = lambda x: h+x
        sp_pos = lambda x: 2*h+x
        dr_pos = lambda x: 3*h+x        # deviation of regulation command
        dc_pos = lambda x: 4*h+x        # change of cement crusher number

        # variable bounds
        x_up = [-1.0]*5*h
        x_lo = [0.0]*5*h
        for x in range(h):
            x_up[c_pos(x)] = self.cement.num_crusher
            x_up[se_pos(x)] = self.storage.e_max_mwh
            x_lo[se_pos(x)] = self.storage.e_min_mwh
            x_up[sp_pos(x)] = self.storage.p_max_mw
            x_lo[sp_pos(x)] = self.storage.p_min_mw
            x_up[dr_pos(x)] = self.reg_MW*2.0
            x_up[dc_pos(x)] = self.cement.num_crusher
        var_types = 'I'*h + 'C'*4*h

        row = 0
        rhs = [0.0]*5*h
        con_senses = ['E']*h + ['G', 'G']*h + ['G']*2*h
        triples = []
        # energy balance
        for x in range(h):
            triples.append([row, sp_pos(x), -self.ts/3600.0])
            triples.append([row, se_pos(x), 1.0])
            if x == 0:
                rhs[row] = storage_e0
            else:
                triples.append([row, se_pos(x-1), -1.0])
            row += 1
        # power balance
        for x in range(h):
            triples.append([row, dr_pos(x), 1.0])
            triples.append([row+1, dr_pos(x), 1.0])
            reg_command = self.base_MW + self.reg_MW*pred[x]
            rhs[row] = reg_command
            rhs[row+1] = -reg_command
            triples.append([row, c_pos(x), self.cement.crusher_power_mw[0]])
            triples.append([row+1, c_pos(x), -self.cement.crusher_power_mw[0]])
            triples.append([row, sp_pos(x), 1.0])
            triples.append([row+1, sp_pos(x), -1.0])
            row += 2
        # cement crusher change
        for x in range(h):
            triples.append([row, dc_pos(x), 1.0])
            triples.append([row+1, dc_pos(x), 1.0])
            triples.append([row, c_pos(x), -1.0])
            triples.append([row+1, c_pos(x), 1.0])
            if x == 0:
                rhs[row] = -cement_p0/self.cement.crusher_power_mw[0]
                rhs[row+1] = cement_p0/self.cement.crusher_power_mw[0]
            else:
                triples.append([row, c_pos(x-1), 1.0])
                triples.append([row+1, c_pos(x-1), -1.0])
            row += 2
        # obj
        obj = [0]*len(x_up)
        for x in range(h):
            obj[dr_pos(x)] = self.mpc_obj_alpha
            obj[dc_pos(x)] = self.mpc_obj_beta

        # cement operation limit
        if 'cement_e_min' in para and False:
            for n1 in range(h):
                rhs.append(para['cement_e_min'][n1])
                for x in range(n1):
                    triples.append([row, c_pos(x), self.cement.crusher_power_mw[0]*self.ts/3600.0])
                con_senses.append("G")
                row += 1
        if 'cement_switch_max' in para and True:
            for n1 in range(h):
                rhs.append(para['cement_switch_max'][n1])
                for x in range(n1):
                    triples.append([row, dc_pos(x), 1.0])
                con_senses.append('L')
                row += 1
        if 'storage_level_penalty' in para and True:
            obj.append(para['storage_level_penalty'])
            x_up.append(0.5*self.storage.e_max_mwh)
            x_lo.append(0)
            var_types += 'C'
            de_pos = 5*h
            rhs.append(-0.5*self.storage.e_max_mwh)
            rhs.append(0.5*self.storage.e_max_mwh)
            triples.append([row, de_pos, 1.0])
            triples.append([row+1, de_pos, 1.0])
            triples.append([row, se_pos(h-1), -1.0])
            triples.append([row+1, se_pos(h-1), 1.0])
            row += 2
            con_senses.append('G')
            con_senses.append('G')

        # config cplex
        my_prob = cplex.Cplex()
        my_prob.set_log_stream(None)
        my_prob.set_results_stream(None)
        gap_tolerance = 1e-6
        my_prob.parameters.clocktype.set(2)
        my_prob.parameters.timelimit.set(7200)
        my_prob.parameters.mip.tolerances.mipgap.set(gap_tolerance)

        # feed cplex
        my_prob.linear_constraints.add(rhs=rhs, senses=con_senses)
        my_prob.objective.set_sense(my_prob.objective.sense.minimize)
        my_prob.variables.add(obj=obj, ub=x_up, lb=x_lo)
        my_prob.linear_constraints.set_coefficients(triples)
        my_prob.variables.set_types(zip(range(len(var_types)), var_types))

        # call cplex
        time1 = time.time()
        my_prob.solve()
        cpu_time = time.time() - time1

        status = my_prob.solution.get_status()
        if status in [103]:
            print 'NO cplex solution with status %d at %d' % (status, para['step'])
            return [None, None, None, None, None]

        var_solution = my_prob.solution.get_values()[0:len(var_types)]

        cement_n = [var_solution[c_pos(x)] for x in range(h)]
        storage_p = [var_solution[sp_pos(x)] for x in range(h)]
        switch = [var_solution[dc_pos(x)] for x in range(h)]
        vio = [var_solution[dr_pos(x)] for x in range(h)]

        return [cement_n, storage_p, vio, switch, cpu_time]

    def simulate_control_ts(self, reg_mw, base_mw, cement_mw_t0, policy, file_name='case_record.txt'):
        self.base_MW = base_mw
        self.reg_MW = reg_mw

        cement_p_mw = [cement_mw_t0]*self.num_ts
        storage_p_mw = [0.0]*self.num_ts
        storage_e_mwh = [0.5*self.storage.e_max_mwh]*self.num_ts

        # moving horizon
        tp = self.forecast.shape[1]*self.ts
        cpu_list = []
        for step in range(self.num_ts):
            pred = self.forecast[step, :]
            para = dict()
            para['step'] = step
            para['storage_e0'] = storage_e_mwh[step-1]
            para['cement_p0'] = cement_p_mw[step-1]
            para['reg_mw'] = reg_mw
            para['base_mw'] = base_mw
            para['cement_switch_max'] = dict()
            for h in range(tp/self.ts):
                past_mw = cement_p_mw[max(0, step+h-self.cement.con_e_period_s/self.ts):step]
                switch_done = np.sum(np.absolute(np.subtract(past_mw[1:], past_mw[:-1])))/self.cement.crusher_power_mw[0]
                para['cement_switch_max'][h] = self.cement.con_switch_max - switch_done
                e_done = np.sum(cement_p_mw[max(0, step+h-self.cement.con_switch_period_s/self.ts):step])*self.ts/3600.0
            para['storage_level_penalty'] = self.mpc_obj_gamma

            [cement_n_h, a, b, c, cpu_time] = self.mpc_control_ts(pred, para)
            if cement_n_h is None:
                with open(file_name, 'a+') as f:
                    msg = 'Fail %s R %.1f B %.1f H %d Pen %d %d %d vio %.2f Cm %d %.1f Stg %.1f %.1f %.1f ' \
                          'cpu %.3f lim %d' \
                          % (policy+'no', self.reg_MW, self.base_MW, len(self.forecast[0]),
                             self.mpc_obj_alpha, self.mpc_obj_beta, self.mpc_obj_gamma,
                             -1, -1, -1, -1, -1, -1, -1, self.cement.con_switch_max)
                    print msg
                    f.write(msg + '\n')
                    f.flush()
                    f.close()
                return [None, None, None, None]

            cpu_list.append(cpu_time)
            # simulate the ts step
            cement_p_mw[step] = cement_n_h[0]*self.cement.crusher_power_mw[0]
            agc_next = self.signals.agc[step]
            desire_p = agc_next*reg_mw + base_mw - cement_p_mw[step]
            # bound storage power
            storage_p_mw[step] = max(min(desire_p, self.storage.p_max_mw), self.storage.p_min_mw)
            storage_e_mwh[step] = storage_e_mwh[step-1] + storage_p_mw[step]*self.ts/3600.0

        [reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh] = \
            self.display_results(cement_p_mw, storage_p_mw, storage_e_mwh, cpu_list, policy)
        return reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh

    def display_results(self, cement_p_mw, storage_p_mw, storage_e_mwh, cpu_list, policy,
                        display=False, verbose=False, file_name='case_record.txt'):
        matplotlib.rc('font', size=12)
        matplotlib.rcParams['xtick.labelsize'] = 10
        matplotlib.rcParams['ytick.labelsize'] = 10

        if verbose:
            switch_record = []
            energy_record = []
            for ts in range(self.num_ts):
                c_mw_1 = cement_p_mw[max(0, ts-self.cement.con_switch_period_s/self.ts):ts]
                c_mw_2 = cement_p_mw[max(0, ts-self.cement.con_e_period_s/self.ts):ts]
                switch_record.append(sum(np.absolute(np.subtract(c_mw_1[1:], c_mw_1[:-1]))))
                energy_record.append(sum(c_mw_2)*self.ts/3600.0)
            np.savetxt('record_switch.txt', switch_record)
            np.savetxt('record_energy.txt', energy_record)
            plt.plot(switch_record)
            plt.plot(energy_record)
            plt.show()

        base_p_mw = [self.base_MW]*self.num_ts
        reg_cmd_mw = [self.reg_MW*self.signals.agc[t_] + base_p_mw[t_] for t_ in range(self.num_ts)]
        plant_p_mw = [storage_p_mw[t_] + cement_p_mw[t_] for t_ in range(self.num_ts)]

        violation = [plant_p_mw[x] - reg_cmd_mw[x] for x in range(self.num_ts)]
        reg_vio_mwh = sum(np.absolute(violation))*self.ts/3600.0
        switch_mw = sum(np.absolute(np.subtract(cement_p_mw[1:], cement_p_mw[:-1])))
        cement_mwh = sum(cement_p_mw)*self.ts/3600.0
        storage_de_mwh = storage_e_mwh[-1] - self.storage.e_initial_mwh
        highest = max(storage_e_mwh)
        lowest = min(storage_e_mwh)

        avg_cpu = np.mean(cpu_list)
        obj = - self.signals.regulation_price_mw[0]*self.reg_MW
        for tc in range(self.num_tc):
            cost = self.signals.energy_price_mwh[0] - self.cement.profit_mwh
            obj += cost*self.tc/3600.0*cement_p_mw[tc]
        # print 'obj %f (regulation revenue + cement production profit)' % -obj

        title = '%s R %.1f B %.1f H %d Pen %d %d %d vio %.2f Cm %d %.1f Stg %.1f %.1f %.1f ' \
                'cpu %.3f lim %d' \
                % (policy, self.reg_MW, self.base_MW, len(self.forecast[0]), self.mpc_obj_alpha,
                   self.mpc_obj_beta, self.mpc_obj_gamma, reg_vio_mwh, switch_mw, cement_mwh,
                   storage_e_mwh[-1], highest, lowest, avg_cpu, self.cement.con_switch_max)
        with open(file_name, 'a+') as f:
            f.write(title + '\n')
            f.flush()
            f.close()

        if display:
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            ax1.plot(range(0, 3600, self.ts), base_p_mw, color='b', ls='-', linewidth=2)
            ax1.plot(range(0, 3600, self.ts), reg_cmd_mw, label='agc command', color='r', ls='-', linewidth=2)
            ax1.plot(range(0, 3600, self.ts), plant_p_mw, label='plant power', color='k', ls='--', linewidth=2)
            ax2.plot(range(0, 3600, self.ts), cement_p_mw, label='cement power', color='r', ls='-', linewidth=2)
            ax2.plot(range(0, 3600, self.ts), storage_p_mw, label='storage power', color='b', ls='-', linewidth=2)
            ax2.plot(range(0, 3600, self.ts), [self.storage.p_max_mw]*(3600/self.ts),  ls='--', color='b', linewidth=1)
            ax2.plot(range(0, 3600, self.ts), [self.storage.p_min_mw]*(3600/self.ts), ls='--', color='b', linewidth=1)
            ax3.plot(range(0, 3600, self.ts), storage_e_mwh, label='storage level', color='b', ls='-', linewidth=2)
            ax1.grid()
            ax2.grid()
            ax3.grid()

            ax1.set_ylabel('regulation [MW]')
            ax2.set_ylabel('devices [MW]')
            ax3.set_ylabel('storage [MWh]')
            ax1.legend(loc='lower center')
            ax2.legend(loc='lower center')
            ax3.legend(loc='upper left')
            # ax1.set_ylabel('regulation [MW]', fontsize=15)
            # ax2.set_ylabel('cement and storage power [MW]', fontsize=15)
            # ax3.set_ylabel('storage energy level [MWh]', fontsize=15)
            ax1.legend(loc='lower center', fontsize=10)
            ax2.legend(loc='lower center', fontsize=10)
            ax3.legend(loc='upper left', fontsize=10)

            ax1.set_ylim([-3.5, 11.5])
            ax2.set_ylim([-3.5, 8.5])
            # ax3.set_ylim([-0.5*self.storage.e_max_mwh, 1.1*self.storage.e_max_mwh])
            ax3.set_ylim([-0.05*self.storage.e_max_mwh, 1.05*self.storage.e_max_mwh])

            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_ticks([i*300 for i in range(0, 13)])
            frame1.axes.get_xaxis().set_ticklabels([5*i for i in range(0, 13)])
            plt.xlabel('Minute')
            # plt.rc('xtick', labelsize=30)
            # plt.rc('ytick', labelsize=30)
            plt.gca().set_xlim(left=0)
            plt.gca().set_xlim(right=3600)

            f.suptitle(title)
            plt.savefig(title + '.jpg', dpi=f.dpi)
            plt.show()

        return [reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh]


def hankel(Y, k):
    n = Y.shape[0]
    return np.hstack([Y[r:n-k+r+1, :] for r in range(k)])


def predict_agc(agc, n_horizon=15, hour=6):
    n_p = 60
    # train
    xy = hankel(agc.reshape((len(agc), 1)), n_p+n_horizon)
    phi = xy[:, :n_p]
    yy = xy[:, -n_horizon:]
    m, n = phi.shape
    lam = 1
    theta = np.linalg.solve(phi.T.dot(phi) + lam*np.eye(n), phi.T.dot(yy))
    yy_pred = phi.dot(theta)
    # err = np.std(yy_pred - yy, axis=0)

    agc = agc[hour*1800: (hour+1)*1800]
    offset = hour*1800 - n_p
    preds = yy_pred[offset:offset+1800, :]

    # plt.figure()
    # plt.plot(agc)
    # for ts in range(0, 1800, 100):
    #     plt.plot(np.array(range(preds.shape[1])) + ts, preds[ts], color='r', linewidth=3)
    # plt.show()

    return preds, agc


def test_hourly_simulation(agc):
    # configs
    reg_list = [4, 5, 6]
    base_list = [4]
    switch_list = [10]
    e_list = [0.5]
    gama_list = [10]
    beta_list = [10]

    for beta, switch_limit, e_ratio, gama, reg_mw, base_mw \
            in itertools.product(beta_list, switch_list, e_list, gama_list, reg_list, base_list):
        signals = MarketSignals(agc, 2)
        cement = CementPlant(switch_limit)
        storage = EnergyStorage()
        opt = BuildOptModel(cement, storage, signals)
        opt.set_mpc([[0]])
        opt.mpc_obj_beta = beta
        opt.mpc_obj_gamma = gama

        opt.reg_MW = reg_mw
        opt.base_MW = base_mw

        para = dict()
        para['step'] = 0
        para['storage_e0'] = 0.5*storage.e_max_mwh
        para['cement_p0'] = base_mw
        para['reg_mw'] = reg_mw
        para['base_mw'] = base_mw
        [cement_n_h, a, b, c, cpu_time] = opt.mpc_control_ts(agc, para)
        if cement_n_h is None:
            print 'overall optimization fails'
            return

        cement_p_mw = [cement_n_h[x]*cement.crusher_power_mw[0] for x in range(len(agc))]
        desire_p = [agc[x]*reg_mw + base_mw - cement_p_mw[x] for x in range(len(agc))]
        # bound storage power
        storage_p_mw = [max(min(desire_p[x], storage.p_max_mw), storage.p_min_mw) for x in range(len(agc))]
        storage_e_mwh = [0.5*storage.e_max_mwh]*len(agc)
        for step in range(1, len(agc)):
            storage_e_mwh[step] = storage_e_mwh[step-1] + storage_p_mw[step]*opt.ts/3600.0

        [reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh] \
            = opt.display_results(cement_p_mw, storage_p_mw, storage_e_mwh, [0], 'Ideal',
                                  display=True, verbose=False)
        print reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh


if __name__ == "__main__":
    hourly_agc = np.loadtxt('data/regD-13-01-04-Fri.txt')[1800*5:1800*6].tolist()
    test_hourly_simulation(hourly_agc)
