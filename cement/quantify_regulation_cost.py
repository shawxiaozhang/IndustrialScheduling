"""
Quantify regulation cost.
First run simulations to obtain historical records.
Then analyze the historical records to get cost coefficients by regression.
"""

import logging
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpc_coordinator import predict_agc, CementPlant, EnergyStorage, BuildOptModel

log = logging.getLogger('cement')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(module)s %(levelname)s %(lineno)d: - %(message)s'))
log.addHandler(ch)


def test_simulate_mpc_hours(log_file='history_records.txt'):
    """Run MPC simulations over many hours and record the hourly summary.
    With various settings of R/B and unique setting of MPC config.
    """

    agc_days = np.loadtxt('data/regD-13-01-04-Fri.txt')
    # agc_days = np.hstack((agc, np.loadtxt('data/regD-13-01-05-Sat.txt')))

    config = {'beta': 10,
              'gamma': 10,
              'switch_limit': 15,
              'mpc_horizon': 15}

    base2reg = {4: [3.5, 4, 4.5, 5, 5.5, 6, 6.5],
                2: [3.75, 4.25],
                6: [3.75, 4.25],
                3: [3.5, 4, 4.5]}

    agc_days = agc_days[:1800*5]
    base2reg = {k: base2reg[k] for k in [4]}

    fo = open(log_file, 'w+')
    msg = 'alpha 10 beta %d gamma %d e_ratio %.1f switch_limit %d horizon %d' % \
          (config['beta'], config['gamma'], -9.9, config['switch_limit'], config['mpc_horizon'])
    fo.write(msg + '\n')

    for base_mw in base2reg.keys():
        for reg_mw in base2reg[base_mw]:
            for hour in range(1, len(agc_days)/1800-2, 1):
                [preds, agc] = predict_agc(agc_days, mpc_horizon=config['mpc_horizon'], hour=hour)
                cement = CementPlant(config['switch_limit'])
                storage = EnergyStorage()
                opt = BuildOptModel(cement, storage)
                opt.set_base_reg(base_mw, reg_mw)
                opt.mpc_obj_beta = config['beta']
                opt.mpc_obj_gamma = config['gamma']
                cement_p_mw, storage_p_mw, storage_e_mwh, cpu_list = opt.simulate_mpc(agc, preds)
                if cement_p_mw is None:
                    continue
                reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh = opt.simulation_summary(agc, cement_p_mw, storage_p_mw, storage_e_mwh, cpu_list)
                penalty = reg_vio_mwh*10 + switch_mw*0.1 + 2*abs(storage_de_mwh)
                msg = 'R %.1f B %d hour %2d P %7.1f = (%.3f,%.1f,%.1f) Cem %.1f' \
                      % (reg_mw, base_mw, hour, penalty, reg_vio_mwh, switch_mw, storage_de_mwh, cement_mwh)
                fo.write(msg + '\n')
    fo.close()


def test_analyze_records(file_name='history_records.txt'):
    """Analyze the hourly summary (violation, switching, etc.) of regulation provision over days."""
    costs = ['Regulation Violation MWh', 'Storage Deviation MWh', 'Cement Energy', 'Switch MW']

    for cost in costs:
        records = dict()
        with open(file_name, 'r+') as f:
            for line in f.readlines()[1:]:
                words = line.split()
                reg = float(words[1])
                base = float(words[3])
                hour = int(words[5])
                penalty = float(words[7])
                vio_reg_mwh = float(words[9][1:-1].split(',')[0])
                switch_mw = float(words[9][1:-1].split(',')[1])
                e_deviation_mwh = float(words[9][1:-1].split(',')[2])
                cement_energy = float(words[11])
                if base not in records:
                    records[base] = dict()
                if reg not in records[base]:
                    records[base][reg] = dict()
                if 'Penalty' not in records[base][reg]:
                    records[base][reg]['Penalty'] = dict()
                    records[base][reg]['Regulation Violation MWh'] = dict()
                    records[base][reg]['Switch MW'] = dict()
                    records[base][reg]['Storage Deviation MWh'] = dict()
                    records[base][reg]['Cement Energy'] = dict()
                records[base][reg]['Penalty'][hour] = penalty
                records[base][reg]['Regulation Violation MWh'][hour] = vio_reg_mwh
                records[base][reg]['Switch MW'][hour] = switch_mw
                records[base][reg]['Storage Deviation MWh'][hour] = e_deviation_mwh
                records[base][reg]['Cement Energy'][hour] = cement_energy

        expectation = dict()
        matplotlib.rcParams['font.size'] = 16
        plt.figure(figsize=(12, 3))
        for base in sorted(records.keys()):
            expectation[base] = dict()
            for reg in sorted(records[base].keys()):
            # for reg in [4, 5, 6]:
                curve = []
                hours = sorted(records[base][reg][cost].keys())
                for hour in hours:
                    curve += [records[base][reg][cost][hour]]
                log.info('R %.1f B %.1f %s %f' % (reg, base, cost, np.mean(curve)))
                expectation[base][reg] = np.mean(curve)
                curve = curve[:48]
                hours = range(len(curve))
                plt.plot(hours, curve, label='R %.1f B %.1f' % (reg, base), ls='-', marker='s', linewidth=2,
                         markersize=9)
                hours = [hours[i] for i in range(0, len(hours), 3)]
                plt.xticks(hours)
        # plt.legend(loc='upper center', prop={'size': 12})
        plt.legend(loc='upper right', prop={'size': 16})
        plt.xlabel('Hour')
        plt.ylabel(cost)
        plt.grid()
        if cost == 'Regulation Violation MWh':
            plt.ylim([-0.0005, 0.0055])
        elif cost == 'Switch MW':
            plt.ylim([-3, 53])
        elif cost == 'Cement Energy':
            plt.ylim([0, 8])
        elif cost == 'Storage Deviation MWh':
            plt.ylim([-0.55, 0.55])
        plt.savefig('%s.pdf' % cost.replace(' ', ''))
        plt.show()

        # hourly average cost
        colors = ['b', 'r', 'g', 'k', 'm']
        plt.figure(figsize=(12, 6))
        for base in expectation.keys():
            regs = sorted(expectation[base].keys())
            vals = [expectation[base][reg] for reg in regs]
            plt.plot(regs, vals, ls='None', marker='s', linewidth=2, markersize=9,
                     color=colors[int(base/2)-1])
            p = np.polyfit(regs, vals, 1)
            xx = np.linspace(3.0, max(regs)+0.5)
            yy = np.polyval(p, xx)
            plt.plot(xx, yy, ls='--', linewidth=3, color=colors[int(base/2)-1], label='B %.1f' % base)
            log.info('B%d p %.3f %.3f' % (base, p[0], p[1]))
        plt.legend(loc='upper center')
        plt.xlabel('Regulation MW')
        plt.ylabel(cost)
        plt.grid()
        plt.savefig('Avg%s.pdf' % cost.replace(' ', ''))
        plt.show()


if __name__ == "__main__":
    log.info(datetime.datetime.today())
    test_simulate_mpc_hours()
    test_analyze_records()



