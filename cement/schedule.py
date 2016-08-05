"""
Optimal schedule of B and R for industrial machines and energy storage
with regulation provision over one day.

B: baseline, R: regulation capacity.
"""


import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cplex


class Scheduler:
    def __init__(self, price_e, price_r, price_switch, price_cement, tau):
        self.H = 24
        self.B = 3
        # profile_id : base mw, C0, C1, Rlo, Rup,
        self.base_profile = {0: [2, -19.481, 6.991, 3.0, 5.0],
                             1: [4, -28.307, 9.069, 3.0, 7.0],
                             2: [6, -18.581, 6.920, 3.0, 5.0]}
        self.q_min = 0
        self.q_max = 50
        self.q0 = 0.5*(self.q_min+self.q_max)
        self.price_e = price_e
        self.price_r = price_r
        self.price_switch = price_switch     # cost per switch
        self.price_cement = price_cement     # revenue of cement per MWh
        self.tau = tau
        self.num_bin = self.B*self.H
        self.num_var = 2*self.B*self.H + self.H

    def set_q_max_min(self, q_max, q_min=0):
        self.q_max = q_max
        self.q_min = q_min
        self.q0 = 0.5*(self.q_max + self.q_min)

    def pos_z(self, b, h):
        """Position of z variable."""
        assert 0 <= b <= self.B-1
        assert 0 <= h <= self.H-1
        return b*self.H+h

    def pos_r(self, b, h):
        """Position of r variable."""
        assert 0 <= b <= self.B-1
        assert 0 <= h <= self.H-1
        return self.B*self.H + b*self.H+h

    def pos_q(self, h):
        """Position of q variable."""
        assert 0 <= h <= self.H-1
        return 2*self.B*self.H + h

    def build_model_solve(self):
        x_lo = [0]*self.num_var
        x_up = [1]*self.num_var
        for h in range(self.H):
            for b in range(self.B):
                x_up[self.pos_r(b, h)] = self.base_profile[b][4]
            x_up[self.pos_q(h)] = self.q_max
            x_lo[self.pos_q(h)] = self.q_min

        f_obj = [0]*self.num_var
        for h in range(self.H):
            for b in range(self.B):
                # switch cost
                f_obj[self.pos_r(b, h)] -= self.base_profile[b][2]*self.price_switch
                f_obj[self.pos_z(b, h)] -= self.base_profile[b][1]*self.price_switch
                # energy net cost
                f_obj[self.pos_z(b, h)] -= self.base_profile[b][0]*(self.price_e[h] - self.price_cement)
                # regulation profit
                f_obj[self.pos_r(b, h)] += self.price_r[h]

        cons = []
        rhs = []
        con_sense = []
        row = 0
        # base power status
        for h in range(self.H):
            for b in range(self.B):
                cons.append([row, self.pos_z(b, h), 1])
            rhs.append(1)
            con_sense.append('E')
            row += 1
        # regulation capacity bounds
        for b in range(self.B):
            r_up = self.base_profile[b][4]
            r_lo = self.base_profile[b][3]
            for h in range(self.H):
                cons.append([row, self.pos_r(b, h), 1])
                cons.append([row, self.pos_z(b, h), -r_lo])
                con_sense.append('G')
                rhs.append(0)
                row += 1
                cons.append([row, self.pos_r(b, h), -1])
                cons.append([row, self.pos_z(b, h), r_up])
                con_sense.append('G')
                rhs.append(0)
                row += 1
        # Q dynamic equation
        for h in range(self.H-1):
            cons.append([row, self.pos_q(h), 1])
            cons.append([row, self.pos_q(h+1), -1])
            for b in range(self.B):
                cons.append([row, self.pos_z(b, h), self.base_profile[b][0]])
            con_sense.append('E')
            rhs.append(self.tau)
            row += 1
        # initial q stock
        cons.append([row, self.pos_q(0), 1])
        for b in range(self.B):
            cons.append([row, self.pos_z(b, 0), -self.base_profile[b][0]])
        rhs.append(self.q0 - self.tau)
        con_sense.append('E')
        row += 1

        var_types = 'I'*self.num_bin + 'C'*(self.num_var-self.num_bin)

        # config cplex
        my_prob = cplex.Cplex()
        my_prob.set_log_stream(None)
        my_prob.set_results_stream(None)
        gap_tolerance = 1e-6
        my_prob.parameters.mip.tolerances.mipgap.set(gap_tolerance)
        my_prob.parameters.clocktype.set(2)
        my_prob.parameters.timelimit.set(7200)

        # feed cplex
        my_prob.linear_constraints.add(rhs=rhs, senses=con_sense)
        my_prob.objective.set_sense(my_prob.objective.sense.maximize)
        my_prob.variables.add(obj=f_obj, ub=x_up, lb=x_lo)
        my_prob.linear_constraints.set_coefficients(cons)
        my_prob.variables.set_types(zip(range(len(var_types)), var_types))

        # solve by cplex
        time1 = time.time()
        my_prob.solve()
        cpu_time = time.time() - time1
        print 'schedule cpu time %.1f' % cpu_time

        status = my_prob.solution.get_status()
        if status in [103]:
            print 'NO cplex solution with status'

        var_solution = my_prob.solution.get_values()[0:len(var_types)]
        return var_solution

    def analyze(self, var_solution):
        # analyze scheduling result
        base_power = [0]*self.H
        regulation = [0]*self.H
        q_stock = [0]*self.H
        for h in range(self.H):
            for b in range(self.B):
                base_power[h] += self.base_profile[b][0]*var_solution[self.pos_z(b, h)]
                regulation[h] += var_solution[self.pos_r(b, h)]
            q_stock[h] += var_solution[self.pos_q(h)]

        matplotlib.rcParams['font.size'] = 11
        plt.figure(figsize=(6, 6))
        plt.subplot(3, 1, 1)
        plt.xlim([0, 24])
        plt.ylabel('Price [$/MW]')
        plt.plot(self.price_e, label='Energy')
        plt.plot(self.price_r, label='Regulation')
        plt.grid()
        plt.legend(loc='upper left', ncol=2, fontsize=12)

        plt.subplot(3, 1, 2)
        plt.ylabel('Power [MW]')
        plt.plot(base_power, label='Base Power')
        plt.plot(regulation, label='Regulation Capacity')
        plt.grid()
        plt.legend(loc='lower left', ncol=2, fontsize=12)
        plt.ylim([-1, 9])
        plt.xlim([0, 24])

        plt.subplot(3, 1, 3)
        plt.ylabel('Stock [MWh]')
        plt.plot(q_stock, color='r', label='Cement Stock')
        plt.grid()
        plt.ylim([self.q_min-1, self.q_max+1])
        plt.xlim([0, 24])
        plt.legend(loc='lower left', ncol=2, fontsize=12)
        plt.savefig('schedule.pdf')
        plt.show()


if __name__ == '__main__':
    e_price = np.loadtxt('data/miso_price/lmp2.txt', delimiter=',')
    r_price = np.loadtxt('data/miso_price/reg2.txt', delimiter=',')
    test = Scheduler(e_price, r_price, 0.5, 30, 4)
    test.set_q_max_min(20, 0)
    solution = test.build_model_solve()
    test.analyze(solution)

