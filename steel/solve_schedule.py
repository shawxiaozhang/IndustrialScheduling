"""

__author__ = xxxzhang
"""

import time

import scipy
from scipy import sparse
import cplex


CONFIG_CPLEX_MAX_CPU = 7200
CONFIG_CPLEX_EPSILON_OBJ = 1e-6
# CONFIG_CPLEX_EPSILON_OBJ = 1e-4


def convert_to_general_opt(math_model):
    ub = math_model.x_up + math_model.y_up
    lb = math_model.x_lo + math_model.y_lo
    obj = math_model.obj_x + math_model.obj_y

    # todo replace coo_matrix by 3 lists
    con_res = scipy.sparse.hstack((math_model.con_res_x, math_model.con_res_y))
    con_exe = scipy.sparse.hstack((math_model.con_exe_x, sparse.coo_matrix((math_model.con_exe_x.shape[0], math_model.num_y))))
    con_wait = scipy.sparse.hstack((sparse.coo_matrix((math_model.con_wait_y.shape[0], math_model.num_x)), math_model.con_wait_y))
    con_en = scipy.sparse.hstack((math_model.con_en_x, math_model.con_en_y))

    rhs = math_model.con_res_b + math_model.con_en_b + math_model.con_exe_b + math_model.con_wait_b
    senses = 'E'*(con_res.shape[0]+con_en.shape[0]+con_exe.shape[0]) + 'L'*con_wait.shape[0]
    my_constraints = scipy.sparse.vstack((con_res, con_en, con_exe, con_wait))

    if hasattr(math_model, 'con_exe_order_x'):
        con_exe_order = scipy.sparse.hstack((math_model.con_exe_order_x, sparse.coo_matrix((math_model.con_exe_order_x.shape[0], math_model.num_y))))
        my_constraints = scipy.sparse.vstack((my_constraints, con_exe_order))
        rhs += math_model.con_exe_order_b
        senses += 'L'*math_model.con_exe_order_x.shape[0]

    con_rows = my_constraints.row.tolist()
    con_cols = my_constraints.col.tolist()
    con_vals = my_constraints.data.tolist()
    return [obj, lb, ub, con_rows, con_cols, con_vals, rhs, senses]


def setup_cplex(obj, lb, ub, con_rows, con_cols, con_vals, rhs, senses):
    cplex_obj = cplex.Cplex()
    cplex_obj.set_log_stream(None)
    # my_prob.set_error_stream(None)
    # my_prob.set_warning_stream(None)
    cplex_obj.set_results_stream(None)

    cplex_obj.parameters.clocktype.set(2)
    cplex_obj.parameters.timelimit.set(CONFIG_CPLEX_MAX_CPU)
    cplex_obj.parameters.mip.tolerances.mipgap.set(CONFIG_CPLEX_EPSILON_OBJ)

    # solver.parameters.lpmethod.set(solver.parameters.lpmethod.values.primal)    # primal simplex
    # solver.parameters.lpmethod.set(solver.parameters.lpmethod.values.dual)      # dual simplex

    cplex_obj.linear_constraints.add(rhs=rhs, senses=senses)
    cplex_obj.objective.set_sense(cplex_obj.objective.sense.minimize)
    cplex_obj.variables.add(obj=obj, ub=ub, lb=lb)
    cplex_obj.linear_constraints.set_coefficients(zip(con_rows, con_cols, con_vals))
    return cplex_obj


def convert_to_cplex(math_model):
    """Convert to CPLEX object, without specifying variable types."""
    [obj, lb, ub, con_rows, con_cols, con_vals, rhs, senses] = convert_to_general_opt(math_model)
    prob = setup_cplex(obj, lb, ub, con_rows, con_cols, con_vals, rhs, senses)
    return prob


def solve_cplex(math_model):
    mip_prob = convert_to_cplex(math_model)
    var_types = 'B' * math_model.num_x + 'C' * math_model.num_y
    mip_prob.variables.set_types(zip(range(math_model.num_x + math_model.num_y), var_types))

    t1 = time.time()
    mip_prob.solve()
    cpu_time = time.time() - t1

    status = mip_prob.solution.get_status()
    if status == 101 or status == 102 or status == 107:
        xx_value = mip_prob.solution.get_values()[0:math_model.num_x]
        yy_value = mip_prob.solution.get_values()[math_model.num_x:math_model.num_x+math_model.num_y]
        obj = mip_prob.solution.get_objective_value()
        rel_gap = mip_prob.solution.MIP.get_mip_relative_gap()
        relaxed_obj = mip_prob.solution.MIP.get_best_objective()
    else:
        xx_value = [0]
        yy_value = [0]
        obj = 0
        rel_gap = 0
        relaxed_obj = 0
    result = {'status': status, 'xx': sparse.coo_matrix(xx_value), 'yy': sparse.coo_matrix(yy_value), 'obj': obj,
              'rel_gap': rel_gap,
              'best_relaxed_obj': relaxed_obj,
              # 'cut_off': mip_prob.solution.MIP.get_cutoff(),
              # 'num_iter': mip_prob.solution.progress.get_num_iterations(),
              # 'nodes_processed': mip_prob.solution`.progress.get_num_nodes_processed(),
              # 'nodes_remained': mip_prob.solution.progress.get_num_nodes_remaining(),
              'cpu_time': cpu_time}
    return result
    # if status == 101 or status == 102:
    #     return result
    # else:
    #     print 'cpu_time %d' % cpu_time
    #     return None


