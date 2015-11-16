"""

__author__ = xxxzhang
"""

import time

import scipy
from scipy.sparse import coo_matrix
import cplex


CONFIG_CPLEX_MAX_CPU = 7200
CONFIG_CPLEX_EPSILON_OBJ = 1e-4


def convert_to_general_opt(math_model):
    ub = [1]*math_model.num_x + math_model.y_up
    lb = [0]*math_model.num_x + math_model.y_lo
    obj = math_model.obj_x + math_model.obj_y
    # todo replace coo_matrix by 3 lists
    con_res = scipy.sparse.hstack((math_model.con_res_x, math_model.con_res_y))
    con_exe = scipy.sparse.hstack((math_model.con_exe_x, coo_matrix((math_model.con_exe_x.shape[0], math_model.num_y))))
    con_wait = scipy.sparse.hstack((coo_matrix((math_model.con_wait_y.shape[0], math_model.num_x)), math_model.con_wait_y))
    con_en = scipy.sparse.hstack((math_model.con_en_x, math_model.con_en_y))
    my_constraints = scipy.sparse.vstack((con_res, con_en, con_exe, con_wait))
    con_rows = my_constraints.row.tolist()
    con_cols = my_constraints.col.tolist()
    con_vals = my_constraints.data.tolist()
    rhs = math_model.con_res_b + math_model.con_en_b + math_model.con_exe_b + math_model.con_wait_b
    senses = 'E'*(con_res.shape[0]+con_en.shape[0]+con_exe.shape[0]) + 'L'*con_wait.shape[0]
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
    xx_value = mip_prob.solution.get_values()[0:math_model.num_x]
    yy_value = mip_prob.solution.get_values()[math_model.num_x+1:math_model.num_x+math_model.num_y]
    obj = mip_prob.solution.get_objective_value()
    result = {'status': status, 'xx': xx_value, 'yy': yy_value, 'obj': obj,
              'cpu_time': cpu_time,
              'rel_gap': mip_prob.solution.MIP.get_mip_relative_gap(),
              'best_relaxed_obj': mip_prob.solution.MIP.get_best_objective(),
              'cut_off': mip_prob.solution.MIP.get_cutoff(),
              'num_iter': mip_prob.solution.progress.get_num_iterations(),
              'nodes_processed': mip_prob.solution.progress.get_num_nodes_processed(),
              'nodes_remained': mip_prob.solution.progress.get_num_nodes_remaining()}

    if status == 101 or status == 102:
        return result
    else:
        return None


