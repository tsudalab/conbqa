import random
import numpy as np
import cplex


def MaximumPosterior(transformed_y, Phi):
    num_weight = np.size(Phi, 1)
    qmat = np.dot(np.transpose(Phi), Phi).astype(np.float64)
    obj = (- np.dot(np.transpose(Phi), transformed_y)).tolist()
    cplex_ins = cplex.Cplex()
    cplex_ins.objective.set_sense(cplex_ins.objective.sense.minimize)
    cplex_ins.variables.add(obj=obj,
                            names=['w' + str(i) for i in range(num_weight)])
    cplex_ins.objective.set_quadratic(
        [
            [
                [j for j in range(num_weight) if qmat[i, j]],
                [qmat[i, k] for k in range(num_weight) if qmat[i, k]]
            ]
            for i in range(num_weight)
        ])
    cplex_ins.solve()
    w = np.array(cplex_ins.solution.get_values())
    return w


def MetropolisHastings(transformed_y, Phi, w, burnin_steps,
                       sigma, rate, max_step_width):
    num_weight = len(w)
    choices = random.choices(range(num_weight), k=burnin_steps)
    estimated_y_old = np.dot(Phi, w)
    log_likelihood_old = - (np.linalg.norm(estimated_y_old - transformed_y,
                                           ord=2) ** 2) / \
                           (2.0 * (sigma ** 2))
    for i in range(burnin_steps):
        j = choices[i]
        w_j_old = w[j]
        log_posterior_old = log_likelihood_old - rate * w_j_old
        if w_j_old > max_step_width:
            w_j_new = w_j_old + \
                      2.0 * max_step_width * np.random.rand() - max_step_width
            log_proposal_old_to_new = np.log(1.0 / (2.0 * max_step_width))
        else:
            w_j_new = (max_step_width + w_j_old) * np.random.rand()
            log_proposal_old_to_new = np.log(1.0 / (max_step_width + w_j_old))
        estimated_y_new = estimated_y_old + (w_j_new - w_j_old) * Phi[:, j]
        log_likelihood_new = - (np.linalg.norm(estimated_y_new - transformed_y,
                                               ord=2) ** 2) / \
                               (2.0 * (sigma ** 2))
        log_posterior_new = log_likelihood_new - rate * w_j_new
        if w_j_new > max_step_width:
            log_proposal_new_to_old = np.log(1.0 / (2.0 * max_step_width))
        else:
            log_proposal_new_to_old = np.log(1.0 / (max_step_width + w_j_new))
        if (log_posterior_new + log_proposal_new_to_old -
            log_posterior_old - log_proposal_old_to_new) >= \
                0:
            w[j] = w_j_new
            estimated_y_old = estimated_y_new
            log_likelihood_old = log_likelihood_new
        elif (log_posterior_new + log_proposal_new_to_old -
              log_posterior_old - log_proposal_old_to_new) >= \
                np.log(np.random.rand()):
            w[j] = w_j_new
            estimated_y_old = estimated_y_new
            log_likelihood_old = log_likelihood_new
    return w
