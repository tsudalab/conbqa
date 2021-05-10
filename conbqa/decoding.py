import numpy as np


def obtain_intersection(solution_qubo, HR_ll, HR_ul):
    active_indices = np.argwhere(solution_qubo).reshape(-1)
    intersection_ll = np.max(HR_ll[active_indices, :], axis=0)
    intersection_ul = np.min(HR_ul[active_indices, :], axis=0)
    return intersection_ll, intersection_ul


def determine_candidate_point(intersection_ll, intersection_ul):
    candidate_point = np.random.rand(intersection_ll.size) * \
                      (intersection_ul - intersection_ll) + intersection_ll
    return candidate_point


def check_decodability(solution_qubo, HR_ll, HR_ul):
    intersection_ll, intersection_ul = obtain_intersection(
        solution_qubo,
        HR_ll,
        HR_ul
    )
    if np.all(intersection_ll <= intersection_ul):
        inactive_indices = np.argwhere(solution_qubo == 0).reshape(-1)
        for i in inactive_indices:
            if not (np.any(HR_ul[i, :] < intersection_ll) or
                    np.any(intersection_ul < HR_ll[i, :])):
                return 1
        return 2
    else:
        # 0 mean that intersection is empty
        # and solution_qubo is not decodable.
        return 0
