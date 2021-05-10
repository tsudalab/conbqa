import numpy as np


def obtain_qubo_dict_linear(w, Overlap, A_qubo, B_qubo):
    num_HR = w.size
    if A_qubo is None:
        A_qubo = 1 / np.max(w)
    Q_dict = dict()
    for i in range(num_HR):
        for j in range(i, num_HR):
            if i == j:
                Q_dict[i, i] = - A_qubo * w[i]
            else:
                # i-th hyperrectangle and j-th hyperrectangle
                # do not have overlap.
                if Overlap[i, j] == 0:
                    Q_dict[i, j] = B_qubo
    return Q_dict


def obtain_qubo_dict_quadratic(w, Overlap, num_HR, A_qubo, B_qubo):
    Q_dict = dict()
    count = 0
    for i in range(num_HR):
        for j in range(i, num_HR):
            if i == j:
                Q_dict[i, i] = - A_qubo * w[i]
            else:
                # i-th hyperrectangle and j-th hyperrectangle
                # have overlap.
                if Overlap[i, j] == 1:
                    Q_dict[i, j] = - w[num_HR + count]
                    count += 1
                # i-th hyperrectangle and j-th hyperrectangle
                # do not have overlap.
                elif Overlap[i, j] == 0:
                    Q_dict[i, j] = B_qubo
    return Q_dict
