import random
import numpy as np


def the_dirichlet_process_method(num_HR, d, d_subspace, n, v, ll, ul):
    assert d_subspace <= d
    HR_ll = np.zeros((num_HR, d))
    HR_ul = np.ones((num_HR, d))
    Z = random.choices(range(n), k=num_HR * d_subspace)
    for i in range(num_HR):
        indices = random.sample(range(d), d_subspace)
        for j in range(d_subspace):
            z = Z[d_subspace * i + j]
            if z == 0:
                pass
            else:
                HR_ll[i, indices[j]] = np.random.beta(v * z, v * (n - z))
            if z == n - 1:
                pass
            else:
                HR_ul[i, indices[j]] = HR_ll[i, indices[j]] + \
                                       (1 - HR_ll[i, indices[j]]) * \
                                       np.random.beta(v, v * (n - 1 - z))
    HR_ll = HR_ll * (ul - ll) + ll
    HR_ul = HR_ul * (ul - ll) + ll
    return HR_ll, HR_ul


def calculate_design_matrix(HR_ll, HR_ul, X):
    num_data = np.size(X, 0)
    Phi = np.zeros((num_data, np.size(HR_ll, 0)), dtype=np.int32)
    for i in range(num_data):
        Phi[i, :] = np.all(
            np.logical_and(HR_ll <= X[i], X[i] <= HR_ul),
            axis=1
        )
    return Phi


def detect_overlap(HR_ll, HR_ul):
    num_HR = np.size(HR_ll, 0)
    Overlap = np.zeros((num_HR, num_HR), dtype=np.int8)
    for i in range(num_HR):
        for j in range(i + 1, num_HR):
            if not (np.any(HR_ul[i, :] < HR_ll[j, :]) or
                    np.any(HR_ul[j, :] < HR_ll[i, :])):
                # i-th hyperrectangle and j-th hyperrectangle have overlap.
                Overlap[i, j] = 1
                if np.all(HR_ll[i, :] <= HR_ll[j, :]) and \
                        np.all(HR_ul[j, :] <= HR_ul[i, :]):
                    # i-th hyperrectangle contains j-th hyperrectangle.
                    Overlap[i, j] = 2
                if np.all(HR_ll[j, :] <= HR_ll[i, :]) and \
                        np.all(HR_ul[i, :] <= HR_ul[j, :]):
                    # j-th hyperrectangle contains i-th hyperrectangle.
                    Overlap[i, j] = 3
    return Overlap


def add_overlap_to_set_of_hyperrectangles(HR_ll, HR_ul, X, Phi, Overlap):
    num_HR = np.size(HR_ll, 0)
    d = np.size(HR_ll, 1)
    for i in range(num_HR):
        for j in range(i + 1, num_HR):
            if Overlap[i, j] == 1:
                max_HR_ll_ij = np.max(
                    np.concatenate(
                        (HR_ll[i, :].reshape(1, d), HR_ll[j, :].reshape(1, d)),
                        axis=0
                    ),
                    axis=0
                )
                min_HR_ul_ij = np.min(
                    np.concatenate(
                        (HR_ul[i, :].reshape(1, d), HR_ul[j, :].reshape(1, d)),
                        axis=0),
                    axis=0
                )
                Phi_ij = calculate_design_matrix(
                    max_HR_ll_ij.reshape(1, d),
                    min_HR_ul_ij.reshape(1, d),
                    X
                )
                if np.sum(np.sum(Phi_ij, axis=0)):
                    Phi = np.concatenate((Phi, Phi_ij), axis=1)
                    HR_ll = \
                        np.append(HR_ll, max_HR_ll_ij.reshape(1, d), axis=0)
                    HR_ul = \
                        np.append(HR_ul, min_HR_ul_ij.reshape(1, d), axis=0)
                else:
                    # Overlap of i-th and j-th hyperrectangle
                    # does not contain any points.
                    Overlap[i, j] = 4
    return HR_ll, HR_ul, Phi, Overlap


def random_subspace_coding(num_HR, d, d_subspace, n, v,
                           ll, ul, X, num_regenerate):
    HR_ll, HR_ul = the_dirichlet_process_method(
        num_HR, d, d_subspace, n, v, ll, ul
    )
    HR_ll[-1, :] = ll
    HR_ul[-1, :] = ul
    Phi = calculate_design_matrix(HR_ll, HR_ul, X)
    sum_of_each_column = np.sum(Phi, axis=0)
    indices_of_meaningless_HR = \
        np.argwhere(sum_of_each_column == 0).reshape(-1)
    sum_of_num_meaningless_HR = np.size(indices_of_meaningless_HR)
    i = 0
    max_i = sum_of_num_meaningless_HR
    if sum_of_num_meaningless_HR != 0:
        for j in range(num_regenerate):

            HR_ll_tmp, HR_ul_tmp = the_dirichlet_process_method(
                num_HR, d, d_subspace, n, v, ll, ul
            )
            Phi_tmp = calculate_design_matrix(HR_ll_tmp, HR_ul_tmp, X)
            sum_of_each_column_tmp = np.sum(Phi_tmp, axis=0)
            indices_of_meaningful_HR_tmp = \
                np.argwhere(sum_of_each_column_tmp > 0).reshape(-1)
            sum_of_num_meaningless_HR += \
                num_HR - np.size(indices_of_meaningful_HR_tmp)

            if i < max_i:
                for k in indices_of_meaningful_HR_tmp:
                    HR_ll[indices_of_meaningless_HR[i], :] = HR_ll_tmp[k, :]
                    HR_ul[indices_of_meaningless_HR[i], :] = HR_ul_tmp[k, :]
                    Phi[:, indices_of_meaningless_HR[i]] = Phi_tmp[:, k]
                    i += 1
                    if i == max_i:
                        break
            if i == max_i:
                break
    return HR_ll, HR_ul, Phi, sum_of_num_meaningless_HR
