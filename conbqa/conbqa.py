import numpy as np

from . import encoding
from . import regression
from . import decoding
from . import qubo


class CONBQAError(Exception):
    pass


class CONBQA:
    """
    CONBQA : CONtinuous Black-box optimization with Quantum Annealing

    Args:
        X (numpy.ndarray):
            Input vectors.

        y (numpy.ndarray):
            Output values.

        ll (numpy.ndarray):
            Lower limit of each dimension.

        ul (numpy.ndarray):
            Upper limit of each dimension.
    """

    def __init__(self, X, y, ll, ul,
                 learning_model="linear",
                 encoding_method="RandomSubspaceCoding",
                 scaling_method="MinMaxNormalization",
                 regression_method="MetropolisHastings",
                 maximization=True):
        assert type(X) is np.ndarray
        assert X.ndim == 2
        self.num_initial_data, self.d = X.shape
        assert type(y) is np.ndarray
        assert y.ndim == 1
        assert self.num_initial_data == y.size
        assert type(ll) is np.ndarray
        assert ll.ndim == 1
        assert self.d == ll.size
        assert type(ul) is np.ndarray
        assert ul.ndim == 1
        assert self.d == ul.size
        assert type(learning_model) is str
        assert type(encoding_method) is str
        assert type(regression_method) is str
        assert type(maximization) is bool
        # =====================================================================
        # Args
        self.X = X
        self.y = y
        self.ll = ll
        self.ul = ul
        self.learning_model = learning_model
        self.encoding_method = encoding_method
        self.scaling_method = scaling_method
        self.regression_method = regression_method
        self.maximization = maximization
        # =====================================================================
        # initialize learning_method parameters
        self.A_qubo = None
        self.B_qubo = None
        # =====================================================================

        self.num_HR = None
        self.d_subspace = None
        self.n = None
        self.v = None
        self.num_regenerate = None
        self.w = None
        self.intersection_was_not_９empty = None
        self.candidate_point = None
        self.burnin_steps = None
        self.sigma = None
        self.rate = None
        self.max_step_width = None
        self.sampleset = None
        self.HR_ll = None
        self.HR_ul = None
        self.Phi = None
        # =====================================================================
        self.sum_meaningless_HR = None
        self.num_one = None
        self.intersection_ll = None
        self.intersection_ul = None
        self.mean_squared_error = None
        self.Overlap = None
        self.solution_qubo = None
        # =====================================================================
        if self.learning_model == "linear":
            self.set_learning_model_to_linear()
        elif self.learning_model == "quadratic":
            self.set_learning_model_to_quadratic()
        else:
            raise CONBQAError("Unexpected learnind_model")

        if self.encoding_method == "RandomSubspaceCoding":
            self.set_encoding_method_to_RandomSubspaceCoding()
        elif self.encoding_method == "OneHot":
            pass
        else:
            raise CONBQAError("Unexpected encoding_method")
        if self.regression_method == "MetropolisHastings":
            self.set_regression_method_to_MetropolisHastings()
        elif self.regression_method == "MaximumPosterior":
            pass
        else:
            raise CONBQAError("Unexpected regression_method")

    def set_learning_model_to_linear(self, A_qubo=None, B_qubo=1.0):
        self.learning_model = "linear"
        self.A_qubo = A_qubo
        self.B_qubo = B_qubo

    def set_learning_model_to_quadratic(self, A_qubo=1.0, B_qubo=2.0):
        self.learning_model = "quadratic"
        self.A_qubo = A_qubo
        self.B_qubo = B_qubo

    def set_encoding_method_to_RandomSubspaceCoding(self, num_HR=60,
                                                    d_subspace=None, n=3,
                                                    v=1.0, num_regenerate=10):
        self.encoding_method = "RandomSubspaceCoding"
        self.num_HR = num_HR
        if d_subspace is None:
            if self.d == 1:
                self.d_subspace = 1
            elif self.d >= 2:
                self.d_subspace = 2
        else:
            self.d_subspace = d_subspace
        self.n = n
        self.v = v
        self.num_regenerate = num_regenerate

    def set_regression_method_to_MetropolisHastings(self, burnin_steps=1000,
                                                    sigma=None, rate=0,
                                                    max_step_width=0.1):
        self.regression_method = "MetropolisHastings"
        self.burnin_steps = burnin_steps
        self.sigma = sigma
        self.rate = rate
        self.max_step_width = max_step_width
        if sigma is None:
            self.sigma = np.std(self.transform_y())

    def generate_hyperrectangles_and_encode(self):
        if self.encoding_method == "RandomSubspaceCoding":
            self.HR_ll, self.HR_ul, self.Phi, self.sum_meaningless_HR = \
                encoding.random_subspace_coding(
                    self.num_HR,
                    self.d,
                    self.d_subspace,
                    self.n,
                    self.v,
                    self.ll,
                    self.ul,
                    self.X,
                    self.num_regenerate
                )
        else:
            raise CONBQAError("Unexpected encoding_method")
        self.Overlap = encoding.detect_overlap(self.HR_ll, self.HR_ul)

        if self.learning_model == "linear":
            pass

        elif self.learning_model == "quadratic":
            self.HR_ll, self.HR_ul, self.Phi, self.Overlap = \
                encoding.add_overlap_to_set_of_hyperrectangles(
                    self.HR_ll,
                    self.HR_ul,
                    self.X,
                    self.Phi,
                    self.Overlap
                )
        else:
            raise CONBQAError("Unexpected learning_model")

    def transform_y(self):
        if self.scaling_method == "MinMaxNormalization":
            if self.maximization:
                transformed_y = (self.y - np.min(self.y)) \
                                / (np.max(self.y) - np.min(self.y))
            else:
                transformed_y = (-self.y - np.min(-self.y)) \
                                / (np.max(-self.y) - np.min(-self.y))
        elif self.scaling_method is None:
            if self.maximization:
                transformed_y = self.y
            else:
                transformed_y = - self.y
        else:
            raise CONBQAError("Unexpected scaling_method")
        return transformed_y

    def learn_weights(self):
        transformed_y = self.transform_y()
        if self.regression_method == "MaximumPosterior":
            self.w = regression.MaximumPosterior(transformed_y, self.Phi)
        elif self.regression_method == "MetropolisHastings":
            self.w = regression.MaximumPosterior(transformed_y, self.Phi)
            self.w = regression.MetropolisHastings(
                transformed_y,
                self.Phi,
                self.w,
                self.burnin_steps,
                self.sigma,
                self.rate,
                self.max_step_width
            )
        else:
            raise CONBQAError("Unexpected regression_method")

    def convert_maximization_of_learned_function_into_qubo(self):
        if self.learning_model == "linear":
            Q_dict = qubo.obtain_qubo_dict_linear(
                self.w,
                self.Overlap,
                self.A_qubo,
                self.B_qubo
            )
        elif self.learning_model == "quadratic":
            Q_dict = qubo.obtain_qubo_dict_quadratic(
                self.w,
                self.Overlap,
                self.num_HR,
                self.A_qubo,
                self.B_qubo
            )
        else:
            raise CONBQAError("Unexpected learning_model")
        return Q_dict

    def decode(self, solution_qubo):
        self.solution_qubo = solution_qubo
        self.intersection_ll, self.intersection_ul = \
            decoding.obtain_intersection(
                self.solution_qubo,
                self.HR_ll,
                self.HR_ul
            )
        if np.all(self.intersection_ll <= self.intersection_ul):
            self.intersection_was_not_empty = True
            self.candidate_point = decoding.determine_candidate_point(
                self.intersection_ll,
                self.intersection_ul
            )
        else:
            self.intersection_was_not_empty = False
            self.candidate_point = decoding.determine_candidate_point(
                self.ll,
                self.ul
            )
        return self.candidate_point

    def add_data(self, x, y):
        self.X = np.append(self.X, x.reshape(1, self.d), axis=0)
        self.y = np.append(self.y, y)

    def calculate_mean_squared_error(self):
        return ((np.dot(self.Phi, self.w) - self.transform_y()) ** 2).mean()
