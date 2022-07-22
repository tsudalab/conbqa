# conbqa

conbqa is a library for converting continuous black-box optimization into quadratic unconstrained binary optimization (QUBO) so that Ising machines and other QUBO solvers may be applied.

Based on a dataset of several input and output values, the `CONBQA` class instance trains a regression model of the objective function on a coarse-grained random subspace.
The resulting model with appropriate constraints is converted into a QUBO matrix, by solving which we can tell which subspace is the promising area for the next evaluation. Once a new evaluation is made, the regression model is trained again and the loop goes on.

## Install
conbqa package depends on numpy, and scikit-learn.
IBM's cplex package is an optional dependency (Please refer [the product page](https://www.ibm.com/products/ilog-cplex-optimization-studio) for details).

Once you satisfied the dependency, download the repository and run `pip install .` inside the directory.

```console
$ git clone https://github.com/tsudalab/conbqa.git
$ cd conbqa
$ pip install .
```

## Example
Here shows an example case of minimizing a benchmarking function, Hartmann-6.
We firstly load all the modules and define the objective function as a callable object `h6`.

```python
import conbqa
import dimod
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler
import numpy as np

class _H6():
    '''
    Hartmann 6-Dimensional function
    Based on the MATLAB code available on
    https://www.sfu.ca/~ssurjano/hart6.html
    '''
    def __init__(self):
        self.lower_bounds = np.array([0, 0, 0, 0, 0, 0])
        self.upper_bounds = np.array([1, 1, 1, 1, 1, 1])
        self.min = np.array([
            0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573
        ])
        self.fmin = -3.32237
        self.ndim = 6

        self.alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self.A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        self.P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])

    def __call__(self, xx : np.ndarray):
        if len(xx.shape) == 2:
            return np.apply_along_axis(self, 1, xx)

        assert len(xx) == 6, "Input vector should be 6 dimensional."

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = xx[jj]
                Aij = self.A[ii, jj]
                Pij = self.P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2
            new = self.alpha[ii] * np.exp(-inner)
            outer = outer + new

        return -outer

h6 = _H6()
```

The `h6` takes its minimum value $-3.32237$ at point $(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)\in\mathbb{R}^6$.

We evaluate the objective function on 10 scattered points to get initial dataset:

```python
num_init = 10
X = np.stack([
    np.linspace(0, 1, num_init)[np.random.permutation(num_init)]
    for _ in range(h6.ndim)
], axis=1)
Y = h6(X)
```

The `CONBQA` object is generated by providing initial dataset and the lower and upper bounds of the input.
We have to tell it that we are aiming at minimization.

```python
model = conbqa.CONBQA(X, Y, h6.lower_bounds, h6.upper_bounds, maximization=False)
```

In this example, we encode the search space into 60-bits binary space, and make the regression model be linear model.

```python
model.set_encoding_method_to_RandomSubspaceCoding(60, v=20.0)
model.set_learning_model_to_linear()
```
The parameter `v=20.0` modifies the random subspace coding by dirichlet process method. See references for details.

Then, run 190 times of guessing and evaluation.
We use [dwave-neal](https://docs.ocean.dwavesys.com/projects/neal/en/latest/)'s simulated annealing sampler for solving the QUBO.
Note that the `CONBQA` instance has a method `decode` to convert the obtained binary vector back to continuous vector.
The `add_data` method appends the new data point into the training dataset.

```python
solver = SimulatedAnnealingSampler()

nepoch = 190
for i in range(nepoch):
    model.generate_hyperrectangles_and_encode()
    model.learn_weights()
    q = model.convert_maximization_of_learned_function_into_qubo()
    bqm = dimod.BQM.from_qubo(q)
    res = solver.sample(bqm)
    new_z = res.record['sample'][0]
    new_x = model.decode(new_z)
    new_y = h6(new_x)
    model.add_data(new_x, new_y)
    # progress bar
    print("\r", "="*int((i+1) / nepoch * 50), ">", "."*(50 - int((i+1) / nepoch * 50)), end="", sep="")
```

We would compare the result with random uniform sampling.

```python
randX = np.random.uniform(size=(200, 6))
randY = h6(randX)
```

Plot the results.

```python
plt.figure(figsize=(8,3), dpi=90)

plt.subplot(1,2,1)
plt.plot(pd.DataFrame(model.y).cummin())
plt.plot(model.y, '.')
plt.ylim(h6.fmin, 0.2)
plt.title("conbqa")
plt.text(100, -3.0, "mean={:.3f}".format(np.mean(model.y)))
plt.ylabel("Objective value")
plt.xlabel("Sampling No.")

plt.subplot(1,2,2)
plt.plot(pd.DataFrame(randY).cummin())
plt.plot(randY, '.')
plt.ylim(h6.fmin, 0.2)
plt.title("random")
plt.text(100, -3.0, "mean={:.3f}".format(np.mean(randY)))
plt.xlabel("Sampling No.")

plt.show()
```

![image](https://user-images.githubusercontent.com/15908202/180357096-f479c7a5-2fe7-40d7-b7eb-9fe7ce72c007.png)

We can see conbqa is generally picking the lower values than random sampling.

## License

The conbqa package is licensed under the MIT "Expat" License.

## Citation

If you find conbqa useful, please cite the article [*Phys. Rev. Research* 4, 023062 (2022).](https://link.aps.org/doi/10.1103/PhysRevResearch.4.023062)

```
@article{izawa_continuous_2022,
	title = {Continuous black-box optimization with an {Ising} machine and random subspace coding},
	volume = {4},
	issn = {2643-1564},
	url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.023062},
	doi = {10.1103/PhysRevResearch.4.023062},
	language = {en},
	number = {2},
	journal = {Physical Review Research},
	author = {Izawa, Syun and Kitai, Koki and Tanaka, Shu and Tamura, Ryo and Tsuda, Koji},
	month = apr,
	year = {2022},
	pages = {023062},
}
```
