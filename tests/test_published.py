"""
The linear solve against worked examples whose answers are printed in a textbook.

Every other reference answer in the tests is computed beside the code. These
two are numbers someone else printed, from Aster, Borchers and Thurber,
*Parameter Estimation and Inverse Problems*, third edition (Elsevier, 2019,
ISBN 978-0-12-804651-7):

- **Example 2.1** (pp. 32-34), a ballistic trajectory: ten heights observed
  at t = 1, ..., 10 s, each with a standard deviation of 8 m, and three
  parameters of ``y = m_1 + m_2 t - m_3 t^2 / 2``. The book prints the
  forward matrix (its eq. 2.37), the estimate (eq. 2.40, to two decimals),
  its covariance (eq. 2.39, to two decimals), its 95% half-widths (eq. 2.40)
  and the chi-squared, "approximately 4.2". The heights are not printed: they
  are the authors' file ``data1.mat`` in the book's companion code
  (``Examples/chap2/ex_2_1_2`` of github.com/brianborchers/PEIP), to four
  decimals, and are copied here. The covariance and the half-widths do not
  depend on them. The example is weighted least squares, a posterior with a
  flat prior; here the prior is centered on zero with a standard deviation of
  10^4 in every parameter, which moves the estimate by about 10^-6 of itself
  and the covariance by about 10^-6, far below the printed digits.
- **Example 11.1** (pp. 284-285), a mass weighed twice with a standard
  deviation of 1 ug: the first weighing, 10.3 ug, is the prior
  ``N(10.3, 1)`` of the second, 10.1 ug, and the posterior is printed as
  ``N(10.2, (1/sqrt(2))^2)`` (eq. 11.20). A proper Gaussian prior, in one
  dimension.

Both run in both forms the analytic method factors in.
"""

import numpy as np
import pandas as pd
import pytest

import fips
from fips import Covariance, Operator, Problem, Space, Vector

# Example 2.1: the heights from the authors' data1.mat, in meters
TIMES = np.arange(1.0, 11.0)
HEIGHTS = np.array(
    [
        109.3827,
        187.5385,
        267.5319,
        331.8753,
        386.0535,
        428.4271,
        452.1644,
        498.1461,
        512.3499,
        512.9753,
    ]
)
SIGMA = 8.0

# what the book prints
ESTIMATE = np.array([16.42, 96.97, 9.41])  # eq. 2.40
COVARIANCE = np.array(  # eq. 2.39
    [
        [88.53, -33.60, -5.33],
        [-33.60, 15.44, 2.67],
        [-5.33, 2.67, 0.48],
    ]
)
HALF_WIDTHS = np.array([18.44, 7.70, 1.36])  # eq. 2.40, 1.96 standard deviations
FORWARD = np.array(  # eq. 2.37, the third column
    [-0.5, -2.0, -4.5, -8.0, -12.5, -18.0, -24.5, -32.0, -40.5, -50.0]
)


@pytest.mark.usefixtures("analytic_form")
def test_aster_example_2_1_ballistic_trajectory():
    """The estimate, its covariance, its 95% half-widths and the chi-squared are the book's, to its printed digits."""
    params = Space({"m": pd.Index(["m1", "m2", "m3"], name="parameter")})
    heights = Space({"y": pd.Index(TIMES, name="t")})
    G = np.column_stack([np.ones(10), TIMES, -0.5 * TIMES**2])
    np.testing.assert_array_equal(G[:, 2], FORWARD)
    post = fips.solve(
        Problem(
            prior=Vector(np.zeros(3), params),
            prior_cov=Covariance(np.diag(np.full(3, 1e4**2)), params),
            obs=Vector(HEIGHTS, heights),
            obs_cov=Covariance(np.diag(np.full(10, SIGMA**2)), heights),
            forward=Operator(G, heights, params),
        )
    )
    # each to half a unit of its last printed digit
    np.testing.assert_allclose(post.estimate.values, ESTIMATE, rtol=0, atol=0.005)
    np.testing.assert_allclose(post.cov().dense(), COVARIANCE, rtol=0, atol=0.005)
    std = np.sqrt(post.variances.values)
    np.testing.assert_allclose(1.96 * std, HALF_WIDTHS, rtol=0, atol=0.005)
    assert round(post.data_misfit, 1) == 4.2


@pytest.mark.usefixtures("analytic_form")
def test_aster_example_11_1_two_weighings():
    """A prior of N(10.3, 1) and a weighing of 10.1 with error 1 give the book's posterior, N(10.2, 1/2)."""
    mass = Space({"m": pd.Index(["mass"], name="object")})
    weighing = Space({"d": pd.RangeIndex(1, name="weighing")})
    post = fips.solve(
        Problem(
            prior=Vector([10.3], mass),
            prior_cov=Covariance(np.array([[1.0]]), mass),
            obs=Vector([10.1], weighing),
            obs_cov=Covariance(np.array([[1.0]]), weighing),
            forward=Operator(np.array([[1.0]]), weighing, mass),
        )
    )
    np.testing.assert_allclose(post.estimate.values, [10.2], rtol=1e-14)
    np.testing.assert_allclose(post.variances.values, [0.5], rtol=1e-14)
