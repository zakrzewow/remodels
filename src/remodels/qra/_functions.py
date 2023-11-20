"""QRA-computing helper functions."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.stats import iqr
from scipy.stats import norm


def _add_intercept(X: np.array) -> np.array:
    new_X = np.ones(shape=(X.shape[0], X.shape[1] + 1))
    new_X[:, 1:] = X
    return new_X


def _lqra(
    X, y, quantile: float = 0.5, lambda_: float = 0.0, fit_intercept: bool = False
):
    # input matrix X - N rows x K columns
    N, K = X.shape

    if fit_intercept:
        K += 1
        X = _add_intercept(X)

    # scaling regularization parameter to make it sample-independent
    lambda_ = N * lambda_

    c = np.concatenate(
        [
            np.full((2 * K,), fill_value=lambda_),
            np.full((N,), fill_value=quantile),
            np.full((N,), fill_value=1 - quantile),
        ]
    )
    if fit_intercept:  # do not penalize intercept
        c[0] = 0
        c[K] = 0

    b_eq = y
    # A_eq = np.concatenate([X, -X, np.eye(N), -np.eye(N)], axis=1)

    eye = sparse.eye(N, dtype=X.dtype, format="csc")
    A_eq = sparse.hstack([X, -X, eye, -eye], format="csc")

    optimize_result = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        method="highs",  # highs is an optimalized solver (simplex or interior-point)
    )

    # beta = beta+ - beta-
    beta = optimize_result.x[:K] - optimize_result.x[K : 2 * K]
    return beta


def _sqra(X, y, quantile: float = 0.5, H: float = None, fit_intercept=False):
    # SQRA can be solved with gradient-descent method (2022-he)
    # for now, TNC (non-linear) minimizer is used

    if fit_intercept:
        X = _add_intercept(X)

    beta__initial_guess = _lqra(X, y, quantile=quantile, lambda_=0, fit_intercept=False)
    if H is None:  # rule-of-thumb estimation
        residuals = y - (X @ beta__initial_guess)
        resid_std = np.std(residuals)
        resid_iqr = iqr(residuals)
        H = min(resid_std, resid_iqr / 1.38898) * 1.06 * (X.shape[0] ** (-1 / 5))

    def rho(beta):
        residuals = y - X @ beta
        return (
            np.sum(H * norm.pdf(residuals / H))
            + (quantile - norm.cdf(-residuals / H)) @ residuals
        )

    return minimize(rho, beta__initial_guess, method="TNC").x
