from robyn.modeling.ridge.models.ridge_utils import (
    create_ridge_model_native,
    create_ridge_model_rpy2,
)

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if rpy2 is available
try:
    import rpy2

    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False


@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data for testing"""
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=8, noise=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 not available")
def test_ridge_implementations_equality(synthetic_data):
    """Test mathematical equality between rpy2 and native implementations with constraints"""
    X_train, X_test, y_train, y_test = synthetic_data

    # Test parameters including constraints
    lambda_value = 0.1
    n_features = X_train.shape[1]

    # Define constraints to test all three constraint types
    lower_limits = np.full(n_features, -0.5)  # Lower limit of -0.5 for all coefficients
    upper_limits = np.full(n_features, 0.5)  # Upper limit of 0.5 for all coefficients
    penalty_factor = np.ones(n_features)  # Start with equal penalty for all features
    penalty_factor[0:3] = 0.5  # Reduce penalty for first 3 features

    # Create models with both implementations
    model_rpy2 = create_ridge_model_rpy2(
        lambda_value=lambda_value,
        n_samples=X_train.shape[0],
        fit_intercept=True,
        standardize=True,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        intercept=True,
        intercept_sign="non_negative",
        penalty_factor=penalty_factor,
    )

    model_native = create_ridge_model_native(
        lambda_value=lambda_value,
        n_samples=X_train.shape[0],
        fit_intercept=True,
        standardize=True,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        intercept=True,
        intercept_sign="non_negative",
        penalty_factor=penalty_factor,
    )

    # Fit both models
    model_rpy2.fit(X_train, y_train)
    model_native.fit(X_train, y_train)

    # Compare coefficients
    coef_rpy2 = model_rpy2.coef_
    coef_native = model_native.coef_

    # Get full coefficients (including intercept)
    full_coef_rpy2 = model_rpy2.get_full_coefficients()
    full_coef_native = model_native.get_full_coefficients()

    # Make predictions
    pred_rpy2 = model_rpy2.predict(X_test)
    pred_native = model_native.predict(X_test)

    # Calculate performance metrics
    mse_rpy2 = mean_squared_error(y_test, pred_rpy2)
    mse_native = mean_squared_error(y_test, pred_native)
    r2_rpy2 = r2_score(y_test, pred_rpy2)
    r2_native = r2_score(y_test, pred_native)

    # Log results for inspection
    logger.info("=== Implementation Comparison Results ===")
    logger.info(f"R model intercept: {model_rpy2.intercept_}")
    logger.info(f"Python model intercept: {model_native.intercept_}")
    logger.info(
        f"Max coefficient difference: {np.max(np.abs(coef_rpy2 - coef_native))}"
    )
    logger.info(
        f"Mean coefficient difference: {np.mean(np.abs(coef_rpy2 - coef_native))}"
    )
    logger.info(f"Max prediction difference: {np.max(np.abs(pred_rpy2 - pred_native))}")
    logger.info(
        f"Mean prediction difference: {np.mean(np.abs(pred_rpy2 - pred_native))}"
    )

    # Assertions for coefficients
    assert np.allclose(
        coef_rpy2, coef_native, rtol=0.2, atol=0.2
    ), "Coefficients from R and Python models differ significantly"

    assert np.allclose(
        full_coef_rpy2, full_coef_native, rtol=0.2, atol=0.2
    ), "Full coefficients from R and Python models differ significantly"

    # Assertions for predictions - use more relaxed tolerances to account for algorithmic differences
    assert np.allclose(
        pred_rpy2, pred_native, rtol=0.2, atol=0.2
    ), "Predictions from R and Python models differ significantly"

    # Assertions for performance metrics
    assert np.isclose(
        mse_rpy2, mse_native, rtol=0.2
    ), "MSE values differ significantly between implementations"
    assert np.isclose(
        r2_rpy2, r2_native, rtol=0.2
    ), "R² values differ significantly between implementations"

    # Check that constraints were respected in both models
    assert np.all(coef_rpy2 >= lower_limits - 1e-6), "R model violated lower limits"
    assert np.all(coef_rpy2 <= upper_limits + 1e-6), "R model violated upper limits"
    assert model_rpy2.intercept_ >= 0, "R model violated non-negative intercept"

    assert np.all(
        coef_native >= lower_limits - 1e-6
    ), "Python model violated lower limits"
    assert np.all(
        coef_native <= upper_limits + 1e-6
    ), "Python model violated upper limits"
    assert model_native.intercept_ >= 0, "Python model violated non-negative intercept"


def test_constraint_effectiveness(synthetic_data):
    """Test that constraints actually affect the model in the expected way"""
    X_train, X_test, y_train, y_test = synthetic_data

    # Scale y to encourage larger coefficients
    y_scaled = y_train * 10

    # Parameters
    lambda_value = 0.01  # Low lambda to allow large coefficients
    n_features = X_train.shape[1]

    # Define tight constraints to clearly show their effect
    tight_limit = 0.15
    lower_limits = np.full(n_features, -tight_limit)
    upper_limits = np.full(n_features, tight_limit)

    # Create models with and without constraints
    model_unconstrained = create_ridge_model_native(
        lambda_value=lambda_value, fit_intercept=True, standardize=True
    )

    model_constrained = create_ridge_model_native(
        lambda_value=lambda_value,
        fit_intercept=True,
        standardize=True,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    # Fit both models
    model_unconstrained.fit(X_train, y_scaled)
    model_constrained.fit(X_train, y_scaled)

    # Get coefficients
    coef_unconstrained = model_unconstrained.coef_
    coef_constrained = model_constrained.coef_

    # Log coefficient ranges
    logger.info("=== Constraint Test Results ===")
    logger.info(
        f"Unconstrained coefficient range: [{coef_unconstrained.min()}, {coef_unconstrained.max()}]"
    )
    logger.info(
        f"Constrained coefficient range: [{coef_constrained.min()}, {coef_constrained.max()}]"
    )

    # Verify the unconstrained model has coefficients outside the limits
    assert np.any(
        np.abs(coef_unconstrained) > tight_limit
    ), "Unconstrained model should have coefficients outside the tight limits"

    # Verify the constrained model respects the limits
    assert np.all(
        coef_constrained >= -tight_limit - 1e-6
    ), "Constrained model violated lower limits"
    assert np.all(
        coef_constrained <= tight_limit + 1e-6
    ), "Constrained model violated upper limits"

    # Verify the models produce different coefficients
    assert not np.allclose(
        coef_unconstrained, coef_constrained, rtol=1e-2, atol=1e-2
    ), "Constrained and unconstrained models should produce different coefficients"


def test_penalty_factor_effectiveness(synthetic_data):
    """Test that penalty_factor has the expected effect on coefficients"""
    X_train, X_test, y_train, y_test = synthetic_data

    # Parameters
    lambda_value = 0.5  # Higher lambda to make penalty effect more noticeable
    n_features = X_train.shape[1]

    # Create models with different penalty factors
    model_uniform = create_ridge_model_native(
        lambda_value=lambda_value, fit_intercept=True, standardize=True
    )

    # Differential penalty: first half gets lower penalty, second half gets higher
    penalty_factor = np.ones(n_features)
    penalty_factor[: n_features // 2] = 0.1  # Much lower penalty
    penalty_factor[n_features // 2 :] = 2.0  # Higher penalty

    model_differential = create_ridge_model_native(
        lambda_value=lambda_value,
        fit_intercept=True,
        standardize=True,
        penalty_factor=penalty_factor,
    )

    # Fit both models
    model_uniform.fit(X_train, y_train)
    model_differential.fit(X_train, y_train)

    # Get absolute coefficient values
    uniform_abs_coef = np.abs(model_uniform.coef_)
    differential_abs_coef = np.abs(model_differential.coef_)

    # Calculate mean coefficient magnitudes by groups
    low_penalty_mean = np.mean(differential_abs_coef[: n_features // 2])
    high_penalty_mean = np.mean(differential_abs_coef[n_features // 2 :])
    uniform_first_half_mean = np.mean(uniform_abs_coef[: n_features // 2])
    uniform_second_half_mean = np.mean(uniform_abs_coef[n_features // 2 :])

    # Log results
    logger.info("=== Penalty Factor Test Results ===")
    logger.info(f"Uniform model - first half mean: {uniform_first_half_mean}")
    logger.info(f"Uniform model - second half mean: {uniform_second_half_mean}")
    logger.info(f"Differential model - low penalty mean: {low_penalty_mean}")
    logger.info(f"Differential model - high penalty mean: {high_penalty_mean}")

    # Verify low penalty features have larger coefficients than high penalty features
    assert (
        low_penalty_mean > high_penalty_mean
    ), "Features with lower penalty should have larger coefficients on average"

    # Calculate ratio of change compared to uniform model
    low_penalty_ratio = low_penalty_mean / uniform_first_half_mean
    high_penalty_ratio = high_penalty_mean / uniform_second_half_mean

    logger.info(f"Low penalty ratio: {low_penalty_ratio}")
    logger.info(f"High penalty ratio: {high_penalty_ratio}")

    # Verify penalty factors had significant effect compared to uniform model
    assert (
        low_penalty_ratio > 1.0
    ), "Features with lower penalty should have larger coefficients than in uniform model"
    assert (
        high_penalty_ratio < 1.0
    ), "Features with higher penalty should have smaller coefficients than in uniform model"


if __name__ == "__main__":
    pytest.main()
