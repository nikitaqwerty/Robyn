from robyn.modeling.ridge.models.ridge_utils import (
    create_ridge_model_native,
    create_ridge_model_rpy2,
)

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import sys


# Use direct print statements that will always appear in pytest output
def print_header(title):
    print("\n" + "=" * 80)
    print(f"=== {title} ===")
    print("=" * 80)


# Check if rpy2 is available
try:
    import rpy2

    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False


@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data for testing"""
    print_header("GENERATING SYNTHETIC DATA")
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=8, noise=0.1, random_state=42
    )

    print(f"Generated data: X shape: {X.shape}, y shape: {y.shape}")
    print(
        f"X statistics: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}, std={X.std():.4f}"
    )
    print(
        f"y statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}"
    )

    # Print a few sample rows
    print("Sample X data (first 3 rows):")
    for i in range(min(3, X.shape[0])):
        print(f"Row {i}: {X[i]}")
    print(f"Sample y data (first 3 values): {y[:3]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(
        f"After train-test split: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    return X_train, X_test, y_train, y_test


@pytest.mark.skipif(not RPY2_AVAILABLE, reason="rpy2 not available")
def test_ridge_implementations_equality(synthetic_data):
    """Test mathematical equality between rpy2 and native implementations with constraints"""
    print_header("TESTING RIDGE IMPLEMENTATIONS EQUALITY")
    X_train, X_test, y_train, y_test = synthetic_data

    # Test parameters including constraints
    lambda_value = 0.1
    n_features = X_train.shape[1]

    # Define constraints to test all three constraint types
    lower_limits = np.full(n_features, -0.5)  # Lower limit of -0.5 for all coefficients
    upper_limits = np.full(n_features, 0.5)  # Upper limit of 0.5 for all coefficients
    penalty_factor = np.ones(n_features)  # Start with equal penalty for all features
    penalty_factor[0:3] = 0.5  # Reduce penalty for first 3 features

    # Print model parameters
    print(f"Model parameters:")
    print(f"  lambda_value: {lambda_value}")
    print(f"  n_features: {n_features}")
    print(f"  lower_limits: {lower_limits}")
    print(f"  upper_limits: {upper_limits}")
    print(f"  penalty_factor: {penalty_factor}")

    print("\n--- Creating RPy2 model ---")
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

    print("RPy2 model created")

    print("\n--- Creating Native model ---")
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

    print("Native model created")

    # Fit both models
    print("\n--- Fitting RPy2 model ---")
    model_rpy2.fit(X_train, y_train)
    print("RPy2 model fitted")

    print("\n--- Fitting Native model ---")
    model_native.fit(X_train, y_train)
    print("Native model fitted")

    # Compare coefficients
    coef_rpy2 = model_rpy2.coef_
    coef_native = model_native.coef_

    print("\n--- Coefficients Comparison ---")
    print("RPy2 coefficients:")
    for i, coef in enumerate(coef_rpy2):
        print(f"  Feature {i}: {coef:.6f}")

    print("\nNative coefficients:")
    for i, coef in enumerate(coef_native):
        print(f"  Feature {i}: {coef:.6f}")

    print("\nCoefficient differences (RPy2 - Native):")
    for i, (r_coef, n_coef) in enumerate(zip(coef_rpy2, coef_native)):
        print(f"  Feature {i}: {r_coef - n_coef:.6f}")

    # Get full coefficients (including intercept)
    full_coef_rpy2 = model_rpy2.get_full_coefficients()
    full_coef_native = model_native.get_full_coefficients()

    print("\n--- Full Coefficients (with intercept) ---")
    print(f"RPy2 full coefficients: {full_coef_rpy2}")
    print(f"Native full coefficients: {full_coef_native}")
    print(f"Full coefficient differences: {full_coef_rpy2 - full_coef_native}")

    # Make predictions
    print("\n--- Making Predictions ---")
    pred_rpy2 = model_rpy2.predict(X_test)
    pred_native = model_native.predict(X_test)

    print("Sample predictions (first 5 samples):")
    for i in range(min(5, len(pred_rpy2))):
        print(
            f"  Sample {i}: RPy2={pred_rpy2[i]:.4f}, Native={pred_native[i]:.4f}, Diff={pred_rpy2[i]-pred_native[i]:.4f}"
        )

    # Calculate performance metrics
    mse_rpy2 = mean_squared_error(y_test, pred_rpy2)
    mse_native = mean_squared_error(y_test, pred_native)
    r2_rpy2 = r2_score(y_test, pred_rpy2)
    r2_native = r2_score(y_test, pred_native)

    # Print results for inspection
    print("\n=== Implementation Comparison Results ===")
    print(f"R model intercept: {model_rpy2.intercept_}")
    print(f"Python model intercept: {model_native.intercept_}")
    print(f"Intercept difference: {model_rpy2.intercept_ - model_native.intercept_}")
    print(f"Max coefficient difference: {np.max(np.abs(coef_rpy2 - coef_native))}")
    print(f"Mean coefficient difference: {np.mean(np.abs(coef_rpy2 - coef_native))}")
    print(f"Max prediction difference: {np.max(np.abs(pred_rpy2 - pred_native))}")
    print(f"Mean prediction difference: {np.mean(np.abs(pred_rpy2 - pred_native))}")

    print("\n--- Performance Metrics ---")
    print(f"RPy2 MSE: {mse_rpy2:.6f}")
    print(f"Native MSE: {mse_native:.6f}")
    print(f"MSE difference: {mse_rpy2 - mse_native:.6f}")
    print(f"RPy2 R²: {r2_rpy2:.6f}")
    print(f"Native R²: {r2_native:.6f}")
    print(f"R² difference: {r2_rpy2 - r2_native:.6f}")

    # Constraint validation
    rpy2_lower_violated = np.any(coef_rpy2 < lower_limits - 1e-6)
    rpy2_upper_violated = np.any(coef_rpy2 > upper_limits + 1e-6)
    native_lower_violated = np.any(coef_native < lower_limits - 1e-6)
    native_upper_violated = np.any(coef_native > upper_limits + 1e-6)

    print("\n--- Constraint Validation ---")
    print(f"RPy2 lower limits violated: {rpy2_lower_violated}")
    print(f"RPy2 upper limits violated: {rpy2_upper_violated}")
    print(f"Native lower limits violated: {native_lower_violated}")
    print(f"Native upper limits violated: {native_upper_violated}")
    print(f"RPy2 non-negative intercept respected: {model_rpy2.intercept_ >= 0}")
    print(f"Native non-negative intercept respected: {model_native.intercept_ >= 0}")

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

    print("All assertions passed!")


def test_constraint_effectiveness(synthetic_data):
    """Test that constraints actually affect the model in the expected way"""
    print_header("TESTING CONSTRAINT EFFECTIVENESS")
    X_train, X_test, y_train, y_test = synthetic_data

    # Scale y to encourage larger coefficients
    y_scaled = y_train * 10
    print(f"Original y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"Scaled y_train range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")

    # Parameters
    lambda_value = 0.01  # Low lambda to allow large coefficients
    n_features = X_train.shape[1]

    # Define tight constraints to clearly show their effect
    tight_limit = 0.15
    lower_limits = np.full(n_features, -tight_limit)
    upper_limits = np.full(n_features, tight_limit)

    print(f"Constraint test parameters:")
    print(f"  lambda_value: {lambda_value}")
    print(f"  tight_limit: {tight_limit}")

    # Create models with and without constraints
    print("\n--- Creating unconstrained model ---")
    model_unconstrained = create_ridge_model_native(
        lambda_value=lambda_value, fit_intercept=True, standardize=True
    )

    print("\n--- Creating constrained model ---")
    model_constrained = create_ridge_model_native(
        lambda_value=lambda_value,
        fit_intercept=True,
        standardize=True,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    # Fit both models
    print("\n--- Fitting unconstrained model ---")
    model_unconstrained.fit(X_train, y_scaled)

    print("\n--- Fitting constrained model ---")
    model_constrained.fit(X_train, y_scaled)

    # Get coefficients
    coef_unconstrained = model_unconstrained.coef_
    coef_constrained = model_constrained.coef_

    print("\n--- Model Coefficients Comparison ---")
    print("Unconstrained coefficients:")
    for i, coef in enumerate(coef_unconstrained):
        print(f"  Feature {i}: {coef:.6f}")

    print("\nConstrained coefficients:")
    for i, coef in enumerate(coef_constrained):
        print(f"  Feature {i}: {coef:.6f}")

    print(
        f"\nIntercepts - Unconstrained: {model_unconstrained.intercept_:.6f}, Constrained: {model_constrained.intercept_:.6f}"
    )

    # Count coefficients that would be constrained
    n_outside_limits = np.sum(np.abs(coef_unconstrained) > tight_limit)
    print(
        f"Number of unconstrained coefficients outside limits: {n_outside_limits} out of {n_features}"
    )

    # Check if any constrained coefficients violate the limits
    constrained_violate_lower = np.any(coef_constrained < -tight_limit - 1e-6)
    constrained_violate_upper = np.any(coef_constrained > tight_limit + 1e-6)
    print(f"Constrained model violates lower limits: {constrained_violate_lower}")
    print(f"Constrained model violates upper limits: {constrained_violate_upper}")

    # Make predictions with both models
    pred_unconstrained = model_unconstrained.predict(X_test)
    pred_constrained = model_constrained.predict(X_test)

    print("\n--- Prediction Comparison ---")
    print("Sample predictions (first 5 samples):")
    for i in range(min(5, len(pred_unconstrained))):
        print(
            f"  Sample {i}: Unconstrained={pred_unconstrained[i]:.4f}, Constrained={pred_constrained[i]:.4f}"
        )

    # Calculate performance metrics
    mse_unconstrained = mean_squared_error(y_test * 10, pred_unconstrained)
    mse_constrained = mean_squared_error(y_test * 10, pred_constrained)
    r2_unconstrained = r2_score(y_test * 10, pred_unconstrained)
    r2_constrained = r2_score(y_test * 10, pred_constrained)

    print("\n--- Performance Metrics ---")
    print(f"Unconstrained MSE: {mse_unconstrained:.6f}")
    print(f"Constrained MSE: {mse_constrained:.6f}")
    print(f"Unconstrained R²: {r2_unconstrained:.6f}")
    print(f"Constrained R²: {r2_constrained:.6f}")

    # Print coefficient ranges
    print("\n=== Constraint Test Results ===")
    print(
        f"Unconstrained coefficient range: [{coef_unconstrained.min():.6f}, {coef_unconstrained.max():.6f}]"
    )
    print(
        f"Constrained coefficient range: [{coef_constrained.min():.6f}, {coef_constrained.max():.6f}]"
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

    print("All assertions passed!")


def test_penalty_factor_effectiveness(synthetic_data):
    """Test that penalty_factor has the expected effect on coefficients"""
    print_header("TESTING PENALTY FACTOR EFFECTIVENESS")
    X_train, X_test, y_train, y_test = synthetic_data

    # Parameters
    lambda_value = 0.5  # Higher lambda to make penalty effect more noticeable
    n_features = X_train.shape[1]

    print(f"Penalty factor test parameters:")
    print(f"  lambda_value: {lambda_value}")
    print(f"  n_features: {n_features}")

    # Create models with different penalty factors
    print("\n--- Creating uniform penalty model ---")
    model_uniform = create_ridge_model_native(
        lambda_value=lambda_value, fit_intercept=True, standardize=True
    )

    # Differential penalty: first half gets lower penalty, second half gets higher
    penalty_factor = np.ones(n_features)
    penalty_factor[: n_features // 2] = 0.1  # Much lower penalty
    penalty_factor[n_features // 2 :] = 2.0  # Higher penalty

    print(f"Differential penalty factors:")
    for i, p in enumerate(penalty_factor):
        print(f"  Feature {i}: {p}")

    print("\n--- Creating differential penalty model ---")
    model_differential = create_ridge_model_native(
        lambda_value=lambda_value,
        fit_intercept=True,
        standardize=True,
        penalty_factor=penalty_factor,
    )

    # Fit both models
    print("\n--- Fitting uniform penalty model ---")
    model_uniform.fit(X_train, y_train)

    print("\n--- Fitting differential penalty model ---")
    model_differential.fit(X_train, y_train)

    # Get absolute coefficient values
    uniform_abs_coef = np.abs(model_uniform.coef_)
    differential_abs_coef = np.abs(model_differential.coef_)

    print("\n--- Coefficient Comparison ---")
    print("Uniform penalty model coefficients:")
    for i, coef in enumerate(model_uniform.coef_):
        print(f"  Feature {i}: {coef:.6f} (abs: {abs(coef):.6f})")

    print("\nDifferential penalty model coefficients:")
    for i, coef in enumerate(model_differential.coef_):
        print(f"  Feature {i}: {coef:.6f} (abs: {abs(coef):.6f})")

    print(
        f"\nIntercepts - Uniform: {model_uniform.intercept_:.6f}, Differential: {model_differential.intercept_:.6f}"
    )

    # Calculate mean coefficient magnitudes by groups
    low_penalty_mean = np.mean(differential_abs_coef[: n_features // 2])
    high_penalty_mean = np.mean(differential_abs_coef[n_features // 2 :])
    uniform_first_half_mean = np.mean(uniform_abs_coef[: n_features // 2])
    uniform_second_half_mean = np.mean(uniform_abs_coef[n_features // 2 :])

    # Compare coefficient magnitudes between low and high penalty groups
    feature_comparison = pd.DataFrame(
        {
            "feature": range(n_features),
            "penalty_factor": penalty_factor,
            "uniform_coef": uniform_abs_coef,
            "differential_coef": differential_abs_coef,
            "ratio": differential_abs_coef / (uniform_abs_coef + 1e-10),
        }
    )

    print("\n--- Feature-by-Feature Penalty Effect ---")
    print(feature_comparison.to_string())

    # Make predictions
    pred_uniform = model_uniform.predict(X_test)
    pred_differential = model_differential.predict(X_test)

    print("\n--- Prediction Comparison ---")
    print("Sample predictions (first 5 samples):")
    for i in range(min(5, len(pred_uniform))):
        print(
            f"  Sample {i}: Uniform={pred_uniform[i]:.4f}, Differential={pred_differential[i]:.4f}"
        )

    # Calculate performance metrics
    mse_uniform = mean_squared_error(y_test, pred_uniform)
    mse_differential = mean_squared_error(y_test, pred_differential)
    r2_uniform = r2_score(y_test, pred_uniform)
    r2_differential = r2_score(y_test, pred_differential)

    print("\n--- Performance Metrics ---")
    print(f"Uniform penalty MSE: {mse_uniform:.6f}")
    print(f"Differential penalty MSE: {mse_differential:.6f}")
    print(f"Uniform penalty R²: {r2_uniform:.6f}")
    print(f"Differential penalty R²: {r2_differential:.6f}")

    # Print results
    print("\n=== Penalty Factor Test Results ===")
    print(f"Uniform model - first half mean: {uniform_first_half_mean:.6f}")
    print(f"Uniform model - second half mean: {uniform_second_half_mean:.6f}")
    print(f"Differential model - low penalty mean: {low_penalty_mean:.6f}")
    print(f"Differential model - high penalty mean: {high_penalty_mean:.6f}")

    # Verify low penalty features have larger coefficients than high penalty features
    assert (
        low_penalty_mean > high_penalty_mean
    ), "Features with lower penalty should have larger coefficients on average"

    # Calculate ratio of change compared to uniform model
    low_penalty_ratio = low_penalty_mean / uniform_first_half_mean
    high_penalty_ratio = high_penalty_mean / uniform_second_half_mean

    print(f"Low penalty ratio: {low_penalty_ratio:.6f}")
    print(f"High penalty ratio: {high_penalty_ratio:.6f}")

    # Verify penalty factors had significant effect compared to uniform model
    assert (
        low_penalty_ratio > 1.0
    ), "Features with lower penalty should have larger coefficients than in uniform model"
    assert (
        high_penalty_ratio < 1.0
    ), "Features with higher penalty should have smaller coefficients than in uniform model"

    print("All assertions passed!")


if __name__ == "__main__":
    pytest.main(["-v"])
