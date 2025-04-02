import logging
import numpy as np
from sklearn.linear_model import Ridge
import time


def create_ridge_model_sklearn(
    lambda_value, n_samples, fit_intercept=True, standardize=True
):
    """Create a Ridge regression model using scikit-learn.

    Args:
        lambda_value: Regularization parameter (lambda) from glmnet
        n_samples: Number of samples (needed for proper scaling)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features

    Returns:
        A configured sklearn Ridge model that behaves like glmnet
    """

    # Create a wrapper class that matches glmnet's behavior
    class GlmnetLikeRidge:
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.lambda_value = lambda_value  # Use raw lambda value
            self.fit_intercept = fit_intercept
            self.standardize = standardize
            self.feature_means = None
            self.feature_stds = None
            self.y_mean = None
            self.coef_ = None
            self.intercept_ = 0.0

        def mysd(self, y):
            """R-like standard deviation"""
            return np.sqrt(np.sum((y - np.mean(y)) ** 2) / len(y))

        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)

            # Debug prints matching R
            self.logger.debug("Lambda calculation debug:")
            self.logger.debug(f"x_means: {np.mean(np.abs(X))}")
            x_sds = np.apply_along_axis(self.mysd, 0, X)
            self.logger.debug(f"x_sds mean: {np.mean(x_sds)}")

            # Center and scale like R's glmnet
            if self.standardize:
                self.feature_means = np.mean(X, axis=0)
                self.feature_stds = np.apply_along_axis(self.mysd, 0, X)
                self.feature_stds[self.feature_stds == 0] = 1.0
                X_scaled = (X - self.feature_means) / self.feature_stds
            else:
                X_scaled = X
                self.feature_means = np.zeros(X.shape[1])
                self.feature_stds = np.ones(X.shape[1])

            if self.fit_intercept:
                self.y_mean = np.mean(y)
                y_centered = y - self.y_mean
            else:
                y_centered = y
                self.y_mean = 0.0

            self.logger.debug(f"sx mean: {np.mean(np.abs(X_scaled))}")
            self.logger.debug(f"sy mean: {np.mean(np.abs(y_centered))}")
            self.logger.debug(f"lambda: {self.lambda_value}")

            # Determine n_samples for scaling
            n_samples = X.shape[0]
            if n_samples == 0:
                raise ValueError("Cannot fit model with 0 samples")

            # Scale lambda for sklearn's Ridge (alpha = lambda / n_samples when standardized)
            sklearn_alpha = self.lambda_value / n_samples
            self.logger.debug(f"n_samples: {n_samples}, sklearn_alpha: {sklearn_alpha}")

            # Fit model using raw lambda (not scaled)
            model = Ridge(
                alpha=sklearn_alpha,  # Use scaled alpha
                fit_intercept=False,  # We handle centering manually
                solver="cholesky",
            )

            model.fit(X_scaled, y_centered)

            # Transform coefficients back to original scale
            if self.standardize:
                self.coef_ = model.coef_ / self.feature_stds
            else:
                self.coef_ = model.coef_

            if self.fit_intercept:
                self.intercept_ = self.y_mean - np.dot(self.feature_means, self.coef_)

            self.logger.debug(
                f"Coefficients range: [{np.min(self.coef_):.6f}, {np.max(self.coef_):.6f}]"
            )
            self.logger.debug(f"Intercept: {self.intercept_:.6f}")

            return self

        def predict(self, X):
            if self.coef_ is None:
                raise ValueError("Model must be fitted before making predictions")

            # Direct prediction using coefficients and intercept
            return np.dot(X, self.coef_) + self.intercept_

    return GlmnetLikeRidge()


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import logging
import hashlib
from scipy.optimize import minimize


def create_ridge_model_native(
    lambda_value,
    n_samples=None,  # Not used but kept for API compatibility
    fit_intercept=True,
    standardize=True,
    lower_limits=None,
    upper_limits=None,
    intercept=True,
    intercept_sign="non_negative",
    penalty_factor=None,
):
    """Create a Ridge regression model using native Python implementations.

    This is a replacement for the rpy2-based glmnet implementation.

    Args:
        lambda_value: Regularization parameter (alpha in scikit-learn)
        n_samples: Number of samples (not directly used, but kept for API consistency)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features
        lower_limits: Lower limits for coefficients
        upper_limits: Upper limits for coefficients
        intercept: Whether to include intercept
        intercept_sign: Constraint on intercept sign ("non_negative" or None)
        penalty_factor: Factors to multiply penalties (array of same length as features)

    Returns:
        A Ridge regression model using native Python.
    """

    class NativeRidgeWrapper:
        def __init__(self):
            self.lambda_value = lambda_value
            self.fit_intercept = fit_intercept and intercept
            self.standardize = standardize
            self.intercept_sign = intercept_sign
            self.coef_ = None
            self.intercept_ = 0.0
            self.logger = logging.getLogger(__name__)
            self._prediction_cache = {}
            self._X_matrix_cache = {}
            self.full_coef_ = None
            self.df_int = 1 if self.fit_intercept else 0
            self.scaler = StandardScaler() if standardize else None

            # Process constraints
            self.penalty_factor = penalty_factor
            self.lower_limits = lower_limits
            self.upper_limits = upper_limits

            # Determine if we need the custom solver
            self.use_custom_solver = (
                penalty_factor is not None
                or lower_limits is not None
                or upper_limits is not None
            )

            # Initialize the standard model if not using custom solver
            if not self.use_custom_solver:
                self.model = Ridge(
                    alpha=lambda_value,
                    fit_intercept=self.fit_intercept,
                    copy_X=True,
                    max_iter=None,
                    tol=0.001,
                    solver="auto",
                    random_state=None,
                )

        def _custom_objective(self, beta, X, y, alpha, penalty_factor=None):
            """Custom objective function for ridge regression - closely matches glmnet implementation"""
            n_samples = X.shape[0]

            # Extract intercept if fitting intercept
            if self.fit_intercept:
                intercept = beta[0]
                coefficients = beta[1:]
                y_pred = X @ coefficients + intercept
            else:
                coefficients = beta
                y_pred = X @ coefficients

            # Mean squared error
            residuals = y - y_pred
            mse = np.sum(residuals**2) / n_samples

            # Penalty term using penalty factors if provided
            if penalty_factor is not None:
                if len(penalty_factor) != len(coefficients):
                    # Ensure penalty factor has correct length
                    pf = np.ones(len(coefficients))
                    pf[: min(len(penalty_factor), len(coefficients))] = penalty_factor[
                        : min(len(penalty_factor), len(coefficients))
                    ]
                    penalty = alpha * np.sum(pf * coefficients**2)
                else:
                    penalty = alpha * np.sum(penalty_factor * coefficients**2)
            else:
                penalty = alpha * np.sum(coefficients**2)

            # Objective: MSE + penalty (similar to glmnet)
            return mse + penalty / n_samples

        def _fit_custom_solver(self, X, y):
            """Custom fitting function that replicates glmnet's behavior with constraints"""
            n_features = X.shape[1]
            n_samples = X.shape[0]

            # Initialize coefficients
            if self.fit_intercept:
                beta_init = np.zeros(n_features + 1)
                # Estimate initial intercept as mean of y
                beta_init[0] = np.mean(y)
            else:
                beta_init = np.zeros(n_features)

            # Prepare bounds for coefficients
            bounds = []

            # Handle intercept bounds if fitting intercept
            if self.fit_intercept:
                # Apply non-negative constraint to intercept if requested
                if self.intercept_sign == "non_negative":
                    bounds.append((0, None))
                else:
                    bounds.append((None, None))

                start_idx = 1  # Start index for feature coefficients
            else:
                start_idx = 0

            # Process bounds for features
            for i in range(n_features):
                lower = None
                upper = None

                # Apply lower limit if specified
                if self.lower_limits is not None:
                    if i < len(self.lower_limits):
                        lower = self.lower_limits[i]

                # Apply upper limit if specified
                if self.upper_limits is not None:
                    if i < len(self.upper_limits):
                        upper = self.upper_limits[i]

                bounds.append((lower, upper))

            # Prepare penalty factors array
            pf = None if self.penalty_factor is None else np.array(self.penalty_factor)

            # Use L-BFGS-B optimizer which handles bounds constraints well
            result = minimize(
                self._custom_objective,
                beta_init,
                args=(X, y, self.lambda_value, pf),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "gtol": 1e-6},
            )

            # Check convergence
            if not result.success:
                self.logger.warning(f"Optimization did not converge: {result.message}")

            # Extract coefficients
            if self.fit_intercept:
                self.intercept_ = result.x[0]
                self.coef_ = result.x[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = result.x

            # Create full coefficients array (R-style)
            self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])

        def fit(self, X, y):
            """Fit the ridge regression model to the data"""
            X = np.asarray(X)
            y = np.asarray(y)

            # Handle case with a single feature
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Apply standardization if requested (similar to glmnet's standardize=TRUE)
            if self.standardize:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            # Choose fitting method based on constraints
            if self.use_custom_solver:
                self._fit_custom_solver(X_scaled, y)
            else:
                # Attempt to fit with standard Ridge first
                self.model.fit(X_scaled, y)

                # Check intercept sign constraint
                if (
                    self.fit_intercept
                    and self.intercept_sign == "non_negative"
                    and self.model.intercept_ < 0
                ):
                    # If intercept is negative but should be non-negative, refit without intercept
                    self.logger.debug(
                        "Intercept was negative, refitting without intercept"
                    )
                    self.model = Ridge(alpha=self.lambda_value, fit_intercept=False)
                    self.model.fit(X_scaled, y)
                    self.fit_intercept = False
                    self.df_int = 0
                else:
                    self.df_int = 1 if self.fit_intercept else 0

                # Store coefficients
                self.coef_ = self.model.coef_
                self.intercept_ = self.model.intercept_ if self.fit_intercept else 0.0

                # Create full coefficients array (R-style)
                self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])

            return self

        def predict(self, X):
            """Make predictions with the fitted model"""
            X = np.asarray(X)

            # Handle case with a single feature
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Generate a hash for cache lookup
            X_hash = hashlib.md5(X.tobytes()).hexdigest()

            if X_hash in self._prediction_cache:
                self.logger.debug("Using cached predictions")
                return self._prediction_cache[X_hash]

            # Apply standardization if used during fitting - crucial for matching glmnet
            if self.standardize:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Make predictions
            if self.use_custom_solver or not hasattr(self, "model"):
                # Use direct calculation for custom solver
                predictions = X_scaled @ self.coef_ + self.intercept_
            else:
                # Use model for sklearn implementation
                predictions = self.model.predict(X_scaled)

            # Log prediction stats
            self.logger.debug(
                f"Predictions stats - min: {predictions.min():.6f}, "
                f"max: {predictions.max():.6f}, mean: {predictions.mean():.6f}"
            )

            # Cache the predictions
            self._prediction_cache[X_hash] = predictions

            return predictions

        def get_full_coefficients(self):
            """Get full coefficient array including intercept (R-style)"""
            return self.full_coef_

    return NativeRidgeWrapper()


def create_ridge_model_rpy2(
    lambda_value,
    n_samples,
    fit_intercept=True,
    standardize=True,
    lower_limits=None,
    upper_limits=None,
    intercept=True,
    intercept_sign="non_negative",
    penalty_factor=None,
):
    """Create a Ridge regression model using rpy2 to access glmnet.

    Args:
        lambda_value: Regularization parameter
        n_samples: Number of samples (not directly used, but kept for API consistency)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features
        **kwargs: Additional arguments to pass to glmnet

    Returns:
        A Ridge regression model using rpy2 to access glmnet.

    Raises:
        ImportError: If rpy2 is not available
        RuntimeError: If glmnet R package cannot be imported
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        raise ImportError(
            "rpy2 is required for using the R implementation. Please install rpy2."
        )

    # Import glmnet only once per Python session
    global glmnet_imported
    if "glmnet_imported" not in globals():
        try:
            importr("glmnet")
            glmnet_imported = True
        except Exception as e:
            raise RuntimeError(f"Failed to import glmnet R package: {e}")

    class GlmnetRidgeWrapper:
        def __init__(self):
            self.lambda_value = lambda_value
            self.fit_intercept = fit_intercept
            self.standardize = standardize
            self.intercept_sign = intercept_sign
            self.fitted_model = None
            self.coef_ = None
            self.intercept_ = 0.0
            self.logger = logging.getLogger(__name__)
            self._prediction_cache = {}
            # Cache for performance
            self._X_matrix_cache = {}
            self.full_coef_ = None  # Add this to store full coefficient array
            self.df_int = 1  # Initialize to 1

        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)

            # Convert Python objects to R
            with localconverter(ro.default_converter + numpy2ri.converter):
                ro.r.assign("X_r", X)
                ro.r.assign("y_r", y)
                ro.r.assign("lambda_value", self.lambda_value)
                ro.r.assign(
                    "lower_limits_r",
                    lower_limits if lower_limits is not None else ro.r("NULL"),
                )
                ro.r.assign(
                    "upper_limits_r",
                    upper_limits if upper_limits is not None else ro.r("NULL"),
                )
                ro.r.assign(
                    "penalty_factor_r",
                    penalty_factor if penalty_factor is not None else ro.r("NULL"),
                )

                # First attempt: Fit with intercept
                r_code = """
                # First fit with intercept
                r_model <<- glmnet(
                    x = X_r,
                    y = y_r,
                    family = "gaussian",
                    alpha = 0,
                    lambda = lambda_value,
                    lower.limits = lower_limits_r,
                    upper.limits = upper_limits_r,
                    type.measure = "mse",
                    penalty.factor = penalty_factor_r,
                    intercept = TRUE
                )
                coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                """
                ro.r(r_code)

                # Check intercept sign constraint
                coef_array = np.array(ro.r["coef_values"])
                if self.intercept_sign == "non_negative" and coef_array[0] < 0:
                    # Second attempt: Refit without intercept
                    r_code = """
                    # Refit without intercept
                    r_model <<- glmnet(
                        x = X_r,
                        y = y_r,
                        family = "gaussian",
                        alpha = 0,
                        lambda = lambda_value,
                        lower.limits = lower_limits_r,
                        upper.limits = upper_limits_r,
                        type.measure = "mse",
                        penalty.factor = penalty_factor_r,
                        intercept = FALSE
                    )
                    coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                    """
                    ro.r(r_code)
                    coef_array = np.array(ro.r["coef_values"])
                    self.fit_intercept = False
                    self.df_int = 0  # Set df_int to 0 when intercept is dropped
                else:
                    self.df_int = 1  # Keep df_int as 1 when intercept is kept

                # Store model and coefficients
                self.fitted_model = ro.r["r_model"]
                if self.fit_intercept:
                    self.intercept_ = float(coef_array[0])
                    self.coef_ = coef_array[1:]
                    self.full_coef_ = coef_array  # Store full array including intercept
                else:
                    self.intercept_ = 0.0
                    self.coef_ = coef_array[1:]
                    # Create full coefficient array with 0 intercept
                    self.full_coef_ = np.concatenate([[0.0], self.coef_])

            return self

        def predict(self, X):
            X = np.asarray(X)

            if X.shape[0] < 1000:
                predictions = np.dot(X, self.coef_) + self.intercept_
                self.logger.debug(f"Using direct computation")
            else:
                # For larger matrices, use R but check cache first
                X_hash = hash(X.tobytes())
                if X_hash in self._prediction_cache:
                    return self._prediction_cache[X_hash]

                # Make predictions using R code directly
                with localconverter(ro.default_converter + numpy2ri.converter):
                    # Pass the data to R environment
                    ro.r.assign("X_new", X)
                    ro.r.assign("lambda_value", self.lambda_value)

                    # Make predictions using R code
                    ro.r(
                        """
                    predictions <<- as.numeric(predict(r_model, newx = X_new, s = lambda_value, type = "response"))
                    """
                    )

                    # Get predictions from R
                    predictions = np.array(ro.r["predictions"])
                    self.logger.debug("\n=== Prediction Output ===")
                    self.logger.debug(
                        f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]"
                    )
                    self.logger.debug(f"Predictions mean: {predictions.mean():.6f}")
                    # Cache the predictions
                    self._prediction_cache[X_hash] = predictions

                    self.logger.debug(f"Using R computation")

            self.logger.debug(
                f"Predictions stats - min: {predictions.min():.6f}, max: {predictions.max():.6f}, mean: {predictions.mean():.6f}"
            )
            return predictions

        def get_full_coefficients(self):
            """Get full coefficient array including intercept (R-style)"""
            return self.full_coef_

    return GlmnetRidgeWrapper()
