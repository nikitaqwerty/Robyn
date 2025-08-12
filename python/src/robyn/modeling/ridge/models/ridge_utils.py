import logging
from typing import Any, List, Optional

import numpy as np

try:
    # python-glmnet package
    from glmnet.linear import ElasticNet as GlmnetElasticNet
except Exception:  # pragma: no cover - optional dependency resolved at runtime
    GlmnetElasticNet = None


def create_ridge_model_python(
    lambda_value,
    n_samples,
    fit_intercept=True,
    standardize=True,
    lower_limits=None,
    upper_limits=None,
    intercept=True,
    intercept_sign="non_negative",
    penalty_factor=None,
    fixed_coefficients=None,
    fixed_intercept=None,
    weights=None,
    intercept_lower_limit=None,
    intercept_upper_limit=None,
):
    """Create a Python-native Ridge regression model with constraints using python-glmnet.

    Mirrors the interface and behavior of create_ridge_model_rpy2().
    """
    if GlmnetElasticNet is None:
        raise ImportError(
            "python-glmnet is required for the Python implementation. Please install python-glmnet."
        )

    class GlmnetRidgePythonWrapper:
        def __init__(self):
            self.lambda_value: float = float(lambda_value)
            self.fit_intercept: bool = bool(fit_intercept and intercept)
            self.standardize: bool = bool(standardize)
            self.intercept_sign: str = (
                str(intercept_sign) if intercept_sign is not None else "default"
            )
            self.fitted_model: Optional[Any] = None
            self.coef_: Optional[np.ndarray] = None
            self.intercept_: float = 0.0
            self.logger = logging.getLogger(__name__)
            self.full_coef_: Optional[np.ndarray] = None
            self.df_int: int = 1  # default assumes intercept included
            self.fixed_coefficients: Optional[List[Optional[float]]] = (
                list(fixed_coefficients) if fixed_coefficients is not None else None
            )
            self.fixed_intercept: Optional[float] = (
                float(fixed_intercept) if fixed_intercept is not None else None
            )
            # Optional intercept bounds
            self.intercept_lower_limit: Optional[float] = (
                float(intercept_lower_limit)
                if intercept_lower_limit is not None
                else None
            )
            self.intercept_upper_limit: Optional[float] = (
                float(intercept_upper_limit)
                if intercept_upper_limit is not None
                else None
            )

            # Persist original constraints/penalties; will be subset as needed
            self._lower_limits_orig = lower_limits
            self._upper_limits_orig = upper_limits
            self._penalty_factor_orig = penalty_factor

        def _build_estimator(self, lower_limits_arr, upper_limits_arr, lambda_path_arr):
            # Disable CV; we'll extract coefficients for our provided lambda directly
            est = GlmnetElasticNet(
                alpha=0.0,
                n_lambda=len(lambda_path_arr),
                lambda_path=lambda_path_arr,
                min_lambda_ratio=1.0 if len(lambda_path_arr) == 1 else 1e-4,
                standardize=self.standardize,
                fit_intercept=self.fit_intercept,
                lower_limits=(
                    lower_limits_arr if lower_limits_arr is not None else -np.inf
                ),
                upper_limits=(
                    upper_limits_arr if upper_limits_arr is not None else np.inf
                ),
                n_splits=0,
            )
            return est

        def _subset_constraints(self, indices):
            def sub_or_none(arr):
                if arr is None:
                    return None
                arr = np.asarray(arr)
                return arr[indices]

            lower = sub_or_none(self._lower_limits_orig)
            upper = sub_or_none(self._upper_limits_orig)
            penalty = sub_or_none(self._penalty_factor_orig)
            return lower, upper, penalty

        def _extract_solution(self, est, lambda_value_use):
            # Use internal interpolation helper to get coef/intercept at lambda
            # We avoid calling est.predict with lamb=None (would require lambda_best_)
            lambda_path = est.lambda_path_
            coef_path = est.coef_path_
            intercept_path = est.intercept_path_

            # When single lambda, coef_path is already the target. Otherwise interpolate
            if (
                lambda_path.shape[0] == 1
                or np.isclose(lambda_value_use, lambda_path).any()
            ):
                # Find nearest index
                inx = int(np.argmin(np.abs(lambda_path - lambda_value_use)))
                coef = coef_path[..., inx]
                intercept = intercept_path[..., inx]
            else:
                # Local import to avoid top-level coupling
                from glmnet.util import _interpolate_model

                coef, intercept = _interpolate_model(
                    lambda_path, coef_path, intercept_path, np.array([lambda_value_use])
                )
                # squeeze the last dimension (lambda)
                coef = np.squeeze(coef, axis=-1)
                intercept = float(np.squeeze(intercept, axis=-1))
            # shape fixes
            if coef.ndim > 1:
                coef = np.squeeze(coef)
            intercept = (
                float(intercept) if not np.isscalar(intercept) else float(intercept)
            )
            return coef, intercept

        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)

            # Handle fixed intercept: adjust target and disable intercept in estimator
            if self.fixed_intercept is not None:
                y_adjusted = y - self.fixed_intercept
                self.fit_intercept = False
                self.intercept_ = float(self.fixed_intercept)
                self.df_int = 0
            else:
                y_adjusted = y

            # Partition columns for fixed coefficients case
            fixed_cols = []
            fit_cols = list(range(X.shape[1]))
            fixed_values = None
            if self.fixed_coefficients is not None:
                if len(self.fixed_coefficients) != X.shape[1]:
                    raise ValueError(
                        f"Length of fixed_coefficients ({len(self.fixed_coefficients)}) must match number of features ({X.shape[1]})"
                    )
                fixed_cols = [
                    i for i, c in enumerate(self.fixed_coefficients) if c is not None
                ]
                fit_cols = [
                    i for i, c in enumerate(self.fixed_coefficients) if c is None
                ]
                if len(fixed_cols) > 0:
                    fixed_values = np.array(
                        [self.fixed_coefficients[i] for i in fixed_cols], dtype=float
                    )
                    y_adjusted = y_adjusted - X[:, fixed_cols] @ fixed_values

            # Prepare constraints and penalties for the columns to be fitted
            if len(fit_cols) < X.shape[1]:
                lower_limits_fit, upper_limits_fit, penalty_factor_fit = (
                    self._subset_constraints(np.array(fit_cols, dtype=int))
                )
                X_fit = X[:, fit_cols]
            else:
                lower_limits_fit = (
                    np.asarray(self._lower_limits_orig)
                    if self._lower_limits_orig is not None
                    else None
                )
                upper_limits_fit = (
                    np.asarray(self._upper_limits_orig)
                    if self._upper_limits_orig is not None
                    else None
                )
                penalty_factor_fit = (
                    np.asarray(self._penalty_factor_orig)
                    if self._penalty_factor_orig is not None
                    else None
                )
                X_fit = X

            # Edge-case: all coefficients fixed, no training needed
            if len(fit_cols) == 0:
                self.coef_ = np.array(self.fixed_coefficients, dtype=float)
                if self.fixed_intercept is None:
                    self.intercept_ = 0.0
                self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])
                self.fitted_model = None
                # Intercept DOF when not fitting intercept depends on setting
                self.df_int = 1 if self.fit_intercept else 0
                return self

            # Build estimator and fit
            est = self._build_estimator(
                lower_limits_fit,
                upper_limits_fit,
                np.array([self.lambda_value], dtype=float),
            )
            # Fit with sample weights and relative penalties mapping the R interface
            sample_weight = None
            if hasattr(self, "weights") and self.weights is not None:
                sample_weight = np.asarray(self.weights)
            # For this wrapper, we pass penalties via relative_penalties
            relative_penalties = (
                np.asarray(penalty_factor_fit, dtype=float)
                if penalty_factor_fit is not None
                else None
            )

            est = est.fit(
                X_fit,
                y_adjusted,
                sample_weight=sample_weight,
                relative_penalties=relative_penalties,
            )

            # Extract coefficients at specified lambda
            coef_fitted, intercept_fitted = self._extract_solution(
                est, self.lambda_value
            )

            # Intercept sign handling: if requested non_negative and negative found, refit without intercept
            if (
                self.fixed_intercept is None
                and self.fit_intercept
                and self.intercept_sign == "non_negative"
                and intercept_fitted < 0
            ):
                self.fit_intercept = False
                self.df_int = 0
                est = self._build_estimator(
                    lower_limits_fit,
                    upper_limits_fit,
                    np.array([self.lambda_value], dtype=float),
                )
                est = est.fit(
                    X_fit,
                    y_adjusted,
                    sample_weight=sample_weight,
                    relative_penalties=relative_penalties,
                )
                coef_fitted, intercept_fitted = self._extract_solution(
                    est, self.lambda_value
                )

            # Save model and coefficients
            self.fitted_model = est
            if self.fixed_intercept is None:
                self.intercept_ = float(intercept_fitted if self.fit_intercept else 0.0)
                self.df_int = 1 if self.fit_intercept else 0

            # Enforce optional intercept bounds by refitting with fixed intercept at boundary if violated
            if self.fixed_intercept is None and (
                self.intercept_lower_limit is not None
                or self.intercept_upper_limit is not None
            ):
                current_intercept = float(self.intercept_)
                violate_lower = (
                    self.intercept_lower_limit is not None
                    and current_intercept < self.intercept_lower_limit
                )
                violate_upper = (
                    self.intercept_upper_limit is not None
                    and current_intercept > self.intercept_upper_limit
                )
                if violate_lower or violate_upper:
                    # Pick the nearest boundary value
                    bound_value = current_intercept
                    if violate_lower:
                        bound_value = self.intercept_lower_limit  # type: ignore[assignment]
                    if violate_upper:
                        bound_value = self.intercept_upper_limit  # type: ignore[assignment]

                    # Refit with intercept fixed to bound: disable intercept in estimator
                    self.fit_intercept = False
                    self.df_int = 0
                    est = self._build_estimator(
                        lower_limits_fit,
                        upper_limits_fit,
                        np.array([self.lambda_value], dtype=float),
                    )
                    # Adjust target to account for fixed intercept at the boundary
                    y_adjusted_refit = y_adjusted - bound_value
                    est = est.fit(
                        X_fit,
                        y_adjusted_refit,
                        sample_weight=sample_weight,
                        relative_penalties=relative_penalties,
                    )
                    coef_fitted, _ = self._extract_solution(est, self.lambda_value)
                    # Update intercept and fitted model
                    self.fitted_model = est
                    self.intercept_ = float(bound_value)
                    self.df_int = 0

            # Combine with fixed coefficients to full coef vector
            combined_coef = np.zeros(X.shape[1], dtype=float)
            if len(fit_cols) == X.shape[1]:
                combined_coef[:] = coef_fitted
            else:
                # place fitted values back
                for i, col_idx in enumerate(fit_cols):
                    combined_coef[col_idx] = coef_fitted[i]
                # add fixed
                for j, col_idx in enumerate(fixed_cols):
                    combined_coef[col_idx] = fixed_values[j]
            self.coef_ = combined_coef
            self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])

            return self

        def predict(self, X):
            X = np.asarray(X)
            return X @ self.coef_ + self.intercept_

        def get_full_coefficients(self):
            return self.full_coef_

    # Attach weights attribute to instance so fit can access it (keep same interface as rpy2 wrapper)
    wrapper = GlmnetRidgePythonWrapper()
    wrapper.weights = weights
    return wrapper


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
    fixed_coefficients=None,  # Parameter for fixed coefficients
    fixed_intercept=None,  # New parameter for fixed intercept
    weights=None,  # New parameter for observation weights
):
    """Create a Ridge regression model using rpy2 to access glmnet.

    Args:
        lambda_value: Regularization parameter
        n_samples: Number of samples (not directly used, but kept for API consistency)
        fit_intercept: Whether to fit the intercept
        standardize: Whether to standardize the input features
        lower_limits: Lower limits for coefficients
        upper_limits: Upper limits for coefficients
        intercept: Whether to include intercept
        intercept_sign: Sign constraint for intercept ("non_negative" or None)
        penalty_factor: Penalty factors for each coefficient
        fixed_coefficients: List of fixed coefficient values (None for coefficients to be fitted)
        fixed_intercept: Fixed value for intercept (None to fit the intercept)
        weights: Observation weights. Can be total counts if responses are proportion matrices.
                Default is 1 for each observation

    Returns:
        A Ridge regression model using rpy2 to access glmnet.

    Raises:
        ImportError: If rpy2 is not available
        RuntimeError: If glmnet R package cannot be imported
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
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
            self.fixed_coefficients = fixed_coefficients  # Store fixed coefficients
            self.fixed_intercept = fixed_intercept  # Store fixed intercept
            self.weights = weights  # Store observation weights

        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y)

            # If fixed intercept is provided, adjust target and fit without intercept
            if self.fixed_intercept is not None:
                y_adjusted = y.copy() - self.fixed_intercept
                self.fit_intercept = False  # Force fit without intercept
                self.intercept_ = self.fixed_intercept  # Use the fixed value
                self.df_int = 0  # No degrees of freedom for intercept
            else:
                y_adjusted = y.copy()

            # Handle fixed coefficients
            fixed_cols = []  # Indices of columns with fixed coefficients
            fit_cols = []  # Indices of columns to be fitted

            # If fixed coefficients are provided, separate the data for fitting
            if self.fixed_coefficients is not None:
                if len(self.fixed_coefficients) != X.shape[1]:
                    raise ValueError(
                        f"Length of fixed_coefficients ({len(self.fixed_coefficients)}) must match number of features ({X.shape[1]})"
                    )

                fixed_cols = [
                    i
                    for i, coef in enumerate(self.fixed_coefficients)
                    if coef is not None
                ]
                fit_cols = [
                    i for i, coef in enumerate(self.fixed_coefficients) if coef is None
                ]

                # Get fixed values for later
                fixed_values = np.array(
                    [coef for coef in self.fixed_coefficients if coef is not None]
                )

                if fixed_cols:  # If we have fixed coefficients
                    # Adjust y by subtracting the contribution of fixed coefficients
                    X_fixed = X[:, fixed_cols]
                    y_adjusted = y_adjusted - np.dot(X_fixed, fixed_values)

                    # Prepare the reduced X with only columns to be fitted
                    X_fit = X[:, fit_cols]

                    # Subset the constraints to match only the columns being fitted
                    lower_limits_fit = None
                    upper_limits_fit = None
                    penalty_factor_fit = None

                    if lower_limits is not None:
                        lower_limits_fit = [lower_limits[i] for i in fit_cols]

                    if upper_limits is not None:
                        upper_limits_fit = [upper_limits[i] for i in fit_cols]

                    if penalty_factor is not None:
                        penalty_factor_fit = [penalty_factor[i] for i in fit_cols]

                    # Convert Python objects to R
                    with localconverter(ro.default_converter + numpy2ri.converter):
                        ro.r.assign("X_r", X_fit)
                        ro.r.assign("y_r", y_adjusted)
                        ro.r.assign("lambda_value", self.lambda_value)
                        ro.r.assign(
                            "lower_limits_r",
                            (
                                lower_limits_fit
                                if lower_limits_fit is not None
                                else ro.r("NULL")
                            ),
                        )
                        ro.r.assign(
                            "upper_limits_r",
                            (
                                upper_limits_fit
                                if upper_limits_fit is not None
                                else ro.r("NULL")
                            ),
                        )
                        ro.r.assign(
                            "penalty_factor_r",
                            (
                                penalty_factor_fit
                                if penalty_factor_fit is not None
                                else ro.r("NULL")
                            ),
                        )
                        ro.r.assign(
                            "weights_r",
                            (
                                self.weights
                                if self.weights is not None
                                else ro.r("NULL")
                            ),
                        )

                        # Set intercept parameter based on fixed_intercept
                        use_intercept = (
                            self.fixed_intercept is None and self.fit_intercept
                        )

                        # First attempt: Fit with or without intercept based on settings
                        r_code = """
                        # Fit model
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
                            weights = weights_r,
                            intercept = %s
                        )
                        coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                        """ % (
                            "TRUE" if use_intercept else "FALSE"
                        )
                        ro.r(r_code)

                        # Check intercept sign constraint if we're fitting the intercept
                        coef_array = np.array(ro.r["coef_values"])
                        if (
                            use_intercept
                            and self.intercept_sign == "non_negative"
                            and coef_array[0] < 0
                        ):
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
                                weights = weights_r,
                                intercept = FALSE
                            )
                            coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                            """
                            ro.r(r_code)
                            coef_array = np.array(ro.r["coef_values"])
                            self.fit_intercept = False
                            self.df_int = 0  # Set df_int to 0 when intercept is dropped
                        elif use_intercept:
                            self.df_int = 1  # Keep df_int as 1 when intercept is kept

                        # Store model and coefficients
                        self.fitted_model = ro.r["r_model"]
                        if use_intercept:
                            if self.fixed_intercept is None:
                                self.intercept_ = float(coef_array[0])
                            fitted_coef = coef_array[1:]
                        else:
                            if self.fixed_intercept is None:
                                self.intercept_ = 0.0
                            fitted_coef = coef_array[1:]

                        # Combine fitted and fixed coefficients
                        combined_coef = np.zeros(X.shape[1])
                        for i, col_idx in enumerate(fit_cols):
                            combined_coef[col_idx] = fitted_coef[i]
                        for i, col_idx in enumerate(fixed_cols):
                            combined_coef[col_idx] = fixed_values[i]

                        self.coef_ = combined_coef
                        # Create full coefficient array including intercept
                        self.full_coef_ = np.concatenate(
                            [[self.intercept_], self.coef_]
                        )
                else:
                    # If all coefficients are fixed, no need to fit
                    self.coef_ = np.array(self.fixed_coefficients)
                    if self.fixed_intercept is None:
                        self.intercept_ = 0.0  # No intercept in this case
                    self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])

                    # Create a dummy model for compatibility
                    with localconverter(ro.default_converter + numpy2ri.converter):
                        ro.r.assign("X_r", X)
                        ro.r.assign("y_r", y_adjusted)
                        ro.r.assign("lambda_value", self.lambda_value)
                        ro.r.assign(
                            "weights_r",
                            (
                                self.weights
                                if self.weights is not None
                                else ro.r("NULL")
                            ),
                        )
                        r_code = """
                        # Create a dummy model
                        r_model <<- glmnet(
                            x = X_r,
                            y = y_r,
                            family = "gaussian",
                            alpha = 0,
                            lambda = lambda_value,
                            weights = weights_r,
                            intercept = FALSE
                        )
                        """
                        ro.r(r_code)
                        self.fitted_model = ro.r["r_model"]
                        self.fit_intercept = False
                        self.df_int = 0
            else:
                # Original fitting logic if no fixed coefficients
                with localconverter(ro.default_converter + numpy2ri.converter):
                    ro.r.assign("X_r", X)
                    ro.r.assign("y_r", y_adjusted)
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
                    ro.r.assign(
                        "weights_r",
                        self.weights if self.weights is not None else ro.r("NULL"),
                    )

                    # Set intercept parameter based on fixed_intercept
                    use_intercept = self.fixed_intercept is None and self.fit_intercept

                    # First attempt: Fit with or without intercept based on settings
                    r_code = """
                    # Fit model
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
                        weights = weights_r,
                        intercept = %s
                    )
                    coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                    """ % (
                        "TRUE" if use_intercept else "FALSE"
                    )
                    ro.r(r_code)

                    # Check intercept sign constraint if we're fitting the intercept
                    coef_array = np.array(ro.r["coef_values"])
                    if (
                        use_intercept
                        and self.intercept_sign == "non_negative"
                        and coef_array[0] < 0
                    ):
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
                            weights = weights_r,
                            intercept = FALSE
                        )
                        coef_values <<- as.numeric(coef(r_model, s = lambda_value))
                        """
                        ro.r(r_code)
                        coef_array = np.array(ro.r["coef_values"])
                        self.fit_intercept = False
                        self.df_int = 0  # Set df_int to 0 when intercept is dropped
                    elif use_intercept:
                        self.df_int = 1  # Keep df_int as 1 when intercept is kept

                    # Store model and coefficients
                    self.fitted_model = ro.r["r_model"]
                    if use_intercept:
                        if self.fixed_intercept is None:
                            self.intercept_ = float(coef_array[0])
                        self.coef_ = coef_array[1:]
                        self.full_coef_ = (
                            coef_array  # Store full array including intercept
                        )
                    else:
                        if self.fixed_intercept is None:
                            self.intercept_ = 0.0
                        self.coef_ = coef_array[1:]
                        # Create full coefficient array with intercept
                        self.full_coef_ = np.concatenate(
                            [[self.intercept_], self.coef_]
                        )

            return self

        def predict(self, X):
            X = np.asarray(X)

            if (
                X.shape[0] < 1000
                or self.fixed_coefficients is not None
                or self.fixed_intercept is not None
            ):
                # Always use direct computation when fixed coefficients/intercept are provided
                predictions = np.dot(X, self.coef_) + self.intercept_
                self.logger.debug(f"Using direct computation")
            else:
                # For larger matrices without fixed values, use R but check cache first
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
