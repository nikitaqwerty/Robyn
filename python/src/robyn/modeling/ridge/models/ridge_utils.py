import logging
import numpy as np


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
            self.fixed_coefficients = fixed_coefficients  # Store fixed coefficients
            self.fixed_intercept = fixed_intercept  # Store fixed intercept

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

                    # Convert Python objects to R
                    with localconverter(ro.default_converter + numpy2ri.converter):
                        ro.r.assign("X_r", X_fit)
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
                            (
                                penalty_factor
                                if penalty_factor is not None
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
                        r_code = """
                        # Create a dummy model
                        r_model <<- glmnet(
                            x = X_r,
                            y = y_r,
                            family = "gaussian",
                            alpha = 0,
                            lambda = lambda_value,
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
