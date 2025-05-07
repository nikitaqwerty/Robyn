import logging
import numpy as np

from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler  # Assuming this is allowed

# n_samples is part of the original API, kept for consistency, though not directly used in fitting logic.
# intercept parameter from original API seems unused/superceded by fit_intercept, so omitted from Python version's direct use.


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
    fixed_coefficients=None,
    fixed_intercept=None,
):
    """Create a Python-native Ridge regression model with constraints."""

    class PythonRidgeWrapper:
        def __init__(self):
            self.lambda_value = lambda_value
            self.fit_intercept_param = fit_intercept
            self.standardize = standardize
            self.intercept_sign = intercept_sign

            self.coef_ = None
            self.intercept_ = 0.0
            self.logger = logging.getLogger(__name__)
            # _prediction_cache can be implemented if strictly needed, but direct computation is often fast.
            # For this version, predict will compute directly.

            self.full_coef_ = None
            self.df_int = 0

            self.fixed_coefficients_param = fixed_coefficients
            self.fixed_intercept_param = fixed_intercept

            self.X_scaler_ = None
            self.penalty_factor_param = penalty_factor
            self.lower_limits_param = lower_limits
            self.upper_limits_param = upper_limits

        def _objective_function(
            self,
            params_to_optimize,
            X_data,
            y_target,
            current_lambda,
            current_penalty_factors_subset,
            fitting_intercept_in_params_list,
        ):

            current_intercept_for_loss = 0.0
            coeffs_for_loss = params_to_optimize

            if fitting_intercept_in_params_list:
                current_intercept_for_loss = params_to_optimize[0]
                coeffs_for_loss = params_to_optimize[1:]

            predictions = X_data @ coeffs_for_loss + current_intercept_for_loss
            mse = 0.5 * np.sum((y_target - predictions) ** 2)

            l2_penalty = 0.0
            if (
                coeffs_for_loss.size > 0 and current_lambda > 0
            ):  # No penalty if no coeffs or lambda is 0
                pen_factors_to_apply = current_penalty_factors_subset
                if pen_factors_to_apply is None:
                    pen_factors_to_apply = np.ones_like(coeffs_for_loss)
                # Ensure it's a NumPy array for element-wise multiplication
                elif not isinstance(pen_factors_to_apply, np.ndarray):
                    pen_factors_to_apply = np.array(pen_factors_to_apply, dtype=float)

                l2_penalty = (
                    0.5
                    * current_lambda
                    * np.sum(pen_factors_to_apply * (coeffs_for_loss**2))
                )

            return mse + l2_penalty

        def fit(self, X, y):
            X_orig = np.asarray(X, dtype=float)
            y_orig = np.asarray(y, dtype=float)

            n_features = X_orig.shape[1]
            y_adjusted_for_optim = y_orig.copy()

            # --- 1. Determine intercept behavior for optimization ---
            # actual_fit_intercept_in_optim: does the optimizer solve for an intercept variable?
            # intercept_offset_for_final_result: base value for final intercept_ (from fixed_intercept_param)
            actual_fit_intercept_in_optim = self.fit_intercept_param
            intercept_offset_for_final_result = 0.0
            self.df_int = 0  # Default df for intercept

            if self.fixed_intercept_param is not None:
                y_adjusted_for_optim -= self.fixed_intercept_param
                actual_fit_intercept_in_optim = False
                intercept_offset_for_final_result = self.fixed_intercept_param
                # self.df_int remains 0

            # --- 2. Handle fixed coefficients ---
            fit_cols = list(range(n_features))
            fixed_cols = []

            current_fixed_coeffs = self.fixed_coefficients_param
            if current_fixed_coeffs is not None:
                if len(current_fixed_coeffs) != n_features:
                    raise ValueError(
                        f"Length of fixed_coefficients ({len(current_fixed_coeffs)}) must match number of features ({n_features})"
                    )
                current_fixed_coeffs = np.asarray(current_fixed_coeffs, dtype=float)
                fixed_mask = ~np.isnan(
                    current_fixed_coeffs
                )  # Assuming None/NaN means not fixed

                fixed_cols = [i for i, is_fixed in enumerate(fixed_mask) if is_fixed]
                fit_cols = [i for i, is_fixed in enumerate(fixed_mask) if not is_fixed]

                if fixed_cols:
                    fixed_values = current_fixed_coeffs[fixed_cols]
                    y_adjusted_for_optim -= np.dot(X_orig[:, fixed_cols], fixed_values)

            X_to_fit = X_orig[:, fit_cols]

            # Subset constraints and penalty factors
            lower_limits_fit = None
            if self.lower_limits_param is not None and fit_cols:
                ll_param = np.asarray(self.lower_limits_param, dtype=float)
                lower_limits_fit = ll_param[fit_cols]

            upper_limits_fit = None
            if self.upper_limits_param is not None and fit_cols:
                ul_param = np.asarray(self.upper_limits_param, dtype=float)
                upper_limits_fit = ul_param[fit_cols]

            penalty_factor_fit_subset = None
            if self.penalty_factor_param is not None and fit_cols:
                if isinstance(self.penalty_factor_param, (int, float)):
                    penalty_factor_fit_subset = np.full(
                        len(fit_cols), float(self.penalty_factor_param)
                    )
                elif (
                    hasattr(self.penalty_factor_param, "__len__")
                    and len(self.penalty_factor_param) == n_features
                ):
                    pf_param = np.asarray(self.penalty_factor_param, dtype=float)
                    penalty_factor_fit_subset = pf_param[fit_cols]
                else:
                    raise ValueError(
                        f"penalty_factor list/array length ({len(self.penalty_factor_param)}) must match number of features ({n_features}) or be scalar."
                    )

            # --- Handle cases with no features to fit ---
            if not fit_cols:
                self.coef_ = np.zeros(n_features)
                if n_features > 0 and current_fixed_coeffs is not None:
                    # Ensure all are numbers, NaNs should have been handled or errored.
                    # If all fixed, current_fixed_coeffs should not contain NaNs.
                    self.coef_ = np.nan_to_num(current_fixed_coeffs)

                self.intercept_ = intercept_offset_for_final_result
                # If intercept was supposed to be fitted (not fixed) and no coeffs were fit:
                if self.fit_intercept_param and self.fixed_intercept_param is None:
                    if y_adjusted_for_optim.size > 0:
                        potential_intercept = np.mean(y_adjusted_for_optim)
                        if (
                            self.intercept_sign == "non_negative"
                            and potential_intercept < 0
                        ):
                            self.intercept_ = (
                                0.0  # Base is 0 as fixed_intercept_param is None
                            )
                            self.df_int = 0
                        else:
                            self.intercept_ = potential_intercept
                            self.df_int = 1
                    # else: self.intercept_ remains 0 (base is 0), self.df_int = 0
                self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])
                return self

            # --- 3. Standardization (of X_to_fit) ---
            X_for_optim_solver = X_to_fit
            self.X_scaler_ = None
            if self.standardize and X_to_fit.shape[0] > 0 and X_to_fit.shape[1] > 0:
                self.X_scaler_ = StandardScaler(with_mean=True, with_std=True)
                X_for_optim_solver = self.X_scaler_.fit_transform(X_to_fit)

            # --- 4. Optimization ---
            num_optim_coeffs = X_for_optim_solver.shape[1]
            optim_bounds_list = []
            if num_optim_coeffs > 0:
                low = (
                    lower_limits_fit
                    if lower_limits_fit is not None
                    else [-np.inf] * num_optim_coeffs
                )
                upp = (
                    upper_limits_fit
                    if upper_limits_fit is not None
                    else [np.inf] * num_optim_coeffs
                )
                for i in range(num_optim_coeffs):
                    l_b = low[i] if not np.isneginf(low[i]) else None
                    u_b = upp[i] if not np.isposinf(upp[i]) else None
                    optim_bounds_list.append((l_b, u_b))

            # Store df_int based on initial plan to fit intercept
            if actual_fit_intercept_in_optim:
                self.df_int = 1

            def run_optimization_routine(
                fit_interc_in_solver_flag, current_y_target_for_solver
            ):
                initial_guess_parts = []
                current_bounds_for_solver = list(optim_bounds_list)

                if fit_interc_in_solver_flag:
                    initial_guess_parts.append(0.0)
                    current_bounds_for_solver.insert(0, (None, None))

                if num_optim_coeffs > 0:
                    initial_guess_parts.extend([0.0] * num_optim_coeffs)

                initial_guess_for_solver = np.array(initial_guess_parts)

                # Handle case: only intercept, no coeffs (num_optim_coeffs == 0)
                if fit_interc_in_solver_flag and num_optim_coeffs == 0:
                    res_interc = (
                        np.mean(current_y_target_for_solver)
                        if current_y_target_for_solver.size > 0
                        else 0.0
                    )
                    return res_interc, np.array([])

                if initial_guess_for_solver.size == 0:  # No params to optimize
                    return 0.0, np.array([])

                opt_result = minimize(
                    self._objective_function,
                    initial_guess_for_solver,
                    args=(
                        X_for_optim_solver,
                        current_y_target_for_solver,
                        self.lambda_value,
                        penalty_factor_fit_subset,
                        fit_interc_in_solver_flag,
                    ),
                    method="L-BFGS-B",
                    bounds=current_bounds_for_solver,
                )

                res_interc_val, res_coeffs_val = 0.0, np.array([])
                if fit_interc_in_solver_flag:
                    res_interc_val = opt_result.x[0]
                    if num_optim_coeffs > 0:
                        res_coeffs_val = opt_result.x[1:]
                elif num_optim_coeffs > 0:
                    res_coeffs_val = opt_result.x
                return res_interc_val, res_coeffs_val

            # First optimization attempt
            # `actual_fit_intercept_in_optim` is True if not fixed_intercept and user wants to fit intercept
            opt_intercept_on_scaled_basis, opt_coeffs_on_scaled_basis = (
                run_optimization_routine(
                    actual_fit_intercept_in_optim, y_adjusted_for_optim
                )
            )

            # Refit if intercept sign constraint violated
            if (
                actual_fit_intercept_in_optim
                and self.intercept_sign == "non_negative"
                and opt_intercept_on_scaled_basis < 0
            ):
                self.logger.debug(
                    "Intercept < 0 & non_negative constraint. Refitting, scaled intercept = 0."
                )
                # Second attempt: do not fit intercept (effectively fix scaled intercept at 0)
                _, opt_coeffs_on_scaled_basis = run_optimization_routine(
                    False, y_adjusted_for_optim
                )
                opt_intercept_on_scaled_basis = 0.0  # Explicitly set for this path
                self.df_int = 0
            # df_int already set if actual_fit_intercept_in_optim was True and no refit,
            # or if actual_fit_intercept_in_optim was False initially.

            # --- 5. Denormalize / finalize intercept and coefficients ---
            final_coeffs_fitted_part_orig_scale = opt_coeffs_on_scaled_basis
            final_intercept_intrinsic_orig_scale = opt_intercept_on_scaled_basis

            if self.X_scaler_ is not None and opt_coeffs_on_scaled_basis.size > 0:
                # Avoid division by zero or issues if scale_ is zero (constant feature)
                # StandardScaler sets scale_ to 1 for constant features, so direct division is fine.
                final_coeffs_fitted_part_orig_scale = (
                    opt_coeffs_on_scaled_basis / self.X_scaler_.scale_
                )

                intercept_adjustment_from_X_mean = -np.dot(
                    self.X_scaler_.mean_, final_coeffs_fitted_part_orig_scale
                )
                final_intercept_intrinsic_orig_scale += intercept_adjustment_from_X_mean

            self.intercept_ = (
                final_intercept_intrinsic_orig_scale + intercept_offset_for_final_result
            )

            # Reconstruct full coefficient vector for X_orig
            self.coef_ = np.zeros(n_features)
            if current_fixed_coeffs is not None and fixed_cols:  # Fill in fixed values
                self.coef_[fixed_cols] = np.nan_to_num(current_fixed_coeffs[fixed_cols])

            if fit_cols:  # Fill in fitted values
                fitted_vals_to_assign = np.array(
                    final_coeffs_fitted_part_orig_scale
                ).flatten()
                self.coef_[fit_cols] = fitted_vals_to_assign

            self.full_coef_ = np.concatenate([[self.intercept_], self.coef_])
            return self

        def predict(self, X):
            X_arr = np.asarray(X, dtype=float)
            if self.coef_ is None or self.full_coef_ is None:
                raise RuntimeError("Model has not been fitted yet.")
            return np.dot(X_arr, self.coef_) + self.intercept_

        def get_full_coefficients(self):
            if self.full_coef_ is None:
                self.logger.warning(
                    "Full coefficients requested before model fitting or if fitting failed."
                )
            return self.full_coef_

    return PythonRidgeWrapper()


def _create_ridge_model_rpy2(
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
