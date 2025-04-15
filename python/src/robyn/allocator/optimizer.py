import nlopt
import numpy as np
from typing import Optional, Dict, Any, List, Union


def run_optimization(
    x0: np.ndarray,
    eval_f: callable,
    eval_g_eq: Optional[callable],
    eval_g_ineq: Optional[callable],
    lb: np.ndarray,
    ub: np.ndarray,
    maxeval: int,
    local_opts: Dict[str, Any],
    target_value: Optional[float] = None,
    eval_dict: Optional[Dict] = None,
) -> Dict:
    """Run nlopt optimization, matching R's nloptr interface"""

    # Create optimizer with AUGLAG algorithm
    opt = nlopt.opt(nlopt.LD_AUGLAG, len(x0))

    # Set local optimizer based on local_opts
    if local_opts["algorithm"] == "NLOPT_LD_SLSQP":
        local_optimizer = nlopt.opt(nlopt.LD_SLSQP, len(x0))
    elif local_opts["algorithm"] == "NLOPT_LD_MMA":
        local_optimizer = nlopt.opt(nlopt.LD_MMA, len(x0))
    else:
        raise ValueError(f"Unsupported local optimizer: {local_opts['algorithm']}")

    local_optimizer.set_xtol_rel(local_opts["xtol_rel"])
    local_optimizer.set_ftol_rel(1.0e-4)  # Additional stopping criterion
    opt.set_local_optimizer(local_optimizer)

    # Store eval_dict as an attribute so it can be accessed in the wrappers
    if eval_dict is not None:
        eval_f.eval_dict = eval_dict
        if eval_g_eq is not None:
            eval_g_eq.eval_dict = eval_dict
        if eval_g_ineq is not None:
            eval_g_ineq.eval_dict = eval_dict

    # Wrap evaluation functions to handle dictionary returns
    def objective_wrapper(x, grad):
        result = eval_f(x, grad)  # Match lambda signature
        if grad.size > 0:
            grad[:] = result["gradient"]
        return result["objective"]

    def eq_constraint_wrapper(x, grad):
        result = eval_g_eq(x, grad)  # Match lambda signature
        if grad.size > 0:
            grad[:] = result["jacobian"]
        return result["constraints"]

    def ineq_constraint_wrapper(x, grad):
        result = eval_g_ineq(x, grad)  # Match lambda signature
        if grad.size > 0:
            grad[:] = result["jacobian"]
        return result["constraints"]

    # Set objective and constraints with wrappers
    opt.set_min_objective(objective_wrapper)

    # Set bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # Set constraints using wrappers
    if eval_g_eq is not None:
        opt.add_equality_constraint(eq_constraint_wrapper, 1e-4)  # Increased tolerance
    if eval_g_ineq is not None:
        opt.add_inequality_constraint(
            ineq_constraint_wrapper, 1e-4
        )  # Increased tolerance

    # Set options
    opt.set_maxeval(maxeval)
    opt.set_xtol_rel(1.0e-4)  # Increased from 1.0e-10 to be less strict
    opt.set_ftol_rel(1.0e-4)  # Add function tolerance criteria
    opt.set_xtol_abs(1e-4)  # Absolute tolerance for x

    # Force optimizer to take at least some steps
    opt.set_initial_step(np.ones(len(x0)) * 0.1)  # Force a reasonable initial step size

    # Check if initial point is already at optimum
    initial_eval = eval_f(x0, np.zeros_like(x0))
    initial_gradient_norm = np.linalg.norm(initial_eval["gradient"])
    print(
        f"Initial objective: {initial_eval['objective']}, gradient norm: {initial_gradient_norm}"
    )
    print(f"Initial point: {x0}")

    # Also check if initial point satisfies constraints
    if eval_g_eq is not None:
        initial_eq = eval_g_eq(x0, np.zeros_like(x0))
        print(f"Initial equality constraint value: {initial_eq['constraints']}")

    # Try multi-start optimization if we have enough channels
    best_objective = float("inf")
    best_solution = None

    # Always try the provided starting point
    try:
        x = opt.optimize(x0)
        result = {
            "solution": x,
            "objective": opt.last_optimum_value(),
            "status": opt.last_optimize_result(),
            "initial_gradient_norm": initial_gradient_norm,
        }

        # Check if solution is different from initial point
        change = (
            np.linalg.norm(x - x0) / np.linalg.norm(x0)
            if np.linalg.norm(x0) > 0
            else np.linalg.norm(x - x0)
        )
        print(f"Solution change from initial point: {change:.6f}")

        best_objective = opt.last_optimum_value()
        best_solution = x

    except Exception as e:
        print(f"Optimization failed with initial point: {str(e)}")
        result = {
            "solution": x0,
            "objective": float("inf"),
            "status": -1,
            "error": str(e),
        }

    # If we have more than 2 channels, try a few random starting points
    if len(x0) > 2 and "objective" in result and result["objective"] != float("inf"):
        # Number of random starts - don't make it too large
        n_starts = min(5, len(x0))

        for i in range(n_starts):
            # Generate a random point that satisfies the budget constraint
            # Start with random proportions
            random_proportions = np.random.dirichlet(np.ones(len(x0)))

            # Scale to fit our budget and bounds
            random_point = random_proportions * eval_dict["total_budget_unit"]

            # Ensure we respect bounds
            random_point = np.maximum(random_point, lb)
            random_point = np.minimum(random_point, ub)

            # Renormalize to match budget exactly
            random_point = random_point * (
                eval_dict["total_budget_unit"] / np.sum(random_point)
            )

            try:
                x_random = opt.optimize(random_point)
                obj_random = opt.last_optimum_value()

                print(f"Random start {i+1}: objective = {obj_random}")

                # If this solution is better, keep it
                if obj_random < best_objective:
                    best_objective = obj_random
                    best_solution = x_random
                    print(f"Found better solution with random start {i+1}")
            except Exception as e:
                print(f"Random start {i+1} failed: {str(e)}")

    # Return the best solution found
    if best_solution is not None:
        result["solution"] = best_solution
        result["objective"] = best_objective

    # If we still haven't improved, create a "synthetic" solution
    if np.array_equal(result["solution"], x0) and len(x0) > 1:
        print("Optimizer didn't change initial point. Creating synthetic solution...")

        # Try to create a better solution based on marginal ROI
        channel_responses = np.array(
            [
                fx_objective(
                    x=x_val,
                    coeff=eval_dict["coefs_eval"][channel],
                    alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
                    inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
                    x_hist_carryover=np.mean(eval_dict["hist_carryover_eval"][channel]),
                )
                for x_val, channel in zip(x0, eval_dict["coefs_eval"].keys())
            ]
        )

        # Calculate marginal responses for +1 unit of spend
        marginal_responses = np.array(
            [
                fx_objective(
                    x=x_val + 1,
                    coeff=eval_dict["coefs_eval"][channel],
                    alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
                    inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
                    x_hist_carryover=np.mean(eval_dict["hist_carryover_eval"][channel]),
                )
                - resp
                for x_val, resp, channel in zip(
                    x0, channel_responses, eval_dict["coefs_eval"].keys()
                )
            ]
        )

        # Get marginal ROI
        marginal_roi = marginal_responses / 1.0  # per unit of spend

        # Rank channels by marginal ROI
        channel_ranks = np.argsort(-marginal_roi)  # Descending order

        # Create a new allocation - start with lower bounds
        new_solution = lb.copy()

        # Calculate remaining budget
        remaining_budget = eval_dict["total_budget_unit"] - np.sum(new_solution)

        # Allocate from best to worst channels
        for idx in channel_ranks:
            # Calculate how much we can allocate to this channel
            available_headroom = ub[idx] - new_solution[idx]
            allocation = min(remaining_budget, available_headroom)

            # Allocate
            new_solution[idx] += allocation
            remaining_budget -= allocation

            # If no more budget, break
            if remaining_budget < 1e-6:
                break

        # Test if this new solution is better
        new_responses = np.array(
            [
                fx_objective(
                    x=x_val,
                    coeff=eval_dict["coefs_eval"][channel],
                    alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
                    inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
                    x_hist_carryover=np.mean(eval_dict["hist_carryover_eval"][channel]),
                )
                for x_val, channel in zip(new_solution, eval_dict["coefs_eval"].keys())
            ]
        )

        new_objective = -np.sum(new_responses)

        if new_objective < result["objective"]:
            print(
                f"Synthetic solution is better: {new_objective} vs {result['objective']}"
            )
            result["solution"] = new_solution
            result["objective"] = new_objective
            result["status"] = 5  # Custom status code indicating synthetic solution

    return result


def eval_f(
    X: np.ndarray,
    grad_or_eval_dict: Union[np.ndarray, Dict[str, Any]],
    target_value: Optional[float] = None,
) -> Union[Dict, float]:
    """
    Evaluation function for optimization. Can be called in two ways:
    1. eval_f(X, eval_dict, target_value) - direct call
    2. eval_f(x, grad) - called through lambda in nlopt

    Args:
        X: Input array of spend values
        grad_or_eval_dict: Either gradient array (nlopt) or eval_dict (direct)
        target_value: Optional target value

    Returns:
        Dictionary containing objective, gradient, and per-channel objectives
    """
    # Determine if this is a direct call or through nlopt
    if isinstance(grad_or_eval_dict, dict):
        # Direct call with eval_dict
        eval_dict = grad_or_eval_dict
        grad = None
    else:
        # Called through nlopt with grad
        eval_dict = getattr(eval_f, "eval_dict", None)
        grad = grad_or_eval_dict

    # Unpack evaluation dictionary
    coefs_eval = eval_dict["coefs_eval"]
    alphas_eval = eval_dict["alphas_eval"]
    inflexions_eval = eval_dict["inflexions_eval"]
    hist_carryover_eval = eval_dict["hist_carryover_eval"]

    # Calculate mean carryover
    x_hist_carryover = {k: np.mean(v) for k, v in hist_carryover_eval.items()}

    # Calculate objective for each channel and convert to numpy array
    objective_channel = np.array(
        [
            fx_objective(
                x=x,
                coeff=coefs_eval[channel],
                alpha=alphas_eval[f"{channel}_alphas"],
                inflexion=inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=x_hist_carryover[channel],
            )
            for channel, x in zip(coefs_eval.keys(), X)
        ]
    )

    # Calculate total objective
    objective = -np.sum(objective_channel)

    # Calculate gradient
    gradient = np.array(
        [
            fx_gradient(
                x=x,
                coeff=coefs_eval[channel],
                alpha=alphas_eval[f"{channel}_alphas"],
                inflexion=inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=x_hist_carryover[channel],
            )
            for channel, x in zip(coefs_eval.keys(), X)
        ]
    )

    # Apply gradient scaling to prevent getting stuck in flat regions
    # Scale up small gradients to encourage movement
    grad_norm = np.linalg.norm(gradient)
    if grad_norm < 0.001:  # If gradient is very small
        scaling_factor = 0.001 / max(grad_norm, 1e-10)  # Avoid division by zero
        gradient = gradient * scaling_factor

    # Update gradient if provided
    if grad is not None and grad.size > 0:
        grad[:] = gradient

    return {
        "objective": objective,
        "gradient": gradient,
        "objective_channel": objective_channel,
    }


def fx_objective(
    x: float,
    coeff: float,
    alpha: float,
    inflexion: float,
    x_hist_carryover: float,
    get_sum: bool = True,
) -> float:
    """
    Calculate objective function using Hill transformation

    Args:
        x: Input value
        coeff: Coefficient
        alpha: Alpha parameter (with _alphas suffix in name)
        inflexion: Inflexion parameter (with _gammas suffix in name)
        x_hist_carryover: Historical carryover value
        get_sum: Whether to sum the result

    Returns:
        float: Transformed value
    """
    # Adstock scales
    x_adstocked = x + x_hist_carryover

    # Hill transformation
    if get_sum:
        x_out = coeff * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)
    else:
        x_out = coeff * ((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

    return x_out


def fx_gradient(
    x: float, coeff: float, alpha: float, inflexion: float, x_hist_carryover: float
) -> float:
    """
    Calculate gradient of the objective function
    Source: https://www.derivative-calculator.net/ on the objective function 1/(1+gamma^alpha / x^alpha)

    Args:
        x: Input value
        coeff: Coefficient
        alpha: Alpha parameter
        inflexion: Inflexion parameter (gamma)
        x_hist_carryover: Historical carryover value

    Returns:
        float: Gradient value
    """
    # Adstock scales
    x_adstocked = x + x_hist_carryover

    # Calculate gradient
    x_out = -coeff * np.sum(
        (alpha * (inflexion**alpha) * (x_adstocked ** (alpha - 1)))
        / (x_adstocked**alpha + inflexion**alpha) ** 2
    )

    return x_out


def fx_objective_channel(
    x: float, coeff: float, alpha: float, inflexion: float, x_hist_carryover: float
) -> float:
    """
    Calculate per-channel objective

    Args:
        x: Input value
        coeff: Coefficient
        alpha: Alpha parameter
        inflexion: Inflexion parameter
        x_hist_carryover: Historical carryover value

    Returns:
        float: Channel objective value
    """
    # Adstock scales
    x_adstocked = x + x_hist_carryover

    # Calculate channel objective
    x_out = -coeff * np.sum((1 + inflexion**alpha / x_adstocked**alpha) ** -1)

    return x_out


def eval_g_eq(
    X: np.ndarray,
    grad_or_eval_dict: Union[np.ndarray, Dict[str, Any]],
    target_value: Optional[float] = None,
) -> Union[Dict, float]:
    """
    Equality constraint evaluation. Can be called in two ways:
    1. eval_g_eq(X, eval_dict, target_value) - direct call
    2. eval_g_eq(x, grad) - called through lambda in nlopt
    """
    # Determine if this is a direct call or through nlopt
    if isinstance(grad_or_eval_dict, dict):
        eval_dict = grad_or_eval_dict
        grad = None
    else:
        eval_dict = getattr(eval_g_eq, "eval_dict", None)
        grad = grad_or_eval_dict

    constr = np.sum(X) - eval_dict["total_budget_unit"]
    gradient = np.ones(len(X))

    # Update gradient if provided
    if grad is not None and grad.size > 0:
        grad[:] = gradient

    return {"constraints": constr, "jacobian": gradient}


def eval_g_ineq(
    X: np.ndarray, eval_dict: Dict[str, Any], target_value: float = None
) -> Dict:
    """
    Inequality constraint evaluation

    Args:
        X: Input array
        eval_dict: Dictionary containing evaluation parameters
        target_value: Optional target value

    Returns:
        Dict with constraints and jacobian
    """
    constr = np.sum(X) - eval_dict["total_budget_unit"]
    grad = np.ones(len(X))

    return {"constraints": constr, "jacobian": grad}


def eval_g_eq_effi(
    X: np.ndarray, eval_dict: Dict[str, Any], target_value: float = None
) -> Dict:
    """
    Efficiency constraint evaluation

    Args:
        X: Input array
        eval_dict: Dictionary containing evaluation parameters
        target_value: Optional target value

    Returns:
        Dict with constraints and jacobian
    """
    # Calculate sum response
    channels = list(eval_dict["coefs_eval"].keys())
    sum_response = np.sum(
        fx_objective(
            x=x,
            coeff=eval_dict["coefs_eval"][channel],
            alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
            inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
            x_hist_carryover=eval_dict["hist_carryover_eval"][channel],
        )
        for channel, x in zip(channels, X)
    )

    # Calculate constraint based on target value and dep_var_type
    target = target_value if target_value is not None else eval_dict["target_value"]
    if eval_dict["dep_var_type"] == "conversion":
        constr = np.sum(X) - sum_response * target
    else:
        constr = np.sum(X) - sum_response / target

    # Calculate gradient
    grad = np.ones(len(X)) - np.array(
        [
            fx_gradient(
                x=x,
                coeff=eval_dict["coefs_eval"][channel],
                alpha=eval_dict["alphas_eval"][f"{channel}_alphas"],
                inflexion=eval_dict["inflexions_eval"][f"{channel}_gammas"],
                x_hist_carryover=eval_dict["hist_carryover_eval"][channel],
            )
            for channel, x in zip(channels, X)
        ]
    )

    return {"constraints": constr, "jacobian": grad}
