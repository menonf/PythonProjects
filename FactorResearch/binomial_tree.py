"""
Black-Derman-Toy (BDT) Lognormal Combining Short-Rate Binomial Tree
====================================================================

Calibrated to a spot curve using continuously-compounded rates
and risk-neutral probabilities (p = 0.5).

Features:
    1. Bootstrap spot rates from par rates
    2. Build arbitrage-free short-rate tree
    3. Price option-free bonds
    4. Price callable / putable bonds
    5. Compute Option-Adjusted Spread (OAS)
    6. Compute embedded option value
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# 1. Bootstrap spot rates from par rates
# ---------------------------------------------------------------------------

def bootstrap_spot_rates(par_rates: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Bootstrap continuously-compounded spot rates from par rates.

    Parameters
    ----------
    par_rates : array of par rates (decimal) for maturities dt, 2*dt, ..., N*dt
    dt : time step in years

    Returns
    -------
    spot_rates : array of continuously-compounded spot rates
    """
    n = len(par_rates)
    spot_rates = np.zeros(n)

    for i in range(n):
        t = (i + 1) * dt
        c = par_rates[i]
        if i == 0:
            # 1 = (1 + c*dt) * exp(-z * t)  =>  z = ln(1 + c*dt) / t
            spot_rates[i] = np.log(1.0 + c * dt) / t
        else:
            # 1 = c*dt * sum_{k=1}^{i} exp(-z_k * k*dt) + (1 + c*dt) * exp(-z_{i+1} * t)
            pv_coupons = sum(
                c * dt * np.exp(-spot_rates[k] * (k + 1) * dt) for k in range(i)
            )
            # Solve: 1 - pv_coupons = (1 + c*dt) * exp(-z * t)
            remaining = 1.0 - pv_coupons
            spot_rates[i] = -np.log(remaining / (1.0 + c * dt)) / t

    return spot_rates


# ---------------------------------------------------------------------------
# 2. Build BDT short-rate tree
# ---------------------------------------------------------------------------

def build_bdt_tree(
    spot_rates: np.ndarray,
    sigma: float | np.ndarray,
    dt: float = 1.0,
) -> list[np.ndarray]:
    """
    Build a BDT lognormal recombining short-rate tree calibrated to the spot curve.

    At time step i with states j = 0, 1, ..., i:
        r(i, j) = a_i * exp(2 * sigma_i * j * sqrt(dt))

    Calibration: for each step i, solve for a_i such that the tree
    prices a zero-coupon bond maturing at (i+1)*dt correctly.

    Parameters
    ----------
    spot_rates : continuously-compounded spot rates for maturities dt, 2*dt, ...
    sigma : volatility (scalar for constant, array for term-structure of vol)
    dt : time step in years

    Returns
    -------
    tree : list of arrays; tree[i] has shape (i+1,) with short rates at step i
    """
    n = len(spot_rates)

    if np.isscalar(sigma):
        sigmas = np.full(n, sigma)
    else:
        sigmas = np.asarray(sigma)

    tree = []

    # Target zero-coupon bond prices from the spot curve
    zcb_prices = np.array([np.exp(-spot_rates[i] * (i + 1) * dt) for i in range(n)])

    for i in range(n):
        s = sigmas[i]

        if i == 0:
            # Single node: r(0,0) = a_0
            # Price of ZCB maturing at dt: P(0, dt) = exp(-r(0,0)*dt)
            # exp(-a_0 * dt) = zcb_prices[0]  =>  a_0 = -ln(zcb_prices[0]) / dt
            a = -np.log(zcb_prices[0]) / dt
            tree.append(np.array([a]))
        else:
            # At step i, states j = 0, ..., i
            # r(i, j) = a * exp(2 * s * j * sqrt(dt))
            # We need to price a ZCB maturing at (i+1)*dt by backward induction
            # through the existing tree to get target price zcb_prices[i].

            def pricing_error(log_a):
                a_trial = np.exp(log_a)
                rates_i = a_trial * np.exp(2.0 * s * np.arange(i + 1) * np.sqrt(dt))

                # Backward induction from step i to step 0
                # At step i: ZCB matures at step i+1, so value at (i, j) is
                # exp(-r(i,j)*dt) * 1.0
                values = np.exp(-rates_i * dt)

                # Roll back from step i-1 to step 0
                for step in range(i - 1, -1, -1):
                    new_values = np.zeros(step + 1)
                    for j in range(step + 1):
                        new_values[j] = np.exp(-tree[step][j] * dt) * 0.5 * (
                            values[j] + values[j + 1]
                        )
                    values = new_values

                return values[0] - zcb_prices[i]

            # Solve for a_i using Brent's method on log(a) for stability
            initial_guess = np.log(-np.log(zcb_prices[i]) / ((i + 1) * dt))
            try:
                log_a_sol = brentq(pricing_error, initial_guess - 10, initial_guess + 10)
            except ValueError:
                # Widen search if needed
                log_a_sol = brentq(pricing_error, -20, 20)

            a_sol = np.exp(log_a_sol)
            rates = a_sol * np.exp(2.0 * s * np.arange(i + 1) * np.sqrt(dt))
            tree.append(rates)

    return tree


# ---------------------------------------------------------------------------
# 3. Price option-free bond
# ---------------------------------------------------------------------------

def price_bond(
    tree: list[np.ndarray],
    coupon_rate: float,
    face: float = 100.0,
    dt: float = 1.0,
    spread: float = 0.0,
) -> float:
    """
    Price an option-free bond using backward induction through the rate tree.

    Parameters
    ----------
    tree : short-rate tree from build_bdt_tree
    coupon_rate : annual coupon rate (decimal)
    face : face value
    dt : time step in years
    spread : parallel spread added to all short rates (for OAS computation)

    Returns
    -------
    price : present value of the bond at time 0
    """
    n = len(tree)
    coupon = coupon_rate * face * dt

    # At maturity (step n): bond pays face + last coupon
    values = np.full(n + 1, face + coupon)

    # Backward induction
    for i in range(n - 1, -1, -1):
        new_values = np.zeros(i + 1)
        for j in range(i + 1):
            r = tree[i][j] + spread
            # Discount expected future value and add coupon at this step
            future = 0.5 * (values[j] + values[j + 1])
            if i > 0:
                new_values[j] = np.exp(-r * dt) * future + coupon
            else:
                new_values[j] = np.exp(-r * dt) * future + coupon
        values = new_values

    # At step 0, discount to get price (coupon at step 0 not included - clean price)
    # Actually the first coupon is at step 1 which is already included
    # The price at node 0 already includes all discounted cashflows
    return values[0] - coupon  # Remove double-counted first coupon


def price_option_free_bond(
    tree: list[np.ndarray],
    coupon_rate: float,
    face: float = 100.0,
    dt: float = 1.0,
    spread: float = 0.0,
) -> float:
    """
    Price an option-free bond via backward induction.

    Cash flows: coupon at each period end (steps 1..N), face at step N.

    Parameters
    ----------
    tree : short-rate tree from build_bdt_tree
    coupon_rate : annual coupon rate (decimal)
    face : face value
    dt : time step in years
    spread : parallel spread added to short rates

    Returns
    -------
    price at time 0
    """
    n = len(tree)
    coupon = coupon_rate * face * dt

    # Terminal values at step n (after last period)
    values = np.full(n + 1, face + coupon)

    # Backward induction from step n-1 to step 0
    for i in range(n - 1, -1, -1):
        new_values = np.zeros(i + 1)
        for j in range(i + 1):
            r = tree[i][j] + spread
            future = 0.5 * (values[j] + values[j + 1])
            pv = np.exp(-r * dt) * future
            # Add coupon received at end of period i (except at step 0, the price
            # represents value at time 0 before any coupon)
            if i > 0:
                pv += coupon
            new_values[j] = pv
        values = new_values

    return values[0]


# ---------------------------------------------------------------------------
# 4. Price callable / putable bonds
# ---------------------------------------------------------------------------

def price_callable_bond(
    tree: list[np.ndarray],
    coupon_rate: float,
    call_price: float,
    first_call_step: int = 1,
    face: float = 100.0,
    dt: float = 1.0,
    spread: float = 0.0,
) -> float:
    """
    Price a callable bond. The issuer can call (redeem) the bond at call_price
    at any step >= first_call_step.

    At each callable node, bond value = min(continuation value, call_price).

    Parameters
    ----------
    tree : short-rate tree
    coupon_rate : annual coupon rate (decimal)
    call_price : call (redemption) price
    first_call_step : first period at which the bond is callable
    face : face value
    dt : time step
    spread : OAS spread

    Returns
    -------
    callable bond price at time 0
    """
    n = len(tree)
    coupon = coupon_rate * face * dt

    values = np.full(n + 1, face + coupon)

    for i in range(n - 1, -1, -1):
        new_values = np.zeros(i + 1)
        for j in range(i + 1):
            r = tree[i][j] + spread
            future = 0.5 * (values[j] + values[j + 1])
            pv = np.exp(-r * dt) * future
            if i > 0:
                pv += coupon

            # Apply call constraint
            if i >= first_call_step:
                pv = min(pv, call_price + coupon)

            new_values[j] = pv
        values = new_values

    return values[0]


def price_putable_bond(
    tree: list[np.ndarray],
    coupon_rate: float,
    put_price: float,
    first_put_step: int = 1,
    face: float = 100.0,
    dt: float = 1.0,
    spread: float = 0.0,
) -> float:
    """
    Price a putable bond. The holder can put (sell back) the bond at put_price
    at any step >= first_put_step.

    At each putable node, bond value = max(continuation value, put_price).

    Parameters
    ----------
    tree : short-rate tree
    coupon_rate : annual coupon rate (decimal)
    put_price : put (redemption) price
    first_put_step : first period at which the bond is putable
    face : face value
    dt : time step
    spread : OAS spread

    Returns
    -------
    putable bond price at time 0
    """
    n = len(tree)
    coupon = coupon_rate * face * dt

    values = np.full(n + 1, face + coupon)

    for i in range(n - 1, -1, -1):
        new_values = np.zeros(i + 1)
        for j in range(i + 1):
            r = tree[i][j] + spread
            future = 0.5 * (values[j] + values[j + 1])
            pv = np.exp(-r * dt) * future
            if i > 0:
                pv += coupon

            # Apply put constraint
            if i >= first_put_step:
                pv = max(pv, put_price + coupon)

            new_values[j] = pv
        values = new_values

    return values[0]


# ---------------------------------------------------------------------------
# 5. Option-Adjusted Spread (OAS)
# ---------------------------------------------------------------------------

def compute_oas(
    tree: list[np.ndarray],
    market_price: float,
    coupon_rate: float,
    bond_type: str = "callable",
    call_price: float = 100.0,
    put_price: float = 100.0,
    first_exercise_step: int = 1,
    face: float = 100.0,
    dt: float = 1.0,
) -> float:
    """
    Compute the Option-Adjusted Spread (OAS) — the constant spread added
    to the short-rate tree that equates the model price to the market price.

    Parameters
    ----------
    tree : short-rate tree
    market_price : observed market price of the bond
    coupon_rate : annual coupon rate (decimal)
    bond_type : 'callable', 'putable', or 'bullet'
    call_price : call price (if callable)
    put_price : put price (if putable)
    first_exercise_step : first period at which option is exercisable
    face : face value
    dt : time step

    Returns
    -------
    oas : option-adjusted spread (continuously compounded)
    """

    def price_at_spread(s):
        if bond_type == "callable":
            return price_callable_bond(
                tree, coupon_rate, call_price, first_exercise_step, face, dt, spread=s
            )
        elif bond_type == "putable":
            return price_putable_bond(
                tree, coupon_rate, put_price, first_exercise_step, face, dt, spread=s
            )
        else:
            return price_option_free_bond(tree, coupon_rate, face, dt, spread=s)

    def objective(s):
        return price_at_spread(s) - market_price

    oas = brentq(objective, -0.50, 0.50)
    return oas


# ---------------------------------------------------------------------------
# 6. Embedded option value
# ---------------------------------------------------------------------------

def embedded_option_value(
    tree: list[np.ndarray],
    coupon_rate: float,
    bond_type: str = "callable",
    call_price: float = 100.0,
    put_price: float = 100.0,
    first_exercise_step: int = 1,
    face: float = 100.0,
    dt: float = 1.0,
) -> float:
    """
    Compute the value of the embedded option.

    For callable bonds:
        Option value = Price(option-free) - Price(callable)
        (issuer owns the call option, so callable < option-free)

    For putable bonds:
        Option value = Price(putable) - Price(option-free)
        (holder owns the put option, so putable > option-free)

    Parameters
    ----------
    tree : short-rate tree
    coupon_rate : annual coupon rate (decimal)
    bond_type : 'callable' or 'putable'
    call_price : call price
    put_price : put price
    first_exercise_step : first exercisable period
    face : face value
    dt : time step

    Returns
    -------
    option_value : value of the embedded option (always >= 0)
    """
    bullet_price = price_option_free_bond(tree, coupon_rate, face, dt)

    if bond_type == "callable":
        callable_price = price_callable_bond(
            tree, coupon_rate, call_price, first_exercise_step, face, dt
        )
        return bullet_price - callable_price
    elif bond_type == "putable":
        putable_price = price_putable_bond(
            tree, coupon_rate, put_price, first_exercise_step, face, dt
        )
        return putable_price - bullet_price
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Utility: Print the rate tree
# ---------------------------------------------------------------------------

def print_tree(tree: list[np.ndarray], label: str = "Short-Rate Tree (%)"):
    """Pretty-print the short-rate tree."""
    print(f"\n{label}")
    print("=" * 50)
    for i, rates in enumerate(tree):
        rates_pct = [f"{r * 100:.4f}%" for r in rates]
        print(f"  Step {i}: {rates_pct}")
    print()


# ---------------------------------------------------------------------------
# Example / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Example par curve (annual, in decimal) ---
    par_rates = np.array([0.03, 0.035, 0.04, 0.042, 0.044])

    # 1. Bootstrap spot rates
    spot_rates = bootstrap_spot_rates(par_rates, dt=1.0)
    print("Par Rates:  ", [f"{r*100:.3f}%" for r in par_rates])
    print("Spot Rates: ", [f"{r*100:.4f}%" for r in spot_rates])

    # 2. Build BDT tree (constant vol = 15%)
    sigma = 0.15
    tree = build_bdt_tree(spot_rates, sigma, dt=1.0)
    print_tree(tree)

    # 3. Price option-free bond (5% coupon, 5-year)
    coupon = 0.05
    face = 100.0
    bullet_price = price_option_free_bond(tree, coupon, face, dt=1.0)
    print(f"Option-Free Bond Price:  {bullet_price:.4f}")

    # 4a. Price callable bond (callable at par after year 2)
    callable_price = price_callable_bond(
        tree, coupon, call_price=100.0, first_call_step=2, face=face, dt=1.0
    )
    print(f"Callable Bond Price:     {callable_price:.4f}")

    # 4b. Price putable bond (putable at par after year 2)
    putable_price = price_putable_bond(
        tree, coupon, put_price=100.0, first_put_step=2, face=face, dt=1.0
    )
    print(f"Putable Bond Price:      {putable_price:.4f}")

    # 5. Compute OAS (assume market price for callable bond)
    market_price = callable_price - 0.50  # Suppose market is 50bp cheaper
    oas = compute_oas(
        tree,
        market_price,
        coupon,
        bond_type="callable",
        call_price=100.0,
        first_exercise_step=2,
        face=face,
        dt=1.0,
    )
    print(f"\nOAS (given mkt price {market_price:.4f}): {oas*10000:.2f} bps")

    # 6. Embedded option value
    call_value = embedded_option_value(
        tree, coupon, bond_type="callable", call_price=100.0, first_exercise_step=2
    )
    put_value = embedded_option_value(
        tree, coupon, bond_type="putable", put_price=100.0, first_exercise_step=2
    )
    print(f"\nCall Option Value:       {call_value:.4f}")
    print(f"Put Option Value:        {put_value:.4f}")
