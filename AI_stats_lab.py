"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    """
    STEP 1: Consider a standard 52-card deck.
            Assume 4 Aces.

    STEP 2: Compute analytically:
            - P(A)
            - P(B)
            - P(B | A)
            - P(A ∩ B)

    STEP 3: Check independence:
            P(A ∩ B) ?= P(A)P(B)

    STEP 4: Simulate 200,000 experiments
            WITHOUT replacement.
            Use random_state=42.

            Estimate:
            - empirical P(A)
            - empirical P(B | A)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(B | A)
            empirical P(B | A)

    RETURN:
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    """


    P_A = 4 / 52
    P_B = 3 / 52
    P_B_given_A = 3 / 51
    P_AB = (4 / 52) * (3 / 51)

    # Independence check (not returned, but computed logically)
    # P_AB == P_A * P_B ?  -> False


    rng = np.random.default_rng(42)
    trials = 200_000

    count_A = 0
    count_A_and_B = 0

    deck = np.array([1]*4 + [0]*48)  # 1 = Ace

    for _ in range(trials):
        draw = rng.choice(deck, size=2, replace=False)
        first, second = draw

        if first == 1:
            count_A += 1
            if second == 1:
                count_A_and_B += 1

    empirical_P_A = count_A / trials
    empirical_P_B_given_A = (
        count_A_and_B / count_A if count_A > 0 else 0
    )

    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    )    


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    """
    STEP 1: Define Bernoulli(p) PMF:
            p_X(x) = p^x (1-p)^(1-x)

    STEP 2: Compute theoretical:
            - P(X = 1)
            - P(X = 0)

    STEP 3: Simulate 100,000 bulbs
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X = 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X = 1)
            empirical P(X = 1)

    RETURN:
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    """
    pmf = lambda x: p**x * (1-p)**(1-x) if x in [0,1] else 0

    theoretical_P_X_1 = pmf(1)
    theoretical_P_X_0 = pmf(0)

    rng = np.random.default_rng(42)
    trials = 100_000
    count_X_1 = sum(1 for _ in range(trials) if rng.choice([0, 1], p=[1-p, p]) == 1)
    empirical_P_X_1 = count_X_1 / trials

    absolute_error = abs(theoretical_P_X_1 - empirical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    )

# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    """
    STEP 1: Define Binomial(n,p) PMF:
            P(X=k) = C(n,k)p^k(1-p)^(n-k)

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 2)
            - P(X ≥ 1)

    STEP 3: Simulate 100,000 inspections
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 1)
            empirical P(X ≥ 1)

    RETURN:
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    """

    P_0 = (1-p)**n
    p_2 = math.comb(n, 2) * (p**2) * ((1-p)**(n-2))
    P_ge_1 = 1 - P_0

    rng = np.random.default_rng(42)
    trails = 100000
    count_ge_1 = sum(1 for _ in range(trails) if sum(rng.choice([0, 1], p=[1-p, p], size=n)) >= 1)
    empirical_P_ge_1 = count_ge_1 / trails
    absolute_error = abs(P_ge_1 - empirical_P_ge_1)

    return (
        P_0,
        p_2,    
        P_ge_1,
        empirical_P_ge_1,
        absolute_error
    )

# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    """
    STEP 1: Let p = 1/6.

    STEP 2: Define Geometric PMF:
            P(X=k) = (5/6)^(k-1)*(1/6)

    STEP 3: Compute theoretical:
            - P(X = 1)
            - P(X = 3)
            - P(X > 4)

    STEP 4: Simulate 200,000 experiments
            using random_state=42.

    STEP 5: Compute empirical:
            - empirical P(X > 4)

    STEP 6: Compute absolute error BETWEEN:
            theoretical P(X > 4)
            empirical P(X > 4)

    RETURN:
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    """

    p = 1 / 6


    P_1 = p
    P_3 = ((5/6) ** 2) * p
    P_gt_4 = (5/6) ** 4


    rng = np.random.default_rng(42)
    trials = 200_000

    samples = rng.geometric(p, size=trials)
    empirical_P_gt_4 = np.mean(samples > 4)

    absolute_error = abs(P_gt_4 - empirical_P_gt_4)

    return (
        P_1,
        P_3,
        P_gt_4,
        empirical_P_gt_4,
        absolute_error
    )

# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    """
    STEP 1: Define Poisson PMF:
            P(X=k) = e^(-λ) λ^k / k!

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 15)
            - P(X ≥ 18)

    STEP 3: Simulate 100,000 hours
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 18)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 18)
            empirical P(X ≥ 18)

    RETURN:
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    """

    P_0 = math.exp(-lam)
    P_15 = math.exp(-lam) * (lam ** 15) / math.factorial(15)

    P_ge_18 = 1 - sum(
        math.exp(-lam) * (lam ** k) / math.factorial(k)
        for k in range(18)
    )


    rng = np.random.default_rng(42)
    trials = 100_000

    samples = rng.poisson(lam, size=trials)
    empirical_P_ge_18 = np.mean(samples >= 18)

    absolute_error = abs(P_ge_18 - empirical_P_ge_18)

    return (
        P_0,
        P_15,
        P_ge_18,
        empirical_P_ge_18,
        absolute_error
    )