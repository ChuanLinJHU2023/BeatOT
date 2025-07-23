import pulp
import numpy as np
from scipy.spatial import distance


def calculate_causal_distance(Matrix1, Matrix2, costs, options=None):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming.
    Note that we transport from P(X^, Y^)  to P(X, Y)
    In other words, Matrix 1 is P(X^, Y^) while Matrix 2 is P(X, Y)
    All elements in Matrix 1 and Matrix 2 must be non-negative. But their sum can be not 1

    Parameters:
    - Matrix1: probabilities for the first distribution (size M x I) where M = |X^| and I = |Y^|
    - Matrix2: probabilities for the second distribution (size N x J) where N = |X| and J = |Y|
    - costs: 4D list or array of costs/cost matrix between points (size M x I x N x J)
    - options: options for solver For example: {"msg":False, "gapRel":0.25}

    Returns:
    - causal_distance: the minimal cost (scalar)
    - transport_plan: the optimal transportation plan matrix (numpy array)
    """
    assert np.all(Matrix1 >= 0)
    assert np.all(Matrix2 >= 0)
    Matrix1 = np.array(Matrix1)
    Matrix2 = np.array(Matrix2)
    costs = np.array(costs)
    Matrix1 = Matrix1 / np.sum(Matrix1)
    Matrix2 = Matrix2 / np.sum(Matrix2)
    M, I = Matrix1.shape
    N, J = Matrix2.shape
    assert (M, I, N, J) == costs.shape

    # Step 1: Initialize Problem
    prob = pulp.LpProblem("Causal_Distance", pulp.LpMinimize)

    # Step 2: Initialize Variables
    T = {}
    for m in range(M):
        for i in range(I):
            # This is a speed up trick. If P(X^, Y^)==0, P~(X^, Y^, X, Y) in transport plan must be 0. We don't need a variable for it
            if Matrix1[m,i]==0:
                continue
            for n in range(N):
                for j in range(J):
                    if Matrix2[n,j]==0:
                        continue
                    T[(m, i, n, j)] = pulp.LpVariable(f"T_{m}_{i}_{n}_{j}", lowBound=0)

    # Step 3: Initialize Objective---minimize total transportation cost
    prob += pulp.lpSum([costs[m, i, n, j] * T[(m, i, n, j)]
                        for m in range(M) for i in range(I) for n in range(N) for j in range(J)
                        if Matrix1[m,i]>0 and Matrix2[n,j]>0])

    # Step 4: Initialize Constraints---two marginal constrains and one causal constrain
    for m in range(M):
        for i in range(I):
            if Matrix1[m,i]==0:
                continue
            # Marginal Constrain 1: P~(X^ = m , Y^ = i) == P(X^ = m, Y^ = i)
            prob += (pulp.lpSum([T[(m, i, n, j)] for n in range(N) for j in range(J) if Matrix2[n,j]>0])
                     == Matrix1[m, i]
                     , f"marginal_constrain1_{m}_{i}")
    for n in range(N):
        for j in range(J):
            if Matrix2[n,j]==0:
                continue
            # Marginal Constrain 2: P~(X = n , Y = j) == P(X = n, Y = j)
            prob += (pulp.lpSum([T[(m, i, n, j)] for m in range(M) for i in range(I) if Matrix1[m,i]>0])
                     == Matrix2[n, j]
                     , f"marginal_constrain2_{n}_{j}")
    for m in range(M):
        for i in range(I):
            if Matrix1[m,i]==0:
                continue
            conditional_prob_of_i_given_m = Matrix1[m, i] / np.sum(Matrix1[m])
            for n in range(N):
                # Causal Constrain: P~(X^ = m , Y^ = i, X = n) == P~(X^ = m, X = n) * P(Y^ = i | X^ = m)
                # This is the equivalent expression of causal independence ( X ind Y^ | X^ )
                # Given X^, X is independent of Y
                prob += (pulp.lpSum([T[(m, i, n, j)] for j in range(J) if Matrix2[n,j]>0])
                    == pulp.lpSum([T[(m, i_, n, j)] for i_ in range(I) for j in range(J) if Matrix2[n,j]>0 and Matrix1[m,i_]>0]) * conditional_prob_of_i_given_m
                    , f"causality_constrain_{m}_{i}_{n}")

    # Step 4: Solve Linear Programming
    if options:
        prob.solve(pulp.GUROBI(**options))
    else:
        prob.solve(pulp.GUROBI())

    # Step5: Retrieve Results
    transport_plan = np.zeros((M, I, N, J))
    for m in range(M):
        for i in range(I):
            if Matrix1[m,i]==0:
                continue
            for n in range(N):
                for j in range(J):
                    if Matrix2[n,j]==0:
                        continue
                    transport_plan[m, i, n, j] = pulp.value(T[(m, i, n, j)])
    causal_distance = pulp.value(prob.objective)
    return causal_distance, transport_plan




def calculate_causal_distance2(Matrix1, Matrix2, costs):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming.
    Note that we transport from Matrix 1 to Matrix 2
    In other words, Matrix 1 is P(X^, Y^) while Matrix 2 is P(X, Y)

    Parameters:
    - Matrix1: probabilities for the first distribution (size M x I)
    - Matrix2: probabilities for the second distribution (size N x J)
    - costs: 4D list or array of costs/cost matrix between points (size M x I x N x J)

    Returns:
    - causal_distance: the minimal cost (scalar)
    - transport_plan: the optimal transportation plan matrix (numpy array)
    """
    Matrix1 = np.array(Matrix1)
    Matrix2 = np.array(Matrix2)
    Matrix1 = Matrix1 / np.sum(Matrix1)
    Matrix2 = Matrix2 / np.sum(Matrix2)
    costs = np.array(costs)
    M, I = Matrix1.shape
    N, J = Matrix2.shape
    assert (M, I, N, J) == costs.shape
    assert np.all(Matrix1 >= 0)
    assert np.all(Matrix2 >= 0)
    prob = pulp.LpProblem("Causal_Distance", pulp.LpMinimize)
    T = {}
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    T[(m, i, n, j)] = pulp.LpVariable(f"T_{m}_{i}_{n}_{j}", lowBound=0)
    # Objective: minimize total transportation cost
    prob += pulp.lpSum([costs[m, i, n, j] * T[(m, i, n, j)] for m in range(M) for i in range(I) for n in range(N) for j in range(J)])
    # Constraints: marginals must match the distributions
    for m in range(M):
        for i in range(I):
            # P~(X^ = m , Y^ = i) == P(X^ = m, Y^ = i)
            prob += pulp.lpSum([T[(m, i, n, j)] for n in range(N) for j in range(J)]) == Matrix1[m, i], f"marginal_prob_Xhat_Yhat__{m}_{i}"
    for n in range(N):
        for j in range(J):
            # P~(X = n , Y = j) == P(X = n, Y = j)
            prob += pulp.lpSum([T[(m, i, n, j)] for m in range(M) for i in range(I)]) == Matrix2[n, j], f"marginal_prob_X_Y_{n}_{j}"
    # Constraints: causality
    # Given X^, X is independent of Y
    for m in range(M):
        if np.sum(Matrix1[m]) == 0:
            # in the case, P(X^ = m) = 0 and P(Y^ = i | X^ = m) is undefined. Causality is satisfied automatically
            continue
        for i in range(I):
            for n in range(N):
                conditional_prob_of_i_given_m = Matrix1[m,i] / np.sum(Matrix1[m])
                # P~(X^ = m , Y^ = i, X = n) == P~(X^ = m, X = n) * P(Y^ = i | X^ = m)
                prob += \
                    pulp.lpSum([T[(m, i, n, j)] for j in range(J)]) \
                    == \
                    pulp.lpSum([T[(m, i, n, j)] for i in range(I) for j in range(J)]) * conditional_prob_of_i_given_m \
                    , f"causality_{m}_{i}_{n}"
    prob.solve(pulp.GUROBI())
    transport_plan = np.zeros((M, I, N, J))
    for m in range(M):
        for i in range(I):
            for n in range(N):
                for j in range(J):
                    transport_plan[m, i, n, j] = pulp.value(T[(m, i, n, j)])
    causal_distance = pulp.value(prob.objective)
    return causal_distance, transport_plan



# Set your dimensions
M, I = 3, 4  # Example sizes for Matrix1
N, J = 4, 5  # Example sizes for Matrix2

Matrix1 = np.random.rand(M, I)
Matrix1[2,0]=0
Matrix1[0,2]=0
Matrix2 = np.random.rand(N, J)
Matrix2[2,1]=0
Matrix2[1,2]=0
costs = np.random.rand(M, I, N, J)


calculate_causal_distance(Matrix1, Matrix2, costs)
calculate_causal_distance2(Matrix1, Matrix2, costs)