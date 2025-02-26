import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


df = pd.read_csv('./data/Dataset.txt', header=None,
                 names=['Player', 'Penalty', 'FreeKick', 'Corner', 'Target'])


m = df.shape[0]
# Extract features and target
X = df[['Penalty', 'FreeKick', 'Corner']].values
y = df['Target'].values

# -----------------------------
# Define helper functions
# -----------------------------
def compute_cost(X, y, theta): 
    """dw
    Compute the cost for linear regression.
    NOTE: Here we use 1/m (not 1/(2*m)) so that the cost
    matches the values in your reference table.
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/m) * np.sum((predictions - y)**2) 
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    """
    Perform gradient descent to learn theta.
    """
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        # Gradient descent update
        theta = theta - (alpha/m) * (X.T.dot(error))
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

def run_experiment(iterations):
    """
    Run gradient descent for a given number of iterations and
    return the final cost, theta values, r2 score, and predictions.
    """
    alpha = 0.1
    theta = np.zeros(X.shape[1])
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    final_cost = cost_history[-1]
    predictions = X.dot(theta)
    r2 = r2_score(y, predictions)
    return final_cost, theta, r2, predictions

# -----------------------------
# Run experiments for different iterations
# -----------------------------
results = {}
for n in [1, 10, 100, 1000]:
    cost, theta_vals, r2, preds = run_experiment(n)
    results[n] = (cost, theta_vals, r2, preds)



# question 5

# We take the columns Penalty, FreeKick, Corner as features.
X = df[['Penalty', 'FreeKick', 'Corner']].values
y = df['Target'].values

# Number of training examples
m = len(y)

# Add a column of ones for the intercept term
# so that X becomes [1, Penalty, FreeKick, Corner]
X = np.hstack([np.ones((m, 1)), X])

# ---------------------------------------------------------------------
# 3. Compute theta using the Normal Equation:
#    theta = (X^T * X)^(-1) * X^T * y
# ---------------------------------------------------------------------
# Note: In practice, you may want to use a pseudo-inverse for numerical stability.
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# ---------------------------------------------------------------------
# 4. Make predictions and compute r^2 score
# ---------------------------------------------------------------------
predictions = X @ theta
r2 = r2_score(y, predictions)


# question 4 results
print("{:<12} {:<14} {:<40} {:<8}".format(
    "# iterations",
    "Cost Function",
    "Optimal Values of Theta",
    "r2_score"
))
print("-" * 80)

# Data rows
for n in [1, 10, 100, 1000]:
    cost, theta_vals, r2, _ = results[n]
    cost_str = f"{cost:.3f}"
    theta_str = ", ".join([f"{theta:.3f}" for theta in theta_vals])
    r2_str = f"{r2:.3f}"
    print(f"n = {n:<10} {cost_str:<14} {theta_str:<40} {r2_str:<8}")

# -----------------------------
# Determine the player with the maximum predicted value after 1000 iterations
# -----------------------------
_, theta_1000, _, preds_1000 = results[1000]
max_index = np.argmax(preds_1000)
player_max = df.iloc[max_index]['Player']
print(f"\nWho has a maximum predicted value after 1000 iterations? {player_max}")


# question 5 results
print("Optimal values of theta (rounded to 3 decimals):")
for i, val in enumerate(theta):
    print(f"theta{i} = {val:.3f}")

print(f"\nr2-score value (rounded to 3 decimals): {r2:.3f}")