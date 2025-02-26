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
# =============

# перепроверь 9 и 10, я фотки кинул в папку. таких ответов как у тя нету тут

# =============

#Question 1
# 1)Mean (SRS): 50.60
# 2)Standard Error (SRS): 6.89
# 3) Confidence Interval Upper (SRS): 64.66
#  Confidence Interval Lower (SRS): 36.54
# 1)Mean (Clustered): 50.60
# 2)Standard Error (Clustered): 7.16
# 3)d-value: 1.04
# 4)d-squared: 1.08
# 5)roh: 0.01
# 6)N_eff: 15.85

#Question 2

# Cost Function	Optimal Theta
# 10	5766668	[0, -54, 839, -47, -31, 1001, 1080, 436, 283]
# 100	5579355	[0, 50, 230, 125, 192, 1244, 1897, 18, 63]
# 1000	5537726	[0, -17, -520, 119, 228, 1231, 2651, -64, 200]

#question 3
# Results table:
# N=100, alpha=0.1, lambda=0.1: Cost=0.28, Max|theta|=1.61
# N=1000, alpha=0.2, lambda=1.0: Cost=0.16, Max|theta|=4.59
# N=10000, alpha=0.3, lambda=10.0: Cost=0.33, Max|theta|=2.02

# Predictions for first 10 rows: [1 1 0 1 0 1 0 1 0 1]
# Number of 1s in the first 10 predictions = 6 Answer:6 

#Question 4 
# a4 = [0.991, 0.009] (округлено до 3 знаков)
# a3.min() = -0.987 (округлено до 3 знаков)
# W4.max() = -1.09 (округлено до 2 знаков)
# W3.min() = -2.37 (округлено до 2 знаков)
# Loss after 10000 epochs = 0.009
# General Conclusion = "NN predicts image of dog" (нейросеть предсказала изображение собаки)

#Question 5
# Accuracy: 0.992
# F1-score (class 0): 0.993
# F1-score (class 1): 0.978
# F1-score (class 2): 0.993
#Question 6
# A: User Roles
# B: User Requirements
# C: Security Requirements
# D: Operational Constraints
# E: User Management
# F: Book Management

# #Question 7 
# | Age Group |   n  | Proportion |   SE   | SE Adjusted |     95% CI      | 95% CI Adjusted | Design Effect |
# |-----------|------|------------|--------|-------------|-----------------|-----------------|---------------|
# |   25-34   |  384 |     0.256  |  0.022 |     0.024   | (0.212, 0.300)  | (0.210, 0.302)  |      1.125    |
# |   35-44   |  336 |     0.224  |  0.023 |     0.024   | (0.179, 0.269)  | (0.177, 0.271)  |      1.120    |

#Question 8 

# Variable Name	Naming Convention
# intEmployeeID	Hungarian
# employeeID	Pascal
# setEmployeeID	Pascal
# strDepartmentName	Hungarian
# strCEO	Hungarian
# strCTO	Hungarian
# AddDepartment	Pascal

#Question 9 
# Test Case	Test Type
# A (test_end_to_end_order_process)	End-to-End Test
# B (test__with_stub_cart)	Unit Test with Stub
# def test__with_stub_cart	Stub-based Test
# def calculate_total	Unit Test

#Question 10

# A	✅ Correct factorial implementation.	OK Case
# B	❌ Indentation error in Python (if is not indented).	Bug Case
# C	❌ The loop runs from 0 to n-1, making multiplication always 0 since 0 * anything = 0.	Bug Case
# D	❌ return result_val is incorrect; result_val is not defined. Should be return result.	Bug Case
# E	❌ Binary search logic is incorrect: It moves left = mid + 1 when arr[mid] > target, which is wrong. Should be right = mid - 1.	Bug Case
# F	❌ Condition while (left < right) should be while (left <= right), otherwise, it might skip the correct index.	Bug Case
# G	✅ Correct binary search implementation.	OK Case
# H	✅ Correct binary search implementation. Same as G.	OK Case



