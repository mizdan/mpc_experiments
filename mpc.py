import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Define system matrices
A = 1.0
B = 1.0
C = 1

# Define cost matrices and prediction horizon
Q = 10  # State cost
R = 1  # Control input cost
N = 10  # Prediction horizon

# Initial state and reference trajectory
x0 = 0
#reference_trajectory = np.ones(N+1)

reference_trajectory = np.ones(40)
for xx in range(10,30):
    reference_trajectory[xx]=0

# Control input constraints
u_min = -0.5
u_max = 0.5

# Simulation parameters
num_steps = 30
x_sim = np.zeros(num_steps+1)
u_sim = np.zeros(num_steps)

# Define variables for optimization
x = cp.Variable(N+1)
u = cp.Variable(N)

for k in range(num_steps):


    # Define the cost function
    cost = 0
    for i in range(N):
        cost += Q * (x[i] - reference_trajectory[k+i])**2 + R * u[i]**2

    # Define constraints
    constraints = [x[0] == x_sim[k]]
    for i in range(N):
        constraints += [x[i+1] == A * x[i] + B * u[i]]
        constraints += [u[i] >= u_min, u[i] <= u_max]  # Control input constraints

    # Create and solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Get the optimal control input
    optimal_u = u.value[0]

    # Simulate the system with the optimal control input
    x_sim[k+1] = A * x_sim[k] + B * optimal_u
    u_sim[k] = optimal_u

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(range(num_steps+1), x_sim, label='System Output')
plt.plot(range(39+1), reference_trajectory, linestyle='dashed', label='Reference Trajectory')
plt.title('System Output and Reference Trajectory')
plt.legend()

plt.subplot(2, 1, 2)
plt.step(range(num_steps), u_sim, label='Control Input')
plt.title('Optimal Control Input')
plt.legend()

plt.tight_layout()
plt.show()
