import casadi as ca

# Create an Opti object
opti = ca.Opti()

# Define decision variables
x = opti.variable(2, 3)
y = opti.variable()

# Define objective function
objective = (x - 1)**2 + (y - 2)**2

# Add objective to minimize
opti.minimize(objective)

# Define constraints
constraint1 = x + y >= 1
constraint2 = x - y <= 2

# Add constraints
opti.subject_to(constraint1)
opti.subject_to(constraint2)

# Choose solver and solve the optimization problem
opti.solver('ipopt')
sol = opti.solve()

# Print solution
print("Optimal solution:")
print("x =", sol.value(x))
print("y =", sol.value(y))
