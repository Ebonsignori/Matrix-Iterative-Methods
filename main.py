import numpy as np
from math import sqrt
from iterative_methods import jacobi, gauss_seidel, successive_over_relaxation
from utility import print_results

A = np.array([
   [-1, 0, 0, sqrt(2)/2, 1, 0, 0, 0],
   [0, -1, 0, sqrt(2)/2, 0, 0, 0, 0],
   [0, 0, -1, 0, 0, 0, 1/2, 0],
   [0, 0, 0, -sqrt(2)/2, 0, -1, -1/2, 0],
   [0, 0, 0, 0, -1, 0, 0, 1],
   [0, 0, 0, 0, 0, 1, 0, 0],
   [0, 0, 0, -sqrt(2)/2, 0, 0, sqrt(3)/2, 0],
   [0, 0, 0, 0, 0, 0, sqrt(3)/2, -1]
])

b = np.array([0, 0, 0, 0, 0, 10000, 0, 0])

# Get the real solution using built in numpy method and solution using 3 iterative methods within 10^-8 error
# x_solved = np.linalg.solve(A, b) # Either this of dot product of A inv b works, but the second is more clear
x_solved = np.dot(np.linalg.inv(A), b)
x_jac = jacobi(A, b, delta=10**-8, actual=x_solved)
x_gauss_seidel = gauss_seidel(A, b, delta=10**-8, actual=x_solved)
x_sor = successive_over_relaxation(A, b, w=1.25, delta=10**-8, actual=x_solved)

# Print the results for each method
print("x [Solved with built in Numpy functions]: ")
print_results(x_solved)
print()  # Newline

print("x [Solved using {} iterations of the Jacobi method]: ".format(x_jac[1]))
print_results(x_jac[0])
print()

print("x [Solved using {} iterations of the Gauss-Seidel method]: ".format(x_gauss_seidel[1]))
print_results(x_gauss_seidel[0])
print()

print("x [Solved using {} iterations of the Successive Over Relation (SOR) method]: ".format(x_sor[1]))
print_results(x_sor[0])
print()
