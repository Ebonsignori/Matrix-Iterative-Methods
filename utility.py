x_variables = [
    "F1", "F2", "F3", "f1", "f2", "f3", "f4", "f5"
]


def print_results(matrix, variables=x_variables):
    for i in range(len(matrix)):
        print("  {} = {}".format(variables[i], matrix[i]))
