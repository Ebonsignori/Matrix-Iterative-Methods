import math
from decimal import Decimal
import numpy as np

default_max_iterations = 999


def jacobi(a, b, n=None, x=None, delta=None, actual=np.array([]), max_iterations=default_max_iterations):
    """
    Solves the equation Ax=b via the Jacobi iterative method.

    If x matrix is passed in, it is used as an initial guess, otherwise an all-zero matrix is used

    If N is passed in, the function performs N iterations of the jacobi method

    If delta and actual are passed in (both must be passed if one is passed), iterations continue until error
     of the norm (calculated using actual) is less than delta.

    :return: [Result Matrix, Number of iterations taken]
    """
    # Make sure that both delta and actual are passed in
    if (delta and not actual.any()) or (actual.any() and not delta):
        raise SyntaxError("Must pass in both delta and actual if one is passed in")
    # Make sure that only N or delta is passed in
    if delta and n:
        raise SyntaxError("Can only pass delta or N option")

    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(a[0]))

    # Create a vector of the diagonal elements of A and subtract them from A
    D = np.diag(a)
    R = a - np.diagflat(D)

    # Iterate for N times if N is passed in
    if n:
        for i in range(n):
            x = (b - np.dot(R, x)) / D
    # Iterate until error is found or max_iterations is exceeded if delta and actual are passed in
    elif delta and actual.any():
        n = 0
        actual_norm = np.linalg.norm(actual)
        while True:
            x = (b - np.dot(R, x)) / D
            x_norm = np.linalg.norm(x)
            n += 1
            # Compare norms of actual matrix with Jacobian-calculated matrix and if difference is within error, return
            # the number of iterations it took to get within the error
            if abs(Decimal(actual_norm) - Decimal(x_norm)) <= delta or n >= max_iterations:
                break
    # If neither N or delta was passed in
    else:
        raise SyntaxError("Must pass in either N or delta options to function")

    # Return the result and the number of iterations taken to find it
    return [x, n]


def gauss_seidel(a, b, n=None, x=None, delta=None, actual=np.array([]), max_iterations=default_max_iterations):
    """
    Solves the equation Ax=b via the Gauss-Seidel iterative method.

    The same optional parameter rules from the jacobi method apply here

    :return: [Result Matrix, Number of iterations taken]
    """
    # Make sure that both delta and actual are passed in
    if (delta and not actual.any()) or (actual.any() and not delta):
        raise SyntaxError("Must pass in both delta and actual if one is passed in")
    # Make sure that only N or delta is passed in
    if delta and n:
        raise SyntaxError("Can only pass delta or N option")

    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(a[0]))

    # Iterate for N times if N is passed in
    if n:
        L = np.tril(a)
        U = a - L
        for i in range(n):
            x = np.dot(np.linalg.inv(L), b - np.dot(U, x))

    # Iterate until error is found or max_iterations is exceeded if delta and actual are passed in
    elif delta and actual.any():
        n = 0
        actual_norm = np.linalg.norm(actual)
        L = np.tril(a)
        U = a - L

        while True:
            x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
            x_norm = np.linalg.norm(x)
            n += 1
            # Compare norms of actual matrix with Jacobian-calculated matrix and if difference is within error, return
            # the number of iterations it took to get within the error
            if abs(Decimal(actual_norm) - Decimal(x_norm)) <= delta or n >= max_iterations:
                break
    # If neither N or delta was passed in
    else:
        raise SyntaxError("Must pass in either N or delta options to function")

    # Return the result and the number of iterations taken to find it
    return [x, n]


def successive_over_relaxation(a, b, w=1, n=None, x=None, delta=None, actual=np.array([]), max_iterations=default_max_iterations):
    """
    Solves the equation Ax=b via the Successive Over Relaxation iterative method.

    If w is not passed in (w = 1), then this method is equivalent to the gauss_seidel method

    The same optional parameter rules from the jacobi method apply here

    :return: [Result Matrix, Number of iterations taken]
    """
    # Make sure that both delta and actual are passed in
    if (delta and not actual.any()) or (actual.any() and not delta):
        raise SyntaxError("Must pass in both delta and actual if one is passed in")
    # Make sure that only N or delta is passed in
    if delta and n:
        raise SyntaxError("Can only pass delta or N option")

    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(a[0]))

    # Iterate for N times if N is passed in
    if n:
        L = np.tril(a)
        U = a - (w * L)
        for i in range(n):
            x = np.dot(np.linalg.inv(L), b - np.dot(U, x))

    # Iterate until error is found or max_iterations is exceeded if delta and actual are passed in
    elif delta and actual.any():
        n = 0
        actual_norm = np.linalg.norm(actual)
        L = np.tril(a) * w
        U = a - L

        while True:
            x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
            x_norm = np.linalg.norm(x)
            n += 1
            # Compare norms of actual matrix with Jacobian-calculated matrix and if difference is within error, return
            # the number of iterations it took to get within the error
            if abs(Decimal(actual_norm) - Decimal(x_norm)) <= delta or n >= max_iterations:
                break
    # If neither N or delta was passed in
    else:
        raise SyntaxError("Must pass in either N or delta options to function")

    # Return the result and the number of iterations taken to find it
    return [x, n]


def steepest_descent(a, b, n=None, x=None, delta=None, actual=np.array([]), max_iterations=default_max_iterations):
    # Make sure that both delta and actual are passed in
    if (delta and not actual.any()) or (actual.any() and not delta):
        raise SyntaxError("Must pass in both delta and actual if one is passed in")
    # Make sure that only N or delta is passed in
    if delta and n:
        raise SyntaxError("Can only pass delta or N option")
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(a[0]))

    # Iterate for N times if N is passed in
    if n:
        for i in range(n):
            r = b - np.dot(a, x)  # Residual
            w = (np.dot(np.transpose(r), r)) / np.dot(np.dot(np.transpose(r), a), r)  # Weight
            x = x + (w * r)  # approximated solution at current iteration

    # Iterate until error is found or max_iterations is exceeded if delta and actual are passed in
    elif delta and actual.any():
        n = 0
        actual_norm = np.linalg.norm(actual)
        while True:
            r = b - np.dot(a, x)  # Residual
            w = (np.dot(np.transpose(r), r)) / np.dot(np.dot(np.transpose(r), a), r)  # Weight
            x = x + (w * r)  # approximated solution at current iteration

            x_norm = np.linalg.norm(x)
            n += 1
            # Compare norms of actual matrix with steepest-descent calculated matrix and if difference is within error,
            # return the number of iterations it took to get within the error
            if abs(Decimal(actual_norm) - Decimal(x_norm)) <= delta or n >= max_iterations:
                break
    # If neither N or delta was passed in
    else:
        raise SyntaxError("Must pass in either N or delta options to function")

    # Return the result and the number of iterations taken to find it
    return [x, n]


def conjugate_gradient(a, b, n=None, x=None, delta=None, actual=np.array([]), max_iterations=default_max_iterations):
    # Make sure that both delta and actual are passed in
    if (delta and not actual.any()) or (actual.any() and not delta):
        raise SyntaxError("Must pass in both delta and actual if one is passed in")
    # Make sure that only N or delta is passed in
    if delta and n:
        raise SyntaxError("Can only pass delta or N option")
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(a[0]))

    r = b - np.dot(a, x)  # Initial Residual
    p = r
    rs_old = np.dot(r.T, r)

    # Iterate for N times if N is passed in
    if n:
        for i in range(n):
            Ap = np.dot(a, p)
            w = rs_old / np.dot(p.T, Ap)
            x = x + np.dot(w, p)  # approximated solution at current iteration

            r = r - np.dot(w, Ap)
            rs_new = np.dot(r.T, r)
            if math.sqrt(rs_new) < delta:
                break

            p = r + np.dot((rs_new / rs_old), p)
            rs_old = rs_new

    # Iterate until error is found or max_iterations is exceeded if delta and actual are passed in
    elif delta and actual.any():
        n = 0
        while True:
            Ap = np.dot(a, p)
            w = rs_old / np.dot(p.T, Ap)
            x = x + np.dot(w, p)  # approximated solution at current iteration

            r = r - np.dot(w, Ap)
            rs_new = np.dot(r.T, r)

            n += 1
            if math.sqrt(rs_new) < delta:
                break

            p = r + np.dot((rs_new / rs_old), p)
            rs_old = rs_new
    # If neither N or delta was passed in
    else:
        raise SyntaxError("Must pass in either N or delta options to function")

    # Return the result and the number of iterations taken to find it
    return [x, n]