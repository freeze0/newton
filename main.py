import numpy as np


def f(x):
    return 3*x[0]**2 + x[1]**2 - x[0]*x[1] + x[0]


def gradient(f, x):
    grad = np.zeros(2)
    for i in range(2):
        eps = 1e-8
        delta = np.zeros(2)
        delta[i] = eps
        f_plus = f(x + delta)
        f_minus = f(x - delta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def newton(eps1, eps2, x0, f, hesse, maxiter=100):
    flag = False
    k = 0
    x_k = x0
    while not flag:
        grad = gradient(f, x_k)
        norma = np.linalg.norm(grad)
        if norma <= eps1:
            x_otv = x_k
            return x_otv
        else:
            eigvals = np.linalg.eigvals(hesse(x_k))
            if np.all(eigvals > 0):
                d_k = -np.linalg.solve(hesse(x_k), grad)
            else:
                d_k = -grad
            t_k = 1
            while f(x_k + t_k*d_k) > f(x_k) + eps2*t_k*np.dot(grad, d_k):
                t_k /= 2
            x_k_next = x_k + t_k*d_k
            if np.linalg.norm(x_k_next - x_k) < eps1 or np.linalg.norm(f(x_k_next) - f(x_k)) < eps1 or k == maxiter:
                flag = True
            else:
                x_k = x_k_next
                k += 1
    return x_k_next


eps1 = 0.1
eps2 = 0.15
x0 = np.array([1.5, 1.5])
hesse = lambda x: np.array([[6, -1], [-1, 2]])
result = newton(eps1, eps2, x0, f, hesse)
print(result)
