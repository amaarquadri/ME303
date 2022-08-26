import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


# noinspection PyUnboundLocalVariable
def solve_1D_heat_equation(T_x_t0, T_x0_t, T_xf_t, x_0, x_f, delta_x, t_0, t_f, delta_t,
                           heat_flux_x0=False, heat_flux_xf=False, alpha=1, method="Explicit", max_saves=500):
    """
    Solves the 1D heat equation.
    All boundary condition functions must be vectorized
    (i.e. they should be capable of taking and returning numpy arrays).

    :param T_x_t0: Temperature at t=t_0, as a function of x.
    :param T_x0_t: Temperature or temperature derivative at x=x_0, as a function of time.
    :param T_xf_t: Temperature or temperature derivative at x=x_f, as a function of time.
    :param x_0: The starting x coordinate. Typically 0.
    :param x_f: The ending x coordinate.
    :param delta_x: The distance between grid points for x (between x_0 and x_f).
    :param t_0: The starting time, typically 0.
    :param t_f: The ending time.
    :param delta_t: The time step to take.
    :param heat_flux_x0: If true, then the boundary condition for x=x_0 specifies the temperature gradient
                         (i.e. heat flux) instead of the temperature.
    :param heat_flux_xf: If true, then the boundary condition for x=x_f specifies the temperature gradient
                         (i.e. heat flux) instead of the temperature.
    :param alpha: The thermal diffusivity.
    :param method: The method to use. Must be either "Explicit" or "Crank-Nicolson".
    :param max_saves: The maximum number of time steps to save, for ram saving purposes.
    :return: The matrix of temperature values, where the indices corresponds to the x and t values respectively.
    """
    if 1 - 2 * alpha * delta_t / delta_x ** 2 <= 0:
        raise Exception("Unstable combination of delta_t and delta_x!")

    x_values, delta_x = np.linspace(x_0, x_f, int(np.ceil((x_f - x_0) / delta_x)), retstep=True)
    t = t_0
    T = T_x_t0(x_values)

    t_values = np.array([t])
    T_values = np.copy(T)

    if method == 'Crank-Nicolson':
        r = alpha * delta_t / (2 * delta_x ** 2)
        # Create matrix with 1 + 2r as all diagonal elements, except first and last which are 1
        a = (1 + 2 * r) * np.eye(len(T))
        a[0, 0] = a[-1, -1] = 1

        # Set all off-diagonal elements to -r
        for k in range(1, len(T) - 1):
            a[k, k - 1] = a[k, k + 1] = -r

        # Set the off diagonal elements to 0 if direct boundary conditions are given,
        # or -1 if heat flux conditions are given
        a[0, 1] = -1 if heat_flux_x0 else 0
        a[-1, -2] = -1 if heat_flux_xf else 0

    count = 0
    save_frequency = np.ceil((t_f - t_0) / delta_t / max_saves)
    while t < t_f:
        # crank nicolson
        if method == 'Crank-Nicolson':
            b = np.concatenate(([- delta_x * T_x0_t(t) if heat_flux_x0 else T_x0_t(t)],
                                r * T[:-2] + (1 - 2 * r) * T[1:-1] + r * T[2:],
                                [delta_x * T_xf_t(t) if heat_flux_xf else T_xf_t(t)]))
            t += delta_t
            T = solve_tridiagonal(np.copy(a), b)
        elif method == 'Explicit':
            # calculate the spatial second derivative
            T_xx = (T[:-2] - 2 * T[1:-1] + T[2:]) / delta_x ** 2

            # perform step
            t += delta_t
            T[1:-1] += alpha * T_xx * delta_t

            # apply starting boundary condition
            if heat_flux_x0:
                T[0] = T[1] - delta_x * T_x0_t(t)
            else:
                T[0] = T_x0_t(t)

            # apply ending boundary condition
            if heat_flux_xf:
                T[-1] = T[-2] + delta_x * T_xf_t(t)
            else:
                T[-1] = T_xf_t(t)

        count += 1
        if count % save_frequency == 0:
            t_values = np.append(t_values, t)
            T_values = np.column_stack((T_values, T))

            print((t - t_0) / (t_f - t_0) * 100, "%")

    return T_values


def solve_tridiagonal(a, b):
    """
    Solves the system of equations ax=b, where a is a tridiagonal matrix.
    This function essentially just implements a special case of gaussian elimination.
    """
    if a.shape[0] != a.shape[1] or a.shape[0] != len(b):
        raise Exception('a must be square with the same size as b!')

    # cancel out lowest diagonal by rippling through
    for k in range(0, len(b) - 1):
        factor = a[k + 1, k] / a[k, k]
        a[k + 1, k:k + 3] -= factor * a[k, k:k + 3]
        b[k + 1] -= factor * b[k]

    # cancel out upper diagonal by rippling through
    for k in range(len(b) - 1, 1, -1):
        factor = a[k - 1, k] / a[k, k]
        # Divide
        # Don't actually have to do division for matrix, since all the a[k -1, k] become 0 and will not be used again
        a[k - 1, k] -= factor * a[k, k]
        b[k - 1] -= factor * b[k]

    # a is now a diagonal matrix, so we can simply divide to get the desired solution vector
    return b / np.diagonal(a)


def graph_1a(x_0=0, x_f=2, delta_x=0.01, t_0=0, t_f=0.1, delta_t=1e-6, interpolation=None):
    for method in ["Explicit", "Crank-Nicolson"]:
        T1_values = solve_1D_heat_equation(lambda x: np.sin(4 * pi * x), lambda t: 0, lambda t: 0,
                                           x_0, x_f, delta_x, t_0, t_f, delta_t, method=method)

        plt.imshow(T1_values, interpolation=interpolation, origin='lower', extent=[t_0, t_f, x_0, x_f],
                   aspect=(t_f - t_0) / (x_f - x_0))
        plt.colorbar()
        plt.title('Temperature Versus Time and Space (Q1, ' + method + ' method)')
        plt.xlabel('t (s)')
        plt.ylabel('X (m)')
        plt.savefig('Graphs/1a_graph1_' + method + '.png')
        plt.clf()

        T1_analytical = analytical_solution_1(np.linspace(x_0, x_f, T1_values.shape[0]),
                                              np.linspace(t_0, t_f, T1_values.shape[1]))
        plt.imshow(T1_values - T1_analytical, interpolation=interpolation, origin='lower', extent=[t_0, t_f, x_0, x_f],
                   aspect=(t_f - t_0) / (x_f - x_0))
        plt.colorbar()
        plt.title('Residual Versus Time and Space (Q1, ' + method + ' method)')
        plt.xlabel('t (s)')
        plt.ylabel('X (m)')
        plt.savefig('Graphs/1a_graph1_' + method + '_error.png')
        plt.clf()

        T2_values = solve_1D_heat_equation(lambda x: np.sin(2 * pi * x), lambda t: 0, lambda t: np.sin(pi * t),
                                           x_0, x_f, delta_x, t_0, t_f, delta_t, method=method)
        plt.imshow(T2_values, interpolation=interpolation, origin='lower', extent=[t_0, t_f, x_0, x_f],
                   aspect=(t_f - t_0) / (x_f - x_0))
        plt.colorbar()
        plt.title('Temperature Versus Time and Space (Q2, ' + method + ' method)')
        plt.xlabel('t (s)')
        plt.ylabel('X (m)')
        plt.savefig('Graphs/1a_graph2_' + method + '.png')
        plt.clf()

        T3_values = solve_1D_heat_equation(lambda x: np.sin(2 * pi * x), lambda t: 0, lambda t: 0,
                                           x_0, x_f, delta_x, t_0, t_f, delta_t,
                                           heat_flux_x0=True, heat_flux_xf=True, method=method)
        plt.imshow(T3_values, interpolation=interpolation, origin='lower', extent=[t_0, t_f, x_0, x_f],
                   aspect=(t_f - t_0) / (x_f - x_0))
        plt.colorbar()
        plt.title('Temperature Versus Time and Space (Q3, ' + method + ' method)')
        plt.xlabel('t (s)')
        plt.ylabel('X (m)')
        plt.savefig('Graphs/1a_graph3_' + method + '.png')
        plt.clf()

        T4_values = solve_1D_heat_equation(lambda x: np.sin(2 * pi * x), lambda t: 0, lambda t: 1,
                                           x_0, x_f, delta_x, t_0, t_f, delta_t,
                                           heat_flux_x0=False, heat_flux_xf=True, method=method)
        plt.imshow(T4_values, interpolation=interpolation, origin='lower', extent=[t_0, t_f, x_0, x_f],
                   aspect=(t_f - t_0) / (x_f - x_0))
        plt.colorbar()
        plt.title('Temperature Versus Time and Space (Q4, ' + method + ' method)')
        plt.xlabel('t (s)')
        plt.ylabel('X (m)')
        plt.savefig('Graphs/1a_graph4_' + method + '.png')
        plt.clf()

        T4_analytical = analytical_solution_4(np.linspace(x_0, x_f, T4_values.shape[0]),
                                              np.linspace(t_0, t_f, T4_values.shape[1]))
        plt.imshow(T4_values - T4_analytical, interpolation=interpolation, origin='lower', extent=[t_0, t_f, x_0, x_f],
                   aspect=(t_f - t_0) / (x_f - x_0))
        plt.colorbar()
        plt.title('Residual Versus Time and Space (Q4, ' + method + ' method)')
        plt.xlabel('t (s)')
        plt.ylabel('X (m)')
        plt.savefig('Graphs/1a_graph4_' + method + '_error.png')
        plt.clf()


def analytical_solution_1(x, t):
    return np.outer(np.sin(4 * pi * x), np.exp(-16 * pi ** 2 * t))


def analytical_solution_4(x, t, n_max=100):
    result = np.zeros((len(x), len(t)))
    result += x[:, np.newaxis]
    for n in range(1, n_max):
        result += 16 * (2 / (np.pi * (-4 * n ** 2 + 4 * n + 63)) + 1 / (np.pi - 2 * np.pi * n) ** 2) * (-1) ** n * \
                  np.outer(np.sin((2 * n - 1) * np.pi * x / 4), np.exp(-(2 * n - 1) ** 2 * np.pi ** 2 * t / 16))
    return result


if __name__ == '__main__':
    graph_1a(delta_x=0.01, delta_t=1e-6, interpolation='bicubic')
