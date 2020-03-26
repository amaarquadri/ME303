import numpy as np


def solve_ivp(x_prime_func, x_0, t_0, t_f, delta_t=None, args=(), method='RK4',
              r_Tol=None, show_progress=True, memory_limit=10_000):
    """
    Computes numerical solution to a system of first order ODEs.

    :param x_prime_func: A function to calculate the x' as a function of t, x (in that order)
    :param x_0: The initial vector value for x
    :param t_0: The initial value for t
    :param t_f: The final value for t, up to which integration will be done.
    :param delta_t: The time increment to use. If None, then a default will be chosen
                    such that there will be a total of 1 million time steps.
    :param args: Any extra arguments that should be passed to the x_prime_func.
    :param method: The numerical method to use. Must be either 'Euler', 'RK2' or 'RK4'. Defaults to 'RK4'.
    :param r_Tol: If specified, then if r goes below r_Tol then the velocity is inverted.
    :param show_progress: If True, then progress updates will be printed.
    :param memory_limit: Specifies the maximum number of data points that will be stored,
                         for memory conservation purposes.
    :return: The tuple t, x, min_t_values
    """
    if delta_t is None:
        # if not specified, split into 1 million time steps
        delta_t = (t_f - t_0) / 10 ** 6

    t_results = np.array([t_0])
    x_results = np.copy(x_0)

    t = t_0
    x = x_0
    min_t_values = np.array([])

    count = 0
    save_frequency = np.maximum(1, np.ceil((t_f - t_0) / delta_t / memory_limit))
    while t < t_f:
        t += delta_t
        x += delta_t * step(x_prime_func, t, x, delta_t, args, method)

        if r_Tol is not None and x[0] < r_Tol:
            if count < 100:
                break  # if we get 2 bounces in rapid succession then halt
            min_t_values = np.append(min_t_values, t)
            x[1] *= -1  # Flip velocity
            count = 0
            continue  # Restart loop

        count += 1
        if count % save_frequency == 0:
            t_results = np.append(t_results, t)
            x_results = np.column_stack((x_results, x))
            if show_progress:
                print((t - t_0) / (t_f - t_0) * 100, "%")

    return t_results, x_results, min_t_values


def step(x_prime_func, t, x, delta_t, args, method):
    k_0 = x_prime_func(t, x, *args)
    if method is 'Euler':
        return k_0

    k_1 = x_prime_func(t + delta_t / 2, x + k_0 * delta_t / 2, *args)
    if method is 'RK2':
        return k_1

    k_2 = x_prime_func(t + delta_t / 2, x + k_1 * delta_t / 2, *args)
    k_3 = x_prime_func(t + delta_t, x + k_2 * delta_t, *args)
    return (k_0 + 2 * k_1 + 2 * k_2 + k_3) / 6
