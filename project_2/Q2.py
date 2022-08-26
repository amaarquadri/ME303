import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

R_QUAIL = 0.032639 / 2  # m
M_QUAIL = 0.01  # kg
R_CHICKEN = 0.0326  # m
M_CHICKEN = 0.0557  # kg
R_OSTRICH = 0.1397 / 2  # m
M_OSTRICH = 1.4  # kg


def solve_1D_spherical_heat_equation(T_r_t0, T_R_t, R, delta_r, t_0, t_f, delta_t,
                                     alpha=1, method="Explicit", max_saves=500):
    """
    Solves the 1D heat equation.
    All boundary condition functions must be vectorized
    (i.e. they should be capable of taking and returning numpy arrays).

    :param T_r_t0: Temperature in degrees celsius at t=t_0, as a function of r.
    :param T_R_t: Temperature in degrees celsius at r=R, as a function of time.
    :param R: The ending r coordinate, aka the total radius in m.
    :param delta_r: The distance between grid points for r (between 0 and R) in m.
    :param t_0: The starting time in seconds, typically 0.
    :param t_f: The ending time in seconds.
    :param delta_t: The time step to take in seconds.
    :param alpha: The thermal diffusivity in m^2/s.
    :param method: The method to use. Must be either "Explicit" or "Crank-Nicolson".
    :param max_saves: The maximum number of time steps to save, for ram saving purposes.
    :return: The matrix of temperature values, where the indices corresponds to the r and t values respectively.
    """
    if method is not "Explicit":
        raise Exception("Not implemented!")

    if 1 - 2 * alpha * delta_t / delta_r ** 2 <= 0:
        raise Exception("Unstable combination of delta_t and delta_x!")

    r_values, delta_r = np.linspace(0, R, int(np.ceil(R / delta_r)), retstep=True)
    t = t_0
    T = T_r_t0(r_values)

    t_values = np.array([t])
    T_values = np.copy(T)

    count = 0
    save_frequency = np.ceil((t_f - t_0) / delta_t / max_saves)
    fully_heated = False
    while t < t_f:
        # calculate the spatial second derivative
        rT = r_values * T
        rT_rr = (rT[:-2] - 2 * rT[1:-1] + rT[2:]) / delta_r ** 2
        rT_rr_over_r = rT_rr / r_values[1:-1]

        # perform step
        t += delta_t
        T[1:-1] += alpha * rT_rr_over_r * delta_t

        # apply center boundary condition
        T[0] += alpha * (T[1] - T[0]) / delta_r ** 2 * delta_t

        # apply ending boundary condition
        T[-1] = T_R_t(t)

        if not fully_heated and T[0] > 80:
            fully_heated = True
            print("Fully heated at t =", t, 'seconds')

        count += 1
        if count % save_frequency == 0:
            t_values = np.append(t_values, t)
            T_values = np.column_stack((T_values, T))

            # print((t - t_0) / (t_f - t_0) * 100, "%, T(0)=", T[0], ', t=', t)

    return T_values


def graph_2(R=46.59E-3, delta_r=0.01, t_0=0, t_f=10, delta_t=1e-3,
            m=47.9E-3, k=0.960, c_p=3320, interpolation=None, file_name='chicken'):
    rho = m / ((4 / 3) * pi * R ** 3)
    alpha = k / (rho * c_p)
    T_values = solve_1D_spherical_heat_equation(lambda r: 25 * np.ones(r.shape), lambda t: 100,
                                                R, delta_r, t_0, t_f, delta_t, alpha=alpha)

    plt.imshow(T_values, interpolation=interpolation, origin='lower', extent=[t_0, t_f, 0, R],
               aspect=(t_f - t_0) / R)
    plt.colorbar()
    plt.title("Temperature Versus Radius and Time (" + file_name + ' egg)')
    plt.xlabel('t (s)')
    plt.ylabel('R (m)')
    plt.savefig('Graphs/2_graph_' + file_name + '.png')
    plt.clf()


if __name__ == '__main__':
    print('Quail:')
    graph_2(R=R_QUAIL, m=M_QUAIL, file_name='quail', t_f=800, interpolation='bicubic')
    print('Chicken:')
    graph_2(R=R_CHICKEN, m=M_CHICKEN, file_name='chicken', t_f=600, interpolation='bicubic')
    print('Ostrich:')
    graph_2(R=R_OSTRICH, m=M_OSTRICH, file_name='ostrich', t_f=4000, interpolation='bicubic')
