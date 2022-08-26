import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import use
use('TkAgg')


def solve_2D_heat_equation(T_x_y_t0, T_x0_y_t, T_xf_y_t, T_x_y0_t, T_x_yf_t,
                           x_0, x_f, delta_x, y_0, y_f, delta_y, t_0, t_f, delta_t,
                           heat_flux_x0=False, heat_flux_xf=False, heat_flux_y0=False, heat_flux_yf=False,
                           alpha=1, method="Explicit", max_saves=150):
    """
    Solves the 2D heat equation.
    All boundary condition functions must be vectorized
    (i.e. they should be capable of taking and returning numpy arrays).

    :param T_x_y_t0: Temperature at t=t_0, as a function of x and y.
    :param T_x0_y_t: Temperature or temperature derivative at x=x_0, as a function of y and time.
    :param T_xf_y_t: Temperature or temperature derivative at x=x_f, as a function of y and time.
    :param T_x_y0_t: Temperature or temperature derivative at y=y_0, as a function of x and time.
    :param T_x_yf_t: Temperature or temperature derivative at y=y_f, as a function of x and time.
    :param x_0: The starting x coordinate. Typically 0.
    :param x_f: The ending x coordinate.
    :param delta_x: The distance between grid points for x (between x_0 and x_f).
    :param y_0: The starting y coordinate. Typically 0.
    :param y_f: The ending y coordinate.
    :param delta_y: The distance between grid points for y (between y_0 and y_f).
    :param t_0: The starting time, typically 0.
    :param t_f: The ending time.
    :param delta_t: The time step to take.
    :param heat_flux_x0: If true, then the boundary condition for x=x_0 specifies the temperature gradient
                         (i.e. heat flux) instead of the temperature.
    :param heat_flux_xf: If true, then the boundary condition for x=x_f specifies the temperature gradient
                         (i.e. heat flux) instead of the temperature.
    :param heat_flux_y0: If true, then the boundary condition for y=y_0 specifies the temperature gradient
                         (i.e. heat flux) instead of the temperature.
    :param heat_flux_yf: If true, then the boundary condition for y=y_f specifies the temperature gradient
                         (i.e. heat flux) instead of the temperature.
    :param alpha: The thermal diffusivity.
    :param method: The method to use. Must be "Explicit".
    :param max_saves: The maximum number of time steps to save, for ram saving purposes.
    :return: The matrix of temperature values, where the indices corresponds to the x, y and t values respectively.
    """
    if method is not "Explicit":
        raise Exception("Not implemented!")

    if 1 - alpha * delta_t / delta_x ** 2 - alpha * delta_t / delta_y ** 2 <= 0:
        raise Exception("Unstable combination of delta_t, delta_x and delta_y!")

    x_values, delta_x = np.linspace(x_0, x_f, int(np.ceil((x_f - x_0) / delta_x)), retstep=True)
    y_values, delta_y = np.linspace(y_0, y_f, int(np.ceil((y_f - y_0) / delta_y)), retstep=True)
    t = t_0
    T = T_x_y_t0(x_values, y_values)

    t_values = np.array([t])
    T_values = np.copy(T)[:, :, np.newaxis]

    count = 0
    save_frequency = np.ceil((t_f - t_0) / delta_t / max_saves)
    while t < t_f:
        # Calculate the spatial Laplacian
        T_xx = (T[2:, 1:-1] + T[:-2, 1:-1] - 2 * T[1:-1, 1:-1]) / delta_x ** 2
        T_yy = (T[1:-1, 2:] + T[1:-1, :-2] - 2 * T[1:-1, 1:-1]) / delta_y ** 2

        # perform step
        t += delta_t
        T[1:-1, 1:-1] += alpha * (T_xx + T_yy) * delta_t

        # apply left boundary condition
        if heat_flux_x0:
            T[0, :] = T[1, :] - delta_x * T_x0_y_t(y_values, t)
        else:
            T[0, :] = T_x0_y_t(y_values, t)

        # apply right boundary condition
        if heat_flux_xf:
            T[-1, :] = T[-2, :] + delta_x * T_xf_y_t(y_values, t)
        else:
            T[-1, :] = T_xf_y_t(y_values, t)

        # apply bottom boundary condition
        if heat_flux_y0:
            T[:, 0] = T[:, 1] - delta_y * T_x_y0_t(x_values, t)
        else:
            T[:, 0] = T_x_y0_t(x_values, t)

        # apply top boundary condition
        if heat_flux_yf:
            T[:, -1] = T[:, -2] + delta_y * T_x_yf_t(x_values, t)
        else:
            T[:, -1] = T_x_yf_t(x_values, t)

        count += 1
        if count % save_frequency == 0:
            t_values = np.append(t_values, t)
            T_values = np.concatenate((T_values, T[:, :, np.newaxis]), axis=-1)

            print((t - t_0) / (t_f - t_0) * 100, "%")

    return T_values


def graph_2b(x_0=0, x_f=1, delta_x=0.01, y_0=0, y_f=1, delta_y=0.01, t_0=0, t_f=0.1, delta_t=1e-5, interpolation=None):
    T1_values = solve_2D_heat_equation(lambda x, y: np.outer(np.sin(pi * x), np.cos(4 * pi * y)),
                                       lambda y, t: np.zeros(y.shape), lambda y, t: np.zeros(y.shape),
                                       lambda x, t: np.sin(pi * x), lambda x, t: np.sin(pi * x),
                                       x_0, x_f, delta_x, y_0, y_f, delta_y, t_0, t_f, delta_t)

    graph_animation(T1_values, x_0, x_f, y_0, y_f, t_0, t_f, '3b_graph1')

    T2_values = solve_2D_heat_equation(lambda x, y: np.outer(np.sin(pi * x), np.sin(4 * pi * y)),
                                       lambda y, t: np.zeros(y.shape), lambda y, t: np.ones(y.shape),
                                       lambda x, t: np.ones(x.shape), lambda x, t: -np.ones(x.shape),
                                       x_0, x_f, delta_x, y_0, y_f, delta_y, t_0, t_f, delta_t,
                                       heat_flux_x0=True, heat_flux_xf=True, heat_flux_y0=False, heat_flux_yf=False)
    graph_animation(T2_values, x_0, x_f, y_0, y_f, t_0, t_f, '3b_graph2')


def graph_animation(T, x_0, x_f, y_0, y_f, t_0, t_f, file_name, interpolation=None):
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(T[:, :, 0].T, interpolation=interpolation, origin='lower', extent=[x_0, x_f, y_0, y_f],
                   aspect=(y_f - y_0) / (x_f - x_0), animated=True)
    plt.colorbar(im, ax=ax)
    plt.title('t=' + str(t_0) + 's')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    def animate(i):
        im.set_data(T[:, :, i].T)
        plt.title('t=' + ('%.4f' % (t_0 + (t_f - t_0) * i / T.shape[2])) + 's')
        return im,

    anim = animation.FuncAnimation(fig, animate, frames=T.shape[2], interval=33, repeat_delay=1000)
    plt.show()
    # anim.save('Graphs/' + file_name + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


if __name__ == '__main__':
    graph_2b(delta_x=0.01, delta_y=0.01, delta_t=1e-5, interpolation='bicubic')
