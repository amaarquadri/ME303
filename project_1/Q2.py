from project_1.ODEUtils import *
from project_1.Constants import *
import matplotlib.pyplot as plt


# noinspection PyUnusedLocal
def rayleigh_plesset(t, x, P_0):
    r, r_prime = x
    r_prime_prime = ((-P_0 - 4 * mu_water * r_prime / r - 2 * sigma_water / r) / rho_water - (3 / 2) * r_prime ** 2) / r
    return np.array([r_prime, r_prime_prime])


def graph_2b(r_0, r_prime_0, P_0, t_0, t_f, delta_t=1E-5, file_name='2b'):
    t, x, min_t_values = solve_ivp(rayleigh_plesset, np.array([r_0, r_prime_0]), t_0, t_f,
                                   delta_t=delta_t, args=(P_0,), method='RK4', r_Tol=r_0 / 10)
    r = x[0, :]

    plt.plot(t, r)
    plt.xlim(t_0, t_f)
    plt.title("Radius Versus Time for " + str(r_0) + r"m Bubble Collapse with $\Delta$t=" + str(delta_t))
    plt.ylabel('Radius (m)')
    plt.xlabel('Time (s)')
    plt.savefig("Graphs/" + file_name + ".png")
    plt.clf()

    frequency = 1 / np.diff(min_t_values)
    t_average = (min_t_values[1:] + min_t_values[:-1]) / 2
    plt.plot(t_average, frequency)
    plt.title("Frequency Versus Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.savefig("Graphs/" + file_name + "_frequency.png")
    plt.clf()

    plt.plot(t_average, np.log(frequency))
    plt.title("Ln Frequency Versus Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Ln Frequency (Ln Hz)")
    plt.savefig("Graphs/" + file_name + "_ln_frequency.png")
    plt.clf()


def graph_2c(r_0, r_prime_0, P_0, t_0, t_f, delta_t_values):
    t_ref, x_ref, _ = solve_ivp(rayleigh_plesset, np.array([r_0, r_prime_0]), t_0, t_f,
                                delta_t=delta_t_values[-1], args=(P_0,), method='RK4', r_Tol=r_0 / 10)
    r_ref = x_ref[0, :]
    plt.plot(t_ref, r_ref, label=r'Reference: $\Delta$t=' + str(delta_t_values[-1]))

    errors = []
    for delta_t in delta_t_values[:-1]:
        t, x, _ = solve_ivp(rayleigh_plesset, np.array([r_0, r_prime_0]), t_0, t_f,
                            delta_t=delta_t, args=(P_0,), method='RK4', r_Tol=r_0 / 10)
        r = x[0, :]

        errors.append(np.max(np.abs(r - np.interp(t, t_ref, r_ref))))
        plt.plot(t, r, label=r'$\Delta$t=' + str(delta_t))

    plt.xlim(t_0, t_f)
    plt.legend()
    plt.title("Radius Versus Time for " + str(r_0) + r"m Bubble Collapse")
    plt.ylabel('Radius (m)')
    plt.xlabel('Time (s)')
    plt.savefig("Graphs/2c_graphs.png")
    plt.clf()

    plt.plot(np.log(delta_t_values[:-1]), np.log(errors), '-o')
    print("Best fit Line:", np.polyfit(np.log(delta_t_values[:-1]), np.log(errors), 1))  # [1.27497756 9.79446239]
    plt.title(r"Maximum Error Versus $\Delta$t")
    plt.xlabel(r"Ln($\Delta$t) (Ln(s))")
    plt.ylabel(r"Ln($\epsilon_{max}$) (Ln(m))")
    plt.savefig("Graphs/2c_errors.png")
    plt.clf()


def graph_2d(r_0_values, r_prime_0, P_0, t_0, t_f_values, delta_t_values):
    mean_period_values = []
    for r_0, t_f, delta_t in zip(r_0_values, t_f_values, delta_t_values):
        t, x, min_t_values = solve_ivp(rayleigh_plesset, np.array([r_0, r_prime_0]), t_0, t_f,
                                       delta_t=delta_t, args=(P_0,), method='RK4', r_Tol=r_0 / 10)
        r = x[0, :]
        mean_period_values.append(np.mean(np.diff(min_t_values)))

        plt.plot(t, r)
        plt.title("Radius Versus Time for " + str(r_0) + r"m Bubble Collapse")
        plt.ylabel('Radius (m)')
        plt.xlabel('Time (s)')
        plt.savefig("Graphs/2d_r_0=" + str(r_0) + ".png")
        plt.clf()

    print("Best fit Line:", np.polyfit(np.log10(r_0_values), np.log10(mean_period_values), 1))  # 1.27497756, 9.79446239
    plt.plot(np.log10(r_0_values), np.log10(mean_period_values), '-o')
    plt.title("Period Versus Initial Bubble Radius")
    plt.xlabel("Log Base 10 of Initial Bubble Radius (Log(m))")
    plt.ylabel("Log Base 10 of Oscillation Period (Log(s))")
    plt.savefig("Graphs/2d_periods.png")
    plt.clf()


def graph_depth_variations(r_0, r_prime_0, depth_values):
    for depth in depth_values:
        P_0 = rho_water * g * depth
        t, x, min_t_values = solve_ivp(rayleigh_plesset, np.array([r_0, r_prime_0]), 0, 1,
                                       delta_t=1e-5, args=(P_0,), method='RK4', r_Tol=2)
        print(1 / (2 * min_t_values[0]))

        r = x[0, :]
        plt.plot(t, r, label="Depth: " + str(depth) + "m")
    plt.xlim(0, 0.1)
    plt.legend()
    plt.savefig("Graphs/2_depth_variations.png")
    plt.clf()


if __name__ == '__main__':
    # Note R'(0)=0 is assumed throughout
    P_0_value = P_atmosphere + rho_water * g * 1000
    graph_2b(100., 0, P_0_value, 0, 10)

    mu_water *= 10 ** 8
    graph_2b(100., 0, P_0_value, 0, 20, file_name='2b_high_viscosity')
    mu_water /= 10 ** 8

    graph_2c(100., 0, P_0_value, 0, 1, [10 ** -i for i in range(3, 8)])
    graph_2d([100., 10., 1., 0.1], 0, P_0_value, 0,
             [10., 1., 0.1, 0.01], [1E-5, 1E-6, 1E-7, 1E-8])
    # graph_depth_variations(10., 0, [10 * 10 ** i for i in range(7)])
