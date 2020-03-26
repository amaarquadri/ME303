from project_1.ODEUtils import *
from project_1.Constants import *
import matplotlib.pyplot as plt


def rayleigh_plesset_simplified(t, x, r_0, P_0):
    r, r_prime = x
    r_prime_prime = -(3 / 2) * P_0 / (rho_water * r_0) - 3 * P_0 / (rho_water * r_0 ** 2) * (r - r_0)
    return np.array([r_prime, r_prime_prime])


def analytical_solution(t, r_0, P_0):
    return (r_0 / 2) * (np.cos((1 / r_0) * np.sqrt(3 * P_0 / rho_water) * t) + 1)


def graph(r_0, r_prime_0, P_0, t_0, t_f, delta_t, file_name='1d'):
    for method in ['Euler', 'RK2', 'RK4']:
        t, x, _ = solve_ivp(rayleigh_plesset_simplified, np.array([r_0, r_prime_0]), t_0, t_f,
                            delta_t=delta_t, args=(r_0, P_0), method=method, show_progress=False)
        r = x[0, :]
        plt.plot(t * 10 ** 3, r * 10 ** 3, label=method)

        print("Max error for", method, ":", np.max(np.abs(r - analytical_solution(t, r_0, P_0))))

    # Graph analytical solution as well
    t = np.linspace(t_0, t_f, 10_000)
    r = analytical_solution(t, r_0, P_0)
    plt.plot(t * 10 ** 3, r * 10 ** 3, label='Analytical')

    plt.legend()
    plt.title(r"Bubble Radius Versus Time for $\Delta$t=" + str(delta_t))
    plt.xlabel("Time (ms)")
    plt.ylabel("Bubble Radius (mm)")
    plt.savefig("Graphs/" + file_name + ".png")
    plt.clf()


def graph_error(r_0, r_prime_0, P_0, t_0, t_f, delta_t_values):
    print("Orders of Accuracy:")
    for method in ['Euler', 'RK2', 'RK4']:
        errors = []
        for delta_t in delta_t_values:
            t, x, _ = solve_ivp(rayleigh_plesset_simplified, np.array([r_0, r_prime_0]), t_0, t_f,
                                delta_t=delta_t, args=(r_0, P_0), method=method, show_progress=False)
            r = x[0, :]
            errors.append(np.max(np.abs(r - analytical_solution(t, r_0, P_0))))
        if method is 'RK4':
            print(method, ":", np.polyfit(np.log10(delta_t_values[:3]), np.log10(errors[:3]), 1)[0])
        else:
            print(method, ":", np.polyfit(np.log10(delta_t_values), np.log10(errors), 1)[0])
        plt.plot(np.log10(delta_t_values), np.log10(errors), '-o', label=method)

    plt.legend()
    plt.title(r"Maximum Error Versus $\Delta$t")
    plt.xlabel(r"Ln($\Delta$t) (Ln(s))")
    plt.ylabel(r"Ln($\epsilon_{max}$) (Ln(m))")
    plt.savefig("Graphs/1d_errors.png")
    plt.clf()


if __name__ == '__main__':
    print("Timestep: 1E-9")
    graph(2E-3, 0, P_atmosphere + rho_water * g * 0.1, 0, 4E-4, 1E-9, file_name='1d')
    print("Timestep: 1E-5")
    graph(2E-3, 0, P_atmosphere + rho_water * g * 0.1, 0, 4E-4, 1E-5, file_name='1d_low_timestep')
    graph_error(2E-3, 0, P_atmosphere + rho_water * g * 0.1, 0, 4E-4, [10 ** -i for i in range(5, 10)])
