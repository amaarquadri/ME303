"""
This is a demo code for the final project of ME303, Winter 2020.
Author: TA X. Zhang, Apr. 2020.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plot settings
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family'] = 'serif'
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
figl = 7.6
figh = 3.8

if __name__ == '__main__':
    '''
    modelling parmeters (according to Table 1, I already typed it up for you)
    '''
    # parameters of the egg
    egg = 'Chicken egg'
    V = 0.054 * 1e-3  # chicken egg volume, unit: [m^3]
    rho = 1.03  # egg density, unit: [kg /m^3]
    Cp = 3800  # egg specific heat capacity, unit: [J/(kg degC)]
    # parameters of cooking ware
    D = 186 * 1e-3  # diameter of the egg ring, unit:[m]
    T_g = 160  # temperature of the griddle surface, unit: [degC]
    T_a = 20  # ambient temperature of the kitchen, unit: [degC]
    T_cook = 80  # threshold temperature for egg to be cooked, unit: [degC]
    # parameters of heat transfer
    h_m = 5  # mean free convection coefficient, unit: [W/(m^2 K)]
    k = 0.5  # thermal conductivity of whisked egg, unit: [W/(m K)]
    k_v = 0.025  # thermal conductivity of water vapour, unit: [W/(m K)]
    # group some parameters (in eq. 2 and eq. 6)
    B = k / (rho * Cp)  # constant in eq 2
    K = k_v / h_m  # constant in eq 6

    '''
    time and space discretization
    '''
    H = V / (np.pi * D * D / 4)  # omlet thickness/height, unit [m]
    T = 0.005  # time domain: t in (0, T)
    N = np.uint(10)  # N space nodes on (0, H)
    M = np.uint(1e4)  # M time nodes on (0, T)
    dy = H / N  # grid spacing for space
    dt = T / M  # grid spacing for time

    # setup space of nodes
    y = list([])
    for i in range(N + 1):
        y.append(i * dy)
    y = np.array(y)

    # setup time nodes
    t = list([])
    for j in range(M + 1):
        t.append(j * dt)
    t = np.array(t)

    '''
    setup initial condition (IC)
    '''
    T0 = list([])
    for i in range(N + 1):
        T0.append(T_a)  # set up your IC HERE!!!
    T0 = np.array(T0)

    T_old = T0 * 1  # initialize the problem
    Temp = np.zeros([len(t), len(T0)])
    Temp[0, :] = T0[:]

    '''
    explicit scheme for partial differntial equation
    '''
    T_new = np.zeros_like(T_old)  # initialize T_new for iteration
    for j in range(1, M + 1):
        '''
        set up PDE using
        '''
        for i in range(1, N):  # space
            T_new[i] = T_old[i] + B * dt / dy ** 2 * (T_old[i + 1] - 2 * T_old[i] + T_old[i - 1])

        '''
        boundary condition at y = 0
        '''
        T_new[0] = T_g  # set up BC (1) HERE!!!

        '''
        boundary condition at y = H
        '''
        T_new[N] = T_a * K * T_new[N - 1] / dy / (1 + K / dy)  # set up BC (2) HERE!!!

        T_old = T_new * 1  # update in time
        Temp[j, :] = T_new[:] # store data

    '''
    visualizing the results
    '''
    fig = plt.figure(figsize = (figl, figh))
    ax = fig.gca(projection='3d')
    X = np.arange(0, H+dy, dy)
    Y = np.arange(0, T+dt, dt)
    X, Y = np.meshgrid(X, Y)
    s = ax.plot_surface(X, Y, Temp, cmap='inferno', linewidth=0, antialiased=False)
    ax.set_xlabel(r'$y\:\mathrm{[m]}$', fontsize = 11)
    ax.set_xticks(np.linspace(0, 0.002, 5))
    ax.set_ylabel(r'$t\:\mathrm{[sec]}$', fontsize = 11)
    fig.colorbar(s, shrink=0.5, aspect=5)
    plt.show()