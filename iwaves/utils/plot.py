import matplotlib.pyplot as plt
import numpy as np

def hide_ticks(ax):
    
    for tick in ax.axes.get_xticklabels():
        tick.set_visible(False)

def vKdV_plot(my_vkdv):

    L = max(my_vkdv.x)
    D = np.min(my_vkdv.Z)

    ninx = 30
    nrows = 9

    f = plt.figure(figsize=(10, 14))

    ax = plt.subplot2grid((nrows, ninx), (0, 0), colspan=ninx-2)

    # plt.plot(x, H, 'k')
    # plt.fill(land_x, land_H, '0.8', label='Shelf')
    plt.xlim(0, L)
    plt.ylim(D, 50)
    plt.ylabel('z (m)')

    mb = plt.pcolor(my_vkdv.X, my_vkdv.Z, my_vkdv.rhoZ, cmap='rainbow')

    cax = plt.subplot2grid((nrows, ninx), (0, ninx-1), colspan=1)
    c = plt.colorbar(mb, cax=cax)
    c.ax.set_ylabel(r'$\rho$ (kgm$^{-3}$)')
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (1, 0), colspan=ninx-2)
    # plt.plot(x, H, 'k')
    # plt.fill(land_x, land_H, '0.8', label='Shelf')
    plt.xlim(0, L)
    plt.ylim(D, 50)
    plt.ylabel('z (m)')

    mb = plt.pcolor(my_vkdv.X, my_vkdv.Z, np.sqrt(my_vkdv.N2), cmap='rainbow')

    cax = plt.subplot2grid((nrows, ninx), (1, ninx-1), colspan=1)
    c = plt.colorbar(mb, cax=cax)
    c.ax.set_ylabel(r'N (s$^{-1}$)')
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (2, 0), colspan=ninx-2)
    # plt.plot(x, H, 'k')
    # plt.fill(land_x, land_H, '0.8', label='Shelf')
    plt.xlim(0, L)
    plt.ylim(D, 50)
    plt.ylabel('z (m)')

    mb = plt.pcolor(my_vkdv.X, my_vkdv.Z, my_vkdv.Phi, cmap='rainbow')    

    cax = plt.subplot2grid((nrows, ninx), (2, ninx-1), colspan=1)
    c = plt.colorbar(mb, cax=cax)
    c.ax.set_ylabel(r'$\partial \phi / \partial z$ ()')
    c.ax.set_ylabel(r'$\phi$ ()')
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (3, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, my_vkdv.Alpha, 'k')
    # plt.plot(my_vkdv.x, my_vkdv.r10, 'k:')
    plt.plot([0, L], [0, 0], 'k--')
    plt.xlim(0, L)
    # plt.ylim([-0.015, 0.005])
    plt.ylabel(r'$\alpha$ (X)')
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (4, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, my_vkdv.Beta, 'k')
    plt.xlim(0, L)
    plt.ylabel(r'$\beta$ (X)')
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (5, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, my_vkdv.c1, 'k')
    plt.xlim(0, L)
    plt.ylabel(r'$c (ms^{-1})$')
    hide_ticks(ax)

    ######
    ## SIC
    # The confusing use of Q term vs. Q here is noted
    Q = my_vkdv.Qterm
    Q_x = np.gradient(my_vkdv.Qterm, my_vkdv.dx_s)
    Qterm = my_vkdv.Cn/(2.*my_vkdv.Qterm) * Q_x

    ax = plt.subplot2grid((nrows, ninx), (6, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, Q, 'k')
    plt.xlim(0, L)
    # plt.ylabel(r'$\frac{1}{2Q}\frac{\partial Q}{\partial x}$', fontsize=15)
    plt.ylabel(r'$Q(x)$', fontsize=15)
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (7, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, 1/np.sqrt(Q), 'k')
    plt.xlim(0, L)
    # plt.ylabel(r'$\frac{1}{2Q}\frac{\partial Q}{\partial x}$', fontsize=15)
    plt.ylabel(r'$1/\sqrt{Q(x)}$', fontsize=15)
    hide_ticks(ax)

    ax = plt.subplot2grid((nrows, ninx), (8, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, Qterm, 'k')
    plt.xlim(0, L)
    # plt.ylabel(r'$\frac{1}{2Q}\frac{\partial Q}{\partial x}$', fontsize=15)
    plt.ylabel(r'$\frac{c(x)}{2Q(x)}.\frac{dQ(x)}{dx} (s^{-1})$', fontsize=15)
    plt.xlabel('x (m)')

    # f.savefig(out_dir + '/Environment.png')
    # plt.show()

    return f