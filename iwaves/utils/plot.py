import matplotlib.pyplot as plt
import numpy as np
import zutils.plotting as zplot

def hide_ticks(ax):
    
    for tick in ax.axes.get_xticklabels():
        tick.set_visible(False)

def vKdV_plot_enviro(my_vkdv, L=None, vmx=None):


    def add_vm_prof(ax, my_vkdv, vmx):
        
        if not vmx is None:
            ax.plot((vmx, vmx), (-250, 0), 'k--', zorder=1)


    def add_vm(ax, my_vkdv, vmx):
        
        if not vmx is None:
            yl = ax.get_ylim()
            ax.plot((vmx, vmx), yl, 'm--', zorder=1)
        
    if L is None:
        L = max(my_vkdv.x)
    D = np.min(my_vkdv.Z)

    ninx = 30
    nrows = 10

    widths=[10, 0.5]     # Width of each column of axes
    heights=[3, 2, 2, 2, 2 ]    # Height of each row of axes
    hspace=[2,3]       # Horizontal spacing between columns. Use single value to set all vertical spaces evenly. 
                    # Use list to set each spacing seperately
    vspace=0.5           # Vertical spacing between rows. Use single value to set all vertical spaces evenly
    hspace=0.5

    bottom = 1.5
    top = 0.5

    zl = zplot.axis_layer(widths=widths, heights=heights, hspace=hspace, vspace=vspace, bottom=bottom, top=top, left=2.5, right=2)
    zl.verbose = False # Reduce printouts

    f = plt.figure(dpi=300)

    # ax = plt.subplot2grid((nrows, ninx), (1, 0), colspan=ninx-2)
    ax=zl.lay(0, 0)

    # plt.plot(x, H, 'k')
    # plt.fill(land_x, land_H, '0.8', label='Shelf')
    plt.xlim(0, L)
    plt.ylim(D, 50)
    plt.ylabel('z (m)')

    mb = plt.pcolor(my_vkdv.X, my_vkdv.Z, np.sqrt(my_vkdv.N2), cmap='rainbow')

    # cax = plt.subplot2grid((nrows, ninx), (1, ninx-1), colspan=1)
    cax=zl.lay(0, 1)

    c = plt.colorbar(mb, cax=cax)
    c.ax.set_ylabel(r'N (s$^{-1}$)')
    hide_ticks(ax)
    add_vm_prof(ax, my_vkdv, vmx)

    ax=zl.lay(1, 0)
    # ax = plt.subplot2grid((nrows, ninx), (3, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, my_vkdv.Alpha, 'k')
    # plt.plot(my_vkdv.x, my_vkdv.r10, 'k:')
    plt.plot([0, L], [0, 0], 'k--')
    plt.xlim(0, L)
    # plt.ylim([-0.015, 0.005])
    plt.ylabel('$\\alpha$ \n ($s^{-1}$)')
    hide_ticks(ax)
    add_vm(ax, my_vkdv, vmx)

    ax=zl.lay(2, 0)
    # ax = plt.subplot2grid((nrows, ninx), (4, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, my_vkdv.Beta, 'k')
    plt.xlim(0, L)
    plt.ylabel('$\\beta$ \n ($m^3s^{-1}$)')
    hide_ticks(ax)
    add_vm(ax, my_vkdv, vmx)

    ax=zl.lay(3, 0)
    # ax = plt.subplot2grid((nrows, ninx), (5, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, my_vkdv.c1, 'k')
    plt.xlim(0, L)
    plt.ylabel('$c$ \n $(ms^{-1})$')
    hide_ticks(ax)
    add_vm(ax, my_vkdv, vmx)

    ######
    ## SIC
    # The confusing use of Q term vs. Q here is noted
    Q = my_vkdv.Qterm
    Q_x = np.gradient(my_vkdv.Qterm, my_vkdv.dx_s)
    Qterm = my_vkdv.Cn/(2.*my_vkdv.Qterm) * Q_x

    ax=zl.lay(4, 0)
    # ax = plt.subplot2grid((nrows, ninx), (7, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, 1/np.sqrt(Q), 'k')
    plt.xlim(0, L)
    # plt.ylabel(r'$\frac{1}{2Q}\frac{\partial Q}{\partial x}$', fontsize=15)
    plt.ylabel(r'$1/\sqrt{Q(x)}$')
    plt.xlabel('x (m)')
    add_vm(ax, my_vkdv, vmx)

    # f.savefig(out_dir + '/Environment.png')
    # plt.show()

    return f


def vKdV_plot_enviro2(my_vkdv):

    L = max(my_vkdv.x)
    D = np.min(my_vkdv.Z)

    ninx = 30
    nrows = 10

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
    hide_ticks(ax)

    rdist = my_vkdv.x[-1] - my_vkdv.x
    spongefac = -np.exp(-6*rdist/my_vkdv.spongedist)/my_vkdv.spongetime
    ax = plt.subplot2grid((nrows, ninx), (9, 0), colspan=ninx-2)
    plt.plot(my_vkdv.x, spongefac, 'k')
    plt.xlim(0, L)
    # plt.ylabel(r'$\frac{1}{2Q}\frac{\partial Q}{\partial x}$', fontsize=15)
    plt.ylabel(r'$r ()$', fontsize=15)
    plt.xlabel('x (m)')

    # f.savefig(out_dir + '/Environment.png')
    # plt.show()

    return f

def vKdV_plot_current(my_vkdv):

    widths=[30, 1]     # Width of each column of axes
    heights=[4, 8]    # Height of each row of axes
    hspace=[1]       # Horizontal spacing between columns. Use single value to set all vertical spaces evenly. 
                    # Use list to set each spacing seperately
    vspace=1           # Vertical spacing between rows. Use single value to set all vertical spaces evenly

    bottom = 3
    top = 1

    ax = []

    zl = zplot.axis_layer(widths=widths, heights=heights, hspace=hspace, vspace=vspace, bottom=bottom, top=top, left=3, right=2)
    zl.verbose = False # Reduce printouts

    f = plt.figure()
    ax.append(zl.lay(0,0))
    plt.plot(my_vkdv.x, my_vkdv.B, 'k')
    plt.ylabel('$\eta$ (m)')
    plt.grid()

    ax.append(zl.lay(1,0))
    rhoZ = my_vkdv.calc_density(nonlinear=True)
    plt.pcolor(my_vkdv.x, my_vkdv.Z, rhoZ.T, cmap = 'Spectral')
    plt.ylabel('z (m)')
    plt.xlabel('x (m)')

    return f, ax