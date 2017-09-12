"""
KdV Solution viewer
"""

from iwaves import from_netcdf

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
# Set some default parameters
matplotlib.rcParams['text.color']='white'
matplotlib.rcParams['savefig.facecolor']='black'
matplotlib.rcParams['savefig.edgecolor']='black'
matplotlib.rcParams['figure.facecolor']='black'
matplotlib.rcParams['figure.edgecolor']='black'
matplotlib.rcParams['axes.facecolor']='black'
matplotlib.rcParams['axes.edgecolor']='white'
matplotlib.rcParams['axes.labelcolor']='white'
matplotlib.rcParams['xtick.color']='white'
matplotlib.rcParams['ytick.color']='white'
matplotlib.rcParams['font.family']='serif'

import pdb

class viewer(object):
    
    tstep = 0
    hscale = 0.5
    uscale = 2.
    xaxis = 'distance'
    cmap = 'RdBu'
    rholevs = np.arange(20,30,0.25)

    def __init__(self, ncfile, **kwargs):
        self.mykdv, self.Bt = from_netcdf(ncfile)

        self.Nt = self.Bt.time.shape[0]

        u, w, rho = self.load_tstep(self.tstep)

        # Compute some scales
        H = np.abs(self.mykdv.z).max()
        umax = self.uscale*np.abs(u).max()
        self.clim = [-umax, umax]

        if self.xaxis == 'time':
            self.x = self.mykdv.x/self.mykdv.c1/3600. # Time hours
            xlabel = 'Time [h]'
        elif self.xaxis == 'distance':
            self.x = self.mykdv.x
            xlabel = 'Distance [m]'

        ### Build the figure
        self.fig = plt.figure(figsize = (12,8), num = 'KdV Viewer')

        # Time slider axes
        self.axtime = plt.subplot2grid((9,3), (8,0), colspan=2, rowspan=1)

        self.ax1 = plt.subplot2grid((9,3), (0,0), colspan=3, rowspan=2)
        self.ax1.set_ylim(-self.hscale*H, self.hscale*H)
        self.ax1.set_ylabel('$\eta(x,t)$ [m]')

        self.ax2 = plt.subplot2grid((9,3), (2,0), colspan=3, rowspan=6, sharex=self.ax1)
        self.ax2.set_ylabel('Depth [m]')
        self.ax2.set_xlabel(xlabel)

        # Plot the amplitude on ax1
        self.p1, = self.ax1.plot(self.x, self.mykdv.B, 'b')

        # Plot the velocity data
        self.p2 = self.ax2.pcolormesh(self.x, self.mykdv.z, u.T,
                vmin=self.clim[0], vmax=self.clim[1], cmap=self.cmap)

        self.p3 = self.ax2.contour(self.x, self.mykdv.z, rho.T, self.rholevs,
                colors='0.5', linewidths=0.5)

        axcb = plt.subplot2grid((9,3), (8,2), colspan=1, rowspan=1,)
        plt.colorbar(self.p2, cax=axcb, orientation='horizontal')


        # Create the time slider
        valstr = ' of %d'%(self.Nt-1)
        self.ts = Slider(self.axtime,\
                'Time', 0, self.Nt-1, valinit=self.tstep,\
                valfmt='%d'+valstr,facecolor='g',)

        self.ts.on_changed(self.update_slider)

        plt.tight_layout()

        plt.show()

    def load_tstep(self, t):
        """
        Get the density and velocity data for the latest time step
        """
        
        self.mykdv.B[:] = self.Bt.values[t,:]
        u,w = self.mykdv.calc_velocity()
        rho = self.mykdv.calc_density()

        return u, w, rho

    def update_slider(self, val):

        # On change of slider: load new data and set the plot objects accordingly
        t = int(np.floor(val))

        # Only update the plot if the data has changed
        if not self.tstep == t:
            u, w, rho = self.load_tstep(t)

            # update the line
            self.p1.set_ydata(self.mykdv.B)

            ## Update the contour plot
            self.ax2.collections=[]
            self.ax2.contour(self.x, self.mykdv.z, rho.T, self.rholevs,
                colors='0.5', linewidths=0.5)

            # Update the pcolor plot
            #self.p2.set_array(u[:-1,:-1].T.ravel())
            self.p2 = self.ax2.pcolormesh(self.x, self.mykdv.z, u.T,
                vmin=self.clim[0], vmax=self.clim[1], cmap=self.cmap)


            #title.set_text('%s [%s]\n%s'%(sun.long_name, sun.units,\
            #    datetime.strftime(sun.time[t], '%Y-%m-%d %H:%M:%S')))
            self.fig.canvas.draw_idle()


        


