"""
KdV Solution viewer
"""

import sys
from iwaves.utils.iwaveio import from_netcdf
from iwaves.utils.iwaveio import vkdv_from_netcdf

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib

# Set some default parameters
#matplotlib.rcParams['text.color']='white'
#matplotlib.rcParams['savefig.facecolor']='black'
#matplotlib.rcParams['savefig.edgecolor']='black'
#matplotlib.rcParams['figure.facecolor']='black'
#matplotlib.rcParams['figure.edgecolor']='black'
#matplotlib.rcParams['axes.facecolor']='black'
#matplotlib.rcParams['axes.edgecolor']='white'
#matplotlib.rcParams['axes.labelcolor']='white'
#matplotlib.rcParams['xtick.color']='white'
#matplotlib.rcParams['ytick.color']='white'
#matplotlib.rcParams['font.family']='serif'

import pdb

class viewer(object):
    
    tstep = 0
    hscale = 0.2
    uscale = 2.
    ulim = None
    xaxis = 'distance'
    cmap = 'RdBu'
    rholevs = np.arange(20,30,0.25)

    density_method='exact'
    isvkdv = False

    use_slider = True
    animate = False

    ylim = None
    xlim = None

    def __init__(self, ncfile, **kwargs):
        """
        KdV NetCDF Viewer

        Views scenes created by "solve_kdv"

        ## Inputs:
             ncfile

        ## Properties:
             fig, ax1, ax2: plot properties

        ## Defaults:
            tstep = 0
            hscale = 0.5
            uscale = 2.
            ulim = None
            xaxis = 'time'
            cmap = 'RdBu'
            rholevs = np.arange(20,30,0.25)

            density_method='exact'
            isvkdv = False

            use_slider = True
            animate = False

            ylim = None
            xlim = None
        """

        self.__dict__.update(**kwargs)

        if self.animate:
            self.use_slider=False 

        if self.isvkdv:
            self.mykdv, self.Bt = vkdv_from_netcdf(ncfile)
        else:
            self.mykdv, self.Bt = from_netcdf(ncfile)
            self.mykdv.X = self.mykdv.x
            self.mykdv.Z = self.mykdv.z

        self.Nt = self.Bt.time.shape[0]

        u, w, rho = self.load_tstep(self.tstep)

        # Compute some scales
        self.H = np.abs(self.mykdv.z).max()
        if self.ulim is None:
            umax = self.uscale*np.abs(u).max()
        else:
            umax = self.ulim
        self.clim = [-umax, umax]

        if self.xaxis == 'time':
            # Time should be backwards
            self.x = -self.mykdv.x/self.mykdv.c1/3600. # Time hours
            self.xlabel = 'Time [h]'
        elif self.xaxis == 'distance':
            self.x = self.mykdv.x
            self.xlabel = 'Distance [m]'

        ### Build the figure
        self.fig = plt.figure(figsize = (12,8), num = 'KdV Viewer')

        # Time slider axes
        if self.use_slider:
            self.axtime = plt.subplot2grid((9,3), (8,0), colspan=2, rowspan=1)

        self.ax1 = plt.subplot2grid((9,3), (0,0), colspan=3, rowspan=2)
        plt.grid(b=True)

        self.ax2 = plt.subplot2grid((9,3), (2,0), colspan=3, rowspan=6,\
                sharex=self.ax1, )
        self.create_scene(self.ax1, self.ax2, u, rho)

        if self.use_slider:
            # Create the time slider
            valstr = ' of %d'%(self.Nt-1)
            self.ts = Slider(self.axtime,\
                    'Time', 0, self.Nt-1, valinit=self.tstep,\
                    valfmt='%d'+valstr,facecolor='g',)

            self.ts.on_changed(self.update_slider)

            self.txtstr = None
        
        else:
            self.txtstr = self.ax2.text(0.8,0.1, '',\
                transform=self.ax2.transAxes)

        plt.tight_layout()

        if self.animate is False:
            plt.show()

    def create_scene(self, ax1, ax2, u, rho):
        """
        Create the current scene
        """
        if self.ylim is None:
            ax1.set_ylim(-self.hscale*self.H, self.hscale*self.H)
        else:
            ax1.set_ylim(self.ylim)

        if self.xlim is not None:
            ax1.set_xlim(self.xlim)

        #ax1.set_ylabel('$\eta(x,t)$ [m]')
        ax1.set_ylabel('$A(x,t)$ [m]')

        ax2.set_ylabel('Depth [m]')
        ax2.set_xlabel(self.xlabel)

        # Plot the amplitude on ax1
        self.p1, = ax1.plot(self.x, self.mykdv.B, 'b')

        # Plot the velocity data
        self.p2 = ax2.pcolormesh(self.mykdv.X, self.mykdv.Z, u.T,
                vmin=self.clim[0], vmax=self.clim[1], cmap=self.cmap)

        self.p3 = ax2.contour(self.mykdv.X, self.mykdv.Z, rho.T, self.rholevs,
                colors='0.5', linewidths=0.5)

        if self.use_slider:
            axcb = plt.subplot2grid((9,3), (8,2), colspan=1, rowspan=1,)
        else:
            axcb = plt.subplot2grid((9,3), (8,1), colspan=2, rowspan=1,)
        plt.colorbar(self.p2, cax=axcb, orientation='horizontal')
        axcb.set_title('u velocity [m s$^{-1}$]', fontsize=12)


    def load_tstep(self, t):
        """
        Get the density and velocity data for the latest time step
        """
        
        self.mykdv.B[:] = self.Bt.values[t,:]
        u,w = self.mykdv.calc_velocity(nonlinear=True)
        rho = self.mykdv.calc_density(method=self.density_method, nonlinear=True)

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
            self.ax2.contour(self.mykdv.X, self.mykdv.Z, rho.T, self.rholevs,
                colors='0.5', linewidths=0.5)

            # Update the pcolor plot
            #self.p2.set_array(u[:-1,:-1].T.ravel())
            self.p2 = self.ax2.pcolormesh(self.mykdv.X, self.mykdv.Z, u.T,
                vmin=self.clim[0], vmax=self.clim[1], cmap=self.cmap)


            if not self.use_slider:
                self.txtstr.set_text('Time: %3.3f [days]'%(self.Bt.time.values[t]/86400.),)

            #title.set_text('%s [%s]\n%s'%(sun.long_name, sun.units,\
            #    datetime.strftime(sun.time[t], '%Y-%m-%d %H:%M:%S')))
            self.fig.canvas.draw_idle()


        return self.p1, self.p2, self.txtstr

def animate_kdv(kdvfile, outfile, **kwargs):
    """
    Animation wrapper
    """
    print('Creating KdV animation...')
    V = viewer(kdvfile, animate=True, **kwargs)
    def init():
        return V.p1, V.p2, V.txtstr

    anim = animation.FuncAnimation(V.fig, V.update_slider,\
        init_func=init, frames=list(range(0, V.Nt)), interval=10, blit=True)

    ##anim.save("%s.mp4"%outfile, writer='mencoder', fps=6, bitrate=3600)
    anim.save("%s.mp4"%outfile, writer='ffmpeg', fps=6, bitrate=3600)
    #anim.save("%s.gif"%outfile,writer='imagemagick',dpi=90)

    print('Saved to %s.gif'%outfile)


  


if __name__=='__main__':
    

    ncfile = sys.argv[1]
    isvkdv = np.bool(sys.argv[2])
    print(ncfile)
    viewer(ncfile, isvkdv=isvkdv)
        


