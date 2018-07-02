import numpy as np

def get_peak_window(y, t, windowsize, nmax, fun='max', vals=[], idxs=[], ctr=0):
    """
    Returns windowed peak values from a time series
    
    Inputs
       y - input vector
       t - time vector
       windowsize - size of the window to remove each min/max (units samples)
       nmax - number of maxima/minima to extract
       fun - (optional) either max, min or maxabs
       
    Outputs
        vals - list with size nmax with vectors of size windowsize
        idxs - time step of each max value
        ctr - counter (used for recursive call)
       
    """
    
    sz = y.size
    
    if fun == 'maxabs':
        idx = np.argsort(np.abs(y))[::-1][0] # descending order
    elif fun == 'max':
        idx = np.argsort(y)[::-1][0] # descending order
    elif fun == 'min':
        idx = np.argsort(y)[0] # ascending order
    else:
        raise Exception('unknown function').with_traceback(fun)
    
    # Find the start and end indices to extract from the initial time series
    i1 = max(0, idx-windowsize//2)
    i2 = min(sz, idx+windowsize//2)

    # Init a zero-padded output window
    yout = np.zeros((windowsize,))
    
    # We want to put the peak at the halfway point
    if i2 == sz:
        istart = np.mod(i1-sz,windowsize)
        iend=windowsize
        #istart = sz - (i2-i1)
        #iend = windowsize 
    elif i1 == 0:
        istart = windowsize - i2
        iend = windowsize
    else:
        istart=0
        iend=windowsize
    
    #print idx, i1, i2, sz, istart, iend, y[i1:i2].shape,
    
    # Insert the peak into the yout array
    yout[istart:iend] = y[i1:i2]
    vals.append(yout)
    idxs.append(t[idx])
    
    # Mask the original array to remove thes maximum values
    mask = np.ones_like(y).astype(np.bool)
    mask[i1:i2] = False
    
    # Recursively call the function
    ctr+=1
    if ctr < nmax:
        vals, idxs, ctr = get_peak_window(y[mask], t[mask], windowsize, nmax,\
                        vals=vals, idxs=idxs, ctr=ctr, fun=fun)
    
    return vals, idxs, ctr
