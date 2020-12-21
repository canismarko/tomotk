import matplotlib.pyplot as plt


def plot_orthogonal_views(vol, z, y, x, figsize=(4, 4), layout="square", vmin=0, vmax=0.0045, **kwargs):
    """Plot orthogonal slices of a 3D volume.

    Parameters
    ==========
    vol
      3-dimensional numpy array
    z, y, x
      Indices for the 3 slices to plot
    figsize
      Same as plt.figure()
    layout
      Either "square" (2x2 square) or "line" (1x3 line) layout
    vmin, vmax
      Minimum and maximum for plotting
    **kwargs
      Extra keyword arguments passed to plt.imshow()
    
    Returns
    =======
    fig
      Matplotlib figure
    axs
      Array of matplotlib Axes
    ims
      Array of matplotlib artists from plt.imshow()
    
    """
    kw = dict(vmin=vmin, vmax=vmax, **kwargs)
    if layout == "square":
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axXY = axs[0,0]
        axYZ = axs[0,1]
        axXZ = axs[1,0]
    elif layout == "line":
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        axXY, axYZ, axXZ = axs
    
    ims = []
    
    line_kw = dict(alpha=0.5, linewidth=1, color="C1")
    
    slc = vol[z]
    extent = (0, slc.shape[1]*px_size, slc.shape[0]*px_size, 0)
    ax = axXY
    ax.axvline(x*px_size, **line_kw)
    ax.axhline(y*px_size, **line_kw)
    ims.append(ax.imshow(slc, extent=extent, **kw))
    ax.set_xlabel("X /µm")
    ax.set_ylabel("Y /µm")
    
    slc = vol[:,y]
    extent = (0, slc.shape[1]*px_size, slc.shape[0]*px_size, 0)
    ax = axXZ
    ims.append(ax.imshow(slc, extent=extent, **kw))
    ax.axvline(x*px_size, **line_kw)
    ax.axhline(z*px_size, **line_kw)
    ax.set_xlabel("X /µm")
    ax.set_ylabel("Z /µm")
    
    slc = vol[:,:,x]
    slc = np.swapaxes(slc, 0, 1)
    extent = (0, slc.shape[1]*px_size, slc.shape[0]*px_size, 0)
    ax = axYZ
    ims.append(ax.imshow(slc, extent=extent, **kw))
    ax.axvline(z*px_size, **line_kw)
    ax.axhline(y*px_size, **line_kw)
    ax.set_xlabel("Z /µm")
    ax.set_ylabel("Y /µm")
    return fig, axs, ()
