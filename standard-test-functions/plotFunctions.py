import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plot_size = 10.0

def plot_map_with_points(d2X, d2Y, d2Col, d1Contours, sColMap, d1MapRange,
                         d1XScat, d1YScat, d1ColScat=None, sScatColMap=None,
                         d1XScat2=None, d1YScat2=None, d1ColScat2=None, sScatColMap2=None,
                         sXLabel=None, sYLabel=None, sTitle=None):
    
    contours = plt.contour(d2X, d2Y, d2Col, d1Contours, colors='black')
    plt.clabel(contours, inline=True, fmt='%1.2f', fontsize=plot_size*1.25)
    plt.imshow(d2Col, extent=[np.min(d2X), np.max(d2X), np.min(d2Y), np.max(d2Y)], origin='lower',
               cmap=sColMap, alpha=0.5, vmin=d1MapRange[0], vmax=d1MapRange[1],
               interpolation='bilinear')

    if d1ColScat is None:
        plt.scatter(d1XScat, d1YScat, s=plot_size*7.5, c='black')
    else:
        plt.scatter(d1XScat, d1YScat, s=plot_size*7.5, c=d1ColScat,
                    cmap=sScatColMap, vmin=d1MapRange[0], vmax=d1MapRange[1],
                    linewidths=1, edgecolors='k')
    
    if d1XScat2 is not None:
        if d1ColScat2 is None:
            plt.scatter(d1XScat2, d1YScat2, s=plot_size*20., c='black')
        else:
            plt.scatter(d1XScat2, d1YScat2, s=plot_size*20., c=d1ColScat2,
                        cmap=sScatColMap2, vmin=d1MapRange[0], vmax=d1MapRange[1],
                        linewidths=1, edgecolors='k')
    
    bShowY = False
    bShowX = False
    if sXLabel is not None:
        plt.xlabel(sXLabel, fontsize=plot_size*3.0)
        bShowX = True
    if sYLabel is not None:
        plt.ylabel(sYLabel, fontsize=plot_size*3.0)
        bShowY = True
    if sTitle is not None:
        plt.title(sTitle, fontsize=plot_size*2.5)
    plt.tick_params(
        axis='both', 
        left=bShowY,
        labelleft=bShowY, 
        bottom=bShowX,
        labelbottom=bShowX)
    for tick in contours.ax.get_xticklabels():
        tick.set_fontsize(plot_size*2.0)
    for tick in contours.ax.get_yticklabels():
        tick.set_fontsize(plot_size*2.0)

def plot_sample_with_points(d1XSample, d1YSample, d1ColSample, sColMap, d1MapRange, sSampleLabel,
                            d1X1, d1Y1, sCol1, sPoints1Label,
                            d1X2=None, d1Y2=None, sCol2=None, sPoints2Label=None,
                            sXLabel=None, sYLabel=None, sTitle=None):

    scat = plt.scatter(d1XSample, d1YSample, s=plot_size*10.0, c=d1ColSample, 
                    cmap=sColMap, vmin=d1MapRange[0], vmax=d1MapRange[1],
                    label = sSampleLabel)
    plt.plot(d1X1, d1Y1,
             c = sCol1, linestyle='', marker = '.', markersize=plot_size*1.5, 
             label = sPoints1Label)
    if d1X2 is not None:
        plt.plot(d1X2, d1Y2, 
                 linestyle = '', marker = 'o', color = sCol2, markersize = plot_size*1.5, 
                 label = sPoints2Label)
    
    bShowY = False
    bShowX = False
    if sXLabel is not None:
        plt.xlabel(sXLabel, fontsize=plot_size*3.0)
        bShowX = True
    if sYLabel is not None:
        plt.ylabel(sYLabel, fontsize=plot_size*3.0)
        bShowY = True
    if sTitle is not None:
        plt.title(sTitle, fontsize=plot_size*2.5)
    plt.tick_params(
        axis='both', 
        left=bShowY,
        labelleft=bShowY, 
        bottom=bShowX,
        labelbottom=bShowX)
    
    for tick in scat.axes.get_xticklabels():
        tick.set_fontsize(plot_size*2.0)
    for tick in scat.axes.get_yticklabels():
        tick.set_fontsize(plot_size*2.0)
    plt.legend(loc='upper right', labelspacing=0.25, fontsize=plot_size*2.0)
