import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from adjustText import adjust_text

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
               interpolation='bilinear', aspect='auto')
    # plt.colorbar()

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

    scat = plt.scatter(d1XSample, d1YSample, s=plot_size*2.0, c=d1ColSample, 
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

def plot_contour_with_points(d2X, d2Y, d2Z, d1Contours, sColours, sLineStyles,
                             d1XPoints, d1YPoints, s1Texts,
                             sXLabel=None, sYLabel=None, sTitle=None):
    contours = plt.contour(d2X, d2Y, d2Z, d1Contours, 
                           colors=sColours, linewidths=1, linestyles=sLineStyles)
    plt.clabel(contours, inline=True, fmt='%1.2f', fontsize=plot_size*1.25)
    plt.plot(d1XPoints, d1YPoints, 'or')
    
    bShowY = False
    bShowX = False
    if sXLabel is not None:
        plt.xlabel(sXLabel, fontsize=plot_size*2.0)
        bShowX = True
    if sYLabel is not None:
        plt.ylabel(sYLabel, fontsize=plot_size*2.0)
        bShowY = True
    if sTitle is not None:
        plt.title(sTitle, fontsize=plot_size*2.0)
    plt.tick_params(
        axis='both', 
        left=bShowY,
        labelleft=bShowY, 
        bottom=bShowX,
        labelbottom=bShowX)
    
    for tick in contours.ax.get_xticklabels():
        tick.set_fontsize(plot_size*1.5)
    for tick in contours.ax.get_yticklabels():
        tick.set_fontsize(plot_size*1.5)
        
    texts = []
    for i in range(len(d1XPoints)):
        texts.append(
            plt.text(d1XPoints[i], d1YPoints[i], s1Texts[i],
                      fontsize=plot_size*1.25))
        
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'))
    
def plot_contours_with_points(d2X, d2Y, d2Z, d1Contours, sColours,
                              d2X2=None, d2Y2=None, d2Z2=None, d1Contours2=None, sColours2=None, sLineStyles2=None,
                              d1XPoints=None, d1YPoints=None, s1Texts=None,
                              sXLabel=None, sYLabel=None, sTitle=None, sColorbarLabel=None):
    contourfs = plt.contourf(d2X, d2Y, d2Z, d1Contours, cmap=sColours)
    if sColorbarLabel is not None:
        cb = plt.colorbar(label=sColorbarLabel)
        cb.ax.yaxis.label.set_fontsize(plot_size*2.0)
    else:
        cb = plt.colorbar()
    for tick in cb.ax.get_yticklabels():
        tick.set_fontsize(plot_size*1.5)
        
    if d2X2 is not None:
        contours = plt.contour(d2X2, d2Y2, d2Z2, d1Contours2, 
                               colors=sColours2, linewidths=1, linestyles=sLineStyles2)
        plt.clabel(contours, inline=True, fmt='%1.2f', fontsize=plot_size*1.25)
        
    if d1XPoints is not None:
        plt.plot(d1XPoints, d1YPoints, 'or')
        texts = []
        for i in range(len(d1XPoints)):
            texts.append(
                plt.text(d1XPoints[i], d1YPoints[i], s1Texts[i],
                          fontsize=plot_size*1.25))
            
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'))
    
    bShowY = False
    bShowX = False
    if sXLabel is not None:
        plt.xlabel(sXLabel, fontsize=plot_size*2.0)
        bShowX = True
    if sYLabel is not None:
        plt.ylabel(sYLabel, fontsize=plot_size*2.0)
        bShowY = True
    if sTitle is not None:
        plt.title(sTitle, fontsize=plot_size*2.0)
    plt.tick_params(
        axis='both', 
        left=bShowY,
        labelleft=bShowY, 
        bottom=bShowX,
        labelbottom=bShowX)
    
    for tick in contourfs.ax.get_xticklabels():
        tick.set_fontsize(plot_size*1.5)
    for tick in contourfs.ax.get_yticklabels():
        tick.set_fontsize(plot_size*1.5)
        
    return cb
        
