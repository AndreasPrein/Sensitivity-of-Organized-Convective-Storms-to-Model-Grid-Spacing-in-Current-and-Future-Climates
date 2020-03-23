#!/usr/bin/env python
'''
    File name: ColdPools_plot.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 11.11.2016
    Date last modified: 03.04.2018

    ##############################################################
    Purpos:

    1) Reads in the core characteristics calculated in:
       /gpfs/u/home/prein/papers/Idealized-MCSs/programs/ColdPools/ColdPoolProperties.py

    2) plot the differences in cold pool properties accross scales


'''

from dateutil import rrule
import datetime
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
from pdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
# from mpl_toolkits import basemap
import ESMF
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec
# from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
# import shapely.geometry
# import descartes
import shapefile
import math
from scipy.stats.kde import gaussian_kde
from math import radians, cos, sin, asin, sqrt
from scipy import spatial
import matplotlib.path as mplPath
from pylab import *
from scipy.optimize import curve_fit
import scipy
import math
import SkewT
from scipy.interpolate import interp1d
# from astropy.io import ascii

################################################################################
################################################################################
#                            Settings
rgrGridSpacing=['12000','12000nc','4000','2000','1000','500','250']
rgsDXcol=['#1f78b4','#6a3d9a','#33a02c','#b2df8a','#fdbf6f','#ff7f00','#e31a1c']
sGrid='12km'  # can be 'native' for native grid or '12km' for 12 km model grid
rgrDX=[12000,12000,4000,2000,1000,500,250]
sDataDir='/glade/scratch/prein/Papers/Idealized_MCSs/data/Coldpools/'+sGrid+'/'

rgsSimulations=['19_2011-07-13_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '03_2011-07-16_CTRL_Midwest_-Loc2_MCS_Storm-Nr_JJA-8-TH5',
                '23_2007-06-19_CTRL_Midwest_-Loc2_MCS_Storm-Nr_JJA-8-TH5',
                '10_2009-06-27_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '13_2003-08-30_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '17_2011-06-27_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '18_2010-06-13_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '38_2007-08-04_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '46_2009-06-14_CTRL_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '07_2011-07-04_CTRL_Midwest_-Loc2_MCS_Storm-Nr_JJA-8-TH5',

                '64_2012-06-17_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '58_2009-06-08_PGW_Midwest_-Loc2_MCS_Storm-Nr_JJA-8-TH5',
                '41_2005-06-10_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '68_2013-07-07_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '31_2006-08-18_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '16_2002-06-11_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '34_2010-07-12_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '51_2003-06-23_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '56_2008-06-18_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5',
                '35_2004-07-02_PGW_Midwest_-Loc1_MCS_Storm-Nr_JJA-8-TH5']

iEvents=10 # number of events that will be considered

sPlotDir='/glade/u/home/prein/papers/Idealized-MCSs/plots/ColdPools/'

rgiDomSize=[51,51,155,311,623,1247,2495]
ihours=7 # runtime in hours

CP_volume=np.zeros((85,len(rgsSimulations),len(rgrGridSpacing))); CP_volume[:]=np.nan
CP_depth=np.copy(CP_volume)
CP_intensity=np.copy(CP_volume)
CP_speed=np.copy(CP_volume)
CP_extend=np.copy(CP_volume)

for si in range(len(rgsSimulations)):
    print '    Load '+rgsSimulations[si]
    grDATA={}
    for dx in range(len(rgrGridSpacing)):
        try:
            fname=sDataDir+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Smooth-12km.pkl'
            dbfile = open(fname, 'rb')      
            grDATA[rgrGridSpacing[dx]] = pickle.load(dbfile) 
        except:
            continue
    for dx in range(len(rgrGridSpacing)):
        try:
            Data_Act=grDATA[rgrGridSpacing[dx]]
        except:
            continue

        sMetrix=Data_Act['rgsMetricCPall']
        CP_volume[:,si,dx]=Data_Act['rgrCPPorpAll'][sMetrix.index('Volume'),:][:85]
        CP_depth[:,si,dx]=Data_Act['rgrCPPorpAll'][sMetrix.index('MeanDepth'),:][:85] # [sMetrix.index('P95Depth'),:][:85]
        CP_extend[:,si,dx]=Data_Act['rgrCPPorpAll'][sMetrix.index('Area'),:][:85] # [sMetrix.index('P95Depth'),:][:85]
        Obj_Nr=Data_Act['IntenseCPs'].keys()
        ObjCharact=np.zeros((3,85,len(Obj_Nr))); ObjCharact[:]=np.nan
        for ob in range(len(Obj_Nr)):
            ObjCharact[0,:,ob]=Data_Act['IntenseCPs'][Obj_Nr[ob]]['rgrCP_Max'][:85] # max int.
            ObjCharact[1,1:,ob]=Data_Act['IntenseCPs'][Obj_Nr[ob]]['rgrObjSpeed'][:84] # speed
            ObjCharact[2,:,ob]=Data_Act['IntenseCPs'][Obj_Nr[ob]]['rgrCP_Vol'][:85]
        ObjCharact[ObjCharact == 0]=np.nan
        CP_vol=np.nansum(ObjCharact[2,:,:], axis=0)
        Largest_CP=np.argmax(CP_vol)

        CP_intensity[:,si,dx]=ObjCharact[0,:,Largest_CP]
        CP_speed[:,si,dx]=ObjCharact[1,:,Largest_CP]

################################################################################
#                  Plot Results

# set the font size
rgsLableABC=list(string.ascii_lowercase)

fig = plt.figure(figsize=(14,7))
plt.rcParams.update({'font.size': 12})

gs1 = gridspec.GridSpec(2,3)
gs1.update(left=0.06, right=0.97,
           bottom=0.15, top=0.93,
           wspace=0.35, hspace=0.35)

rgsColors=['k','#e31a1c']
rgsPeriods=['CUR','FUT']
rgsStatsSel=['extend [km$^3$]','P95 depth [km]', 'volume [km$^3$]', 'max. intensity [m s$^{-1}$]','volume evolution []', 'depth evolution []']

iXX=[0,1,2,0,1,2]
iYY=[0,0,0,1,1,1]
rgsColors=['k','#e31a1c']
for st in range(len(rgsStatsSel)):
    ax = plt.subplot(gs1[iYY[st],iXX[st]])
    if rgsStatsSel[st] == 'volume [km$^3$]':
        DATA=CP_volume
        Yrange=[-60,20]
        Label='cold pool volume [%]'
    if rgsStatsSel[st] == 'P95 depth [km]':
        DATA=CP_depth
        Yrange=[-60,10]
        Label='mean depth [%]'
    if rgsStatsSel[st] == 'volume evolution []':
        DATA=CP_volume
        Yrange=[0,1]
        Label='volume evolution [km$^{3}$]'
    if rgsStatsSel[st] == 'max. intensity [m s$^{-1}$]':
        DATA=CP_intensity
        Yrange=[-45,25]
        Label='max. intensity [%]'
    if rgsStatsSel[st] == 'extend [km$^3$]':
        DATA=CP_extend
        Yrange=[-55,55]
        Label='horizontal extent [%]'
    if rgsStatsSel[st] == 'depth evolution []':
        DATA=CP_depth
        Yrange=[0,1]
        Label='depth evolution [km]'

    Xaxis=np.round(np.arange(0,len(rgrGridSpacing)+0.025,0.025),3)
    RealPoints = [np.where(Xaxis == ii)[0][0] for ii in range(len(rgrDX))]
    if 'evolution' not in rgsStatsSel[st]:
        for pe in range(2):
            col=rgsColors[pe]
            if pe == 0:
                DATA_pe=np.nanmean(DATA[12*3:,:10,:], axis=(0))
            else:
                DATA_pe=np.nanmean(DATA[12*3:,-10:,:], axis=(0))
            Data_Diff=((DATA_pe[:,:]-DATA_pe[:,-1][:,None])/DATA_pe[:,-1][:,None])*100.
            # median
            DataInt=np.copy(Xaxis); DataInt[:]=np.nan; DataInt[RealPoints[:]]=np.nanmedian(Data_Diff, axis=0)
            s = pd.Series(DataInt)
            plt.plot(Xaxis, s.interpolate(method='pchip'), c=col, lw=2, alpha=1)
    
            # plot P0-P100
            P25=np.copy(Xaxis); P25[:]=np.nan; P25[RealPoints[:]]=np.nanpercentile(Data_Diff, 10, axis=0)
            s25 = pd.Series(P25)
            P75=np.copy(Xaxis); P75[:]=np.nan; P75[RealPoints[:]]=np.nanpercentile(Data_Diff, 90, axis=0)
            s75 = pd.Series(P75)
            ax.fill_between(Xaxis, s25.interpolate(method='pchip'), s75.interpolate(method='pchip'), facecolor=col, alpha=0.1)
    
            # plot P25-P75
            P25=np.copy(Xaxis); P25[:]=np.nan; P25[RealPoints[:]]=np.nanpercentile(Data_Diff, 25, axis=0)
            s25 = pd.Series(P25)
            P75=np.copy(Xaxis); P75[:]=np.nan; P75[RealPoints[:]]=np.nanpercentile(Data_Diff, 75, axis=0)
            s75 = pd.Series(P75)
            ax.fill_between(Xaxis, s25.interpolate(method='pchip'), s75.interpolate(method='pchip'), facecolor=col, alpha=0.5)
        
        labels = np.array([ '12 km C',  '12 km',  '4 km', '2 km','1 km','500 m', '250 m'])
        # for dx in range(len(xx)):
        #     ax.axvline(x=xx[dx], ls='--', c='k', zorder=-1, alpha=0.6, lw=0.2)
        #     ax.text(xx[dx],Yrange[ud][0], labels[dx], ha='left',va='bottom', \
        #             fontname="Times New Roman Bold", fontsize=9, rotation=90)
        
        plt.xticks(range(len(labels)), labels, rotation=20)
        ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    
        ax.set_ylabel(Label)
        if iYY[st] == 1:
            ax.set_xlabel('horizontal grid spacing [km]')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(0, len(rgrDX)-1)
        plt.axhline(y=0, c='k', lw=0.5)
        plt.ylim(Yrange[0],Yrange[1])
        for dx in range(len(rgrDX)):
            ax.axvline(x=dx, ls='--', c='#737373', zorder=-1, lw=0.2)
    
        ax.text(0.03,1.03, rgsLableABC[st]+') ', ha='left',va='bottom', \
                transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=14)
        # if st == 0:
        #     plt.legend(loc="upper right",
        #                ncol=1, prop={'size':12})
    else:
        # plot the evolution of a property
        LW=[3,1]
        for pe in range(2):
            col=rgsColors[pe]
            if pe == 0:
                DATA_pe=DATA[:,:10,:]
                colors=['#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
            else:
                DATA_pe=DATA[:,-10:,:]
                colors=['#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']
            Xaxis=np.linspace(0,1,100)
            NormDynamics=np.zeros((len(Xaxis),10,len(rgrDX))); NormDynamics[:]=np.nan
            for si in range(10):
                for dx in range(len(rgrDX)):
                    if np.sum(~np.isnan(DATA_pe[:,si,dx])) == 0:
                        continue
                    else:
                        CPvol=DATA_pe[:,si,dx]
                        if CPvol.min() == 0:
                            NonZero=CPvol[(CPvol != 0)]
                        else:
                            NonZero=CPvol[~np.isnan(CPvol)]
                        f = interp1d(np.linspace(0,1,len(NonZero)), NonZero)
                        NormDynamics[:,si,dx]=f(Xaxis)
            # normalize intensity
            # NormDynamics=np.nanmean(NormDynamics/np.max(NormDynamics,axis=0)[None,:],axis=1)
            NormDynamics=np.nanmean(NormDynamics,axis=1)
            for dx in range(len(rgrDX)):
                ax.plot(Xaxis,NormDynamics[:,dx], lw=LW[pe], c=colors[dx], label=rgrGridSpacing[dx] if (pe == 0) & (iXX[st] == 1) else '')
                ax.plot(Xaxis,NormDynamics[:,dx], lw=LW[pe], c=colors[dx], label=rgrGridSpacing[dx] if (pe == 1) & (iXX[st] == 2) else '')
            plt.legend(loc="lower right",
                       ncol=1, prop={'size':9})
            
        ax.set_ylabel(Label)
        ax.set_xlabel('normalized time []')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(0, 1)
        # plt.ylim(Yrange[0],Yrange[1])
    
        ax.text(0.03,1.03, rgsLableABC[st]+') ', ha='left',va='bottom', \
                transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=14)


# Plot the figure
sPlotFile=sPlotDir
sPlotName= 'MCS-Coldpool-Characteristics_'+sGrid+'.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
