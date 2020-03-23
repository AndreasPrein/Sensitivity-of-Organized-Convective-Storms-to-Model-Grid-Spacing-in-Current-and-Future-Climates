#!/usr/bin/env python
'''
    File name: CoreProperties_W_3D-plot.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 11.11.2016
    Date last modified: 03.04.2018

    ##############################################################
    Purpos:

    1) Reads in the core characteristics calculated in:
       /gpfs/u/home/prein/papers/Idealized-MCSs/programs/CoreProperties/CoreProperties.py

    2) plot the convergence of properties


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


def distance(origin, destination): # 
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

# http://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def radial_profileSUM(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin
    return radialprofile, nr

# exponential Distributon
def func(x, a, c, d):
    return a*np.exp(-c*x)+d

# # Waibull Distributon
# def func(x, a, c,d):
#     # c...shape parameter
#     # a...scale parameter
#     # d...offset
#     return d*((c*a**c)*(x**(c-1))*np.exp((-a*x)**c))

# # Waibull Distributon Wilcks
# def funcWB(x, a, c):
#     # c...shape parameter
#     # a...scale parameter
#     # d...offset
#     return (c/a)*(x/a)**(c-1)*np.exp(-(x/a)**c)


def area_of_ring(r1,r2):
    """Function that defines an area of a circle"""
    a1 = r1**2 * math.pi
    a2 = r2**2 * math.pi
    a=a2-a1
    return a
xx=np.array(range(23)) # [km]
rgrArea=np.array([area_of_ring(xx[rr],xx[rr+1]) for rr in range(len(xx)-1)])


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def intertial_axis(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    angle = 0.5 * np.arctan(2 * u11 / (u20 - u02))
    return x_bar, y_bar, angle


################################################################################
################################################################################
#                            Settings
rgrGridSpacing=['12000','12000nc','4000','2000','1000','500','250']
rgsDXcol=['#1f78b4','#6a3d9a','#33a02c','#b2df8a','#fdbf6f','#ff7f00','#e31a1c']

rgrDX=[12000,12000,4000,2000,1000,500,250]
sDataDir='/glade/scratch/prein/Papers/Idealized_MCSs/data/3D_Cores/'

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

PRcharacteristic=['Updrafts','Downdrafts']
rgsStats=['Wmean','WP95','Wdepth','Wwidth','WwidthMax','Wvolume']
iEvents=10 # number of events that will be considered

sPlotDir='/glade/u/home/prein/papers/Idealized-MCSs/plots/CoreProperties/'

rgiDomSize=[51,51,155,311,623,1247,2495]
ihours=7 # runtime in hours

rgrWMean=np.zeros((2,len(rgsSimulations),len(rgrGridSpacing),101)); rgrWMean[:]=np.nan
rgrWP95=np.copy(rgrWMean)

for si in range(len(rgsSimulations)):
    print '    Load '+rgsSimulations[si]
    grDATA={}
    for dx in range(len(rgrGridSpacing)):
        try:
            fname=sDataDir+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Sigm-0.0_Random-50_MinSpeed-3.npz'
            grDATA[rgrGridSpacing[dx]]=np.load( open( fname, "rb" ) )['rgrObAll']
        except:
            continue
    for dx in range(len(rgrGridSpacing)):
        try:
            Data_Act=grDATA[rgrGridSpacing[dx]]
        except:
            continue
        # calculate geometry
        rgrWMean[0,si,dx,:]=np.nanpercentile(Data_Act[rgsStats.index('Wmean'),0,:], range(101))
        rgrWMean[1,si,dx,:]=np.nanpercentile(Data_Act[rgsStats.index('Wmean'),1,:], range(101))
        rgrWP95[0,si,dx,:]=np.nanpercentile(Data_Act[rgsStats.index('WP95'),0,:], range(101))
        rgrWP95[1,si,dx,:]=np.nanpercentile(Data_Act[rgsStats.index('WP95'),1,:], range(101))



################################################################################
#                  Plot Results

# set the font size
rgsLableABC=list(string.ascii_lowercase)

fig = plt.figure(figsize=(9,6))
plt.rcParams.update({'font.size': 12})

gs1 = gridspec.GridSpec(2,2)
gs1.update(left=0.12, right=0.99,
           bottom=0.10, top=0.93,
           wspace=0.35, hspace=0.15)

rgsColors=['k','#e31a1c']
rgsPeriods=['CUR','FUT']
rgsStatsSel=['Wmean','WP95']
UpDown=['updraft','downdraft']
for ud in range(2):
    for st in range(len(rgsStatsSel)):
        ax = plt.subplot(gs1[ud,st])
        if rgsStatsSel[st] == 'Wmean':
            DATA=rgrWMean[ud,:,:,:]
            Yrange=[[3,11],[3,6]]
            Label='W mean [m s$^{-1}$]'
        if rgsStatsSel[st] == 'WP95':
            DATA=rgrWP95[ud,:,:,:]
            Yrange=[[3,25],[3,9]]
            Label='W P95 [m s$^{-1}$]'

        # for ii in range(101):
        #     plt.plot(np.array(rgrDX).astype('float')/1000.,np.nanmean(DATA[:10,:,ii], axis=0), c='r', lw=5, alpha=0.1)
        #     plt.plot(np.array(rgrDX).astype('float')/1000.,np.nanmean(DATA[-10:,:,ii], axis=0), c='r', lw=5, alpha=0.1)

        Xaxis=np.round(np.arange(0.25,12.025,0.025),3)
        RealPoints=[np.where(Xaxis == np.array(rgrDX[ii]).astype('float')/1000.)[0][0] for ii in range(len(rgrDX))]
        for pe in range(2):
            if pe == 0:
                DATA_pe=np.nanmean(DATA[:10,:,:], axis=0)
                col='k'
            else:
                DATA_pe=np.nanmean(DATA[-10:,:,:], axis=0)
                col='r'
            # plot P5-P95
            P25=np.copy(Xaxis); P25[:]=np.nan; P25[RealPoints[1:]]=DATA_pe[1:,10]
            s25 = pd.Series(P25)
            P75=np.copy(Xaxis); P75[:]=np.nan; P75[RealPoints[1:]]=DATA_pe[1:,90]
            s75 = pd.Series(P75)
            ax.fill_between(Xaxis, s25.interpolate(method='pchip'), s75.interpolate(method='pchip'), facecolor=col, alpha=0.2)
            # plot qartile range
            P25=np.copy(Xaxis); P25[:]=np.nan; P25[RealPoints[1:]]=DATA_pe[1:,25]
            s25 = pd.Series(P25)
            P75=np.copy(Xaxis); P75[:]=np.nan; P75[RealPoints[1:]]=DATA_pe[1:,75]
            s75 = pd.Series(P75)
            ax.fill_between(Xaxis, s25.interpolate(method='pchip'), s75.interpolate(method='pchip'), facecolor=col, alpha=0.4)
            # plot median
            ii=50
            DataInt=np.copy(Xaxis); DataInt[:]=np.nan; DataInt[RealPoints[1:]]=DATA_pe[1:,ii]
            s = pd.Series(DataInt)
            plt.plot(Xaxis, s.interpolate(method='pchip'), c=col, lw=2, alpha=1)
            # plot the 12 km simulation with cumulus scheme as box wisker plot
            plt.plot([14+pe*4,14+pe*4],[DATA_pe[0,25],DATA_pe[0,75]] , c=col, lw=10, alpha=0.4,solid_capstyle="butt")
            plt.plot([14+pe*4,14+pe*4],[DATA_pe[0,10],DATA_pe[0,25]] , c=col, lw=3, alpha=0.2,solid_capstyle="butt")
            plt.plot([14+pe*4,14+pe*4],[DATA_pe[0,75],DATA_pe[0,90]] , c=col, lw=3, alpha=0.2,solid_capstyle="butt")
            plt.plot([13.2+pe*4,14.8+pe*4],[DATA_pe[0,50],DATA_pe[0,50]] , c=col, lw=2, alpha=1)
        
        xx = [16,12, 4, 2,1,0.5,0.25]
        labels = np.array([ '12 km C',  '12 km',  '4 km', '2 km','1 km','500 m', '250 m'])
        for dx in range(len(xx)):
            ax.axvline(x=xx[dx], ls='--', c='k', zorder=-1, alpha=0.6, lw=0.2)
            ax.text(xx[dx],Yrange[ud][0], labels[dx], ha='left',va='bottom', \
                    fontname="Times New Roman Bold", fontsize=9, rotation=90)
        
        plt.xticks(xx, labels, rotation=0)
        ax.get_xaxis().get_major_formatter().labelOnlyBase = False

        ax.set_ylabel(UpDown[ud]+' '+Label)
        if ud == 1:
            ax.set_xlabel('horizontal grid spacing [km]')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(0.25, 20)
        plt.axhline(y=0, c='k', lw=0.5)
        plt.ylim(Yrange[ud][0],Yrange[ud][1])
        # if rgsStatsSel[st] == 'Wmean':
        #     ax.set_yscale('log')
        ax.set_xscale('log')

        ax.text(0.03,1.03, rgsLableABC[st+ud*2]+') ', ha='left',va='bottom', \
                transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=14)
        # if st == 0:
        #     plt.legend(loc="upper right",
        #                ncol=1, prop={'size':12})

# Plot the figure
sPlotFile=sPlotDir
sPlotName= 'MCS-3D-Draft-W-properties.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
