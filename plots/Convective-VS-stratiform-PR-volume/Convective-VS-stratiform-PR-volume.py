#!/usr/bin/env python
'''
    File name: Convective-VS-stratiform-PR-volume.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 11.11.2016
    Date last modified: 03.04.2018

    ##############################################################
    Purpos:

    1) Reads in the precipitation characteristics calculated in:
       /gpfs/u/home/prein/papers/Idealized-MCSs/programs/Convective-VS-stratiform-PR-volume/Convective-VS-stratiform-PR-volume.py

    2) calculates the differences between the 250 m characteristics and the coarser models
       Only analyse the largest object, which has key '0'!

    3) plot the differences dependent on dx


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
sGrid='12km'  # can be 'native' for native grid or '12km' for 12 km model grid
rgrGridSpacing=['12000','12000nc','4000','2000','1000','500','250']
rgsDXcol=['#1f78b4','#6a3d9a','#33a02c','#b2df8a','#fdbf6f','#ff7f00','#e31a1c']

if sGrid == 'native':
    rgrDX=[12000,12000,4000,2000,1000,500,250]
if sGrid == '12km':
    rgrDX=[12000]*len(rgrGridSpacing)
sDataDir='/glade/scratch/prein/Papers/Idealized_MCSs/data/Conv-vs-Strat_PR/'+sGrid+'/'

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

PRcharacteristic=['Convective','Stratiform']
rgsStats=['Mean PR Diff.','Area Diff.','PR Volume Diff.', 'Mean PR Diff.','Area Diff.','PR Volume Diff.']
iEvents=10 # number of events that will be considered

sPlotDir='/glade/u/home/prein/papers/Idealized-MCSs/plots/Convective-VS-stratiform-PR-volume/'

rgiDomSize=[51,51,155,311,623,1247,2495]
ihours=7 # runtime in hours
iSkipTi=1*12 # hours times intervals per hour that should be excluded from the simulation

rgrPR_CS_Vol=np.zeros((2,len(rgsSimulations),len(rgrGridSpacing),5)); rgrPR_CS_Vol[:]=np.nan
rgrPR_CS_Mean=np.copy(rgrPR_CS_Vol)
CS_Size=np.copy(rgrPR_CS_Vol)

for si in range(len(rgsSimulations)):
    print '    Load '+rgsSimulations[si]
    grDATA={}
    for dx in range(len(rgrGridSpacing)):
        try:
            fname=sDataDir+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Smooth-36000_'+sGrid+'-grid_DBZ-2km.pkl'
            grDATA[rgrGridSpacing[dx]]=pickle.load( open( fname, "rb" ) )['0']
        except:
            continue
    grRef=grDATA['250']
    for dx in range(len(rgrGridSpacing)-1):
        try:
            grMod=grDATA[rgrGridSpacing[dx]]
        except:
            continue

        # calculate mean PR diff.
        rgrRefMeanPR_Conv=grRef['rgrPR_Conv_Mean'][24:]
        rgrModMeanPR_Conv=grMod['rgrPR_Conv_Mean'][24:]
        if len(rgrModMeanPR_Conv) < len(rgrRefMeanPR_Conv):
            rgrRefMeanPR_Conv=rgrRefMeanPR_Conv[:len(rgrModMeanPR_Conv)]
        if len(rgrRefMeanPR_Conv) < len(rgrModMeanPR_Conv):
            rgrModMeanPR_Conv=rgrModMeanPR_Conv[:len(rgrRefMeanPR_Conv)]
        rgrPR_CS_Mean[0,si,dx,:]=np.nanpercentile((rgrModMeanPR_Conv-rgrRefMeanPR_Conv)/rgrRefMeanPR_Conv, (0,25,50,75,100))*100
        # rgrPR_CS_Mean[0,si,dx,:]=((np.nanpercentile(rgrModMeanPR_Conv, (0,25,50,75,100))-np.nanpercentile(rgrRefMeanPR_Conv, (0,25,50,75,100)))/np.nanpercentile(rgrRefMeanPR_Conv, (0,25,50,75,100)))*100 
        rgrRefMeanPR_Strat=grRef['rgrPR_Strat_Mean'][24:]
        rgrModMeanPR_Strat=grMod['rgrPR_Strat_Mean'][24:]
        if len(rgrModMeanPR_Strat) < len(rgrRefMeanPR_Strat):
            rgrRefMeanPR_Strat=rgrRefMeanPR_Strat[:len(rgrModMeanPR_Strat)] 
        if len(rgrRefMeanPR_Strat) < len(rgrModMeanPR_Strat):
            rgrModMeanPR_Strat=rgrModMeanPR_Strat[:len(rgrRefMeanPR_Strat)]     
        # rgrPR_CS_Mean[1,si,dx,:]=((np.nanpercentile(rgrModMeanPR_Strat, (0,25,50,75,100))-np.nanpercentile(rgrRefMeanPR_Strat, (0,25,50,75,100)))/np.nanpercentile(rgrRefMeanPR_Strat, (0,25,50,75,100)))*100     
        rgrPR_CS_Mean[1,si,dx,:]=np.nanpercentile((rgrModMeanPR_Strat-rgrRefMeanPR_Strat)/rgrRefMeanPR_Strat, (0,25,50,75,100))*100
        MinLen=np.min([len(grRef['rgrPR_Strat_Mean']),len(grMod['rgrPR_Strat_Mean'])])

        # calculate PR vol difference.
        rgrRefVolPR_Conv=grRef['rgrPR_Conv_Vol'][24:MinLen]
        rgrModVolPR_Conv=grMod['rgrPR_Conv_Vol'][24:MinLen]
        NAN=((rgrRefVolPR_Conv == 0) | (rgrModVolPR_Conv == 0))
        rgrRefVolPR_Conv[NAN]=np.nan; rgrModVolPR_Conv[NAN]=np.nan
        # rgrPR_CS_Vol[0,si,dx,:]=((np.nanpercentile(rgrModVolPR_Conv, (0,25,50,75,100))-np.nanpercentile(rgrRefVolPR_Conv, (0,25,50,75,100)))/np.nanpercentile(rgrRefVolPR_Conv, (0,25,50,75,100)))*100
        rgrPR_CS_Vol[0,si,dx,:]=np.nanpercentile((rgrModVolPR_Conv-rgrRefVolPR_Conv)/rgrRefVolPR_Conv, (0,25,50,75,100))*100
        rgrRefVolPR_Strat=grRef['rgrPR_Strat_Vol'][24:MinLen]
        rgrModVolPR_Strat=grMod['rgrPR_Strat_Vol'][24:MinLen]
        NAN=((rgrRefVolPR_Strat == 0) | (rgrModVolPR_Strat == 0))
        rgrRefVolPR_Strat[NAN]=np.nan; rgrModVolPR_Strat[NAN]=np.nan
        # rgrPR_CS_Vol[1,si,dx,:]=((np.nanpercentile(rgrModVolPR_Strat, (0,25,50,75,100))-np.nanpercentile(rgrRefVolPR_Strat, (0,25,50,75,100)))/np.nanpercentile(rgrRefVolPR_Strat, (0,25,50,75,100)))*100
        rgrPR_CS_Vol[1,si,dx,:]=np.nanpercentile((rgrModVolPR_Strat-rgrRefVolPR_Strat)/rgrRefVolPR_Strat, (0,25,50,75,100))*100
        
        # calculate size diff.
        rgrRefSize_Conv=grRef['Conv_Size'][24:MinLen]
        rgrModSize_Conv=grMod['Conv_Size'][24:MinLen]
        NAN=((rgrRefSize_Conv == 0) | (rgrModSize_Conv == 0))
        rgrRefSize_Conv[NAN]=np.nan; rgrModSize_Conv[NAN]=np.nan
        # CS_Size[0,si,dx,:]=((np.nanpercentile(rgrModSize_Conv, (0,25,50,75,100))-np.nanpercentile(rgrRefSize_Conv, (0,25,50,75,100)))/np.nanpercentile(rgrRefSize_Conv, (0,25,50,75,100)))*100
        CS_Size[0,si,dx,:]=np.nanpercentile((rgrModSize_Conv-rgrRefSize_Conv)/rgrRefSize_Conv, (0,25,50,75,100))*100
        rgrRefSize_Strat=grRef['Strat_Size'][24:MinLen]
        rgrModSize_Strat=grMod['Strat_Size'][24:MinLen]
        NAN=((rgrRefSize_Strat == 0) | (rgrModSize_Strat == 0))
        rgrRefSize_Strat[NAN]=np.nan; rgrModSize_Strat[NAN]=np.nan
        CS_Size[1,si,dx,:]=np.nanpercentile((rgrModSize_Strat-rgrRefSize_Strat)/rgrRefSize_Strat, (0,25,50,75,100))*100
        # CS_Size[1,si,dx,:]=((np.nanpercentile(rgrModSize_Strat, (0,25,50,75,100))-np.nanpercentile(rgrRefSize_Strat, (0,25,50,75,100)))/np.nanpercentile(rgrRefSize_Strat, (0,25,50,75,100)))*100


################################################################################
#                  Plot Results

# set the font size
rgsLableABC=list(string.ascii_lowercase)

fig = plt.figure(figsize=(14,6))
plt.rcParams.update({'font.size': 12})
iXX=[0,1,2,0,1,2]
iYY=[0,0,0,1,1,1]
gs1 = gridspec.GridSpec(2,3)
gs1.update(left=0.08, right=0.99,
           bottom=0.10, top=0.93,
           wspace=0.35, hspace=0.4)

rgsColors=['k','#e31a1c']
rgsPeriods=['CUR','FUT']
for st in range(len(rgsStats)):
    ax = plt.subplot(gs1[iYY[st],iXX[st]])
    if rgsStats[st] == 'Mean PR Diff.':
        DATA=rgrPR_CS_Mean[iYY[st],:,:,2]
        Yrange=[-100,100]
    if rgsStats[st] == 'PR Volume Diff.':
        DATA=rgrPR_CS_Vol[iYY[st],:,:,2]
        Yrange=[-100,100]
    if rgsStats[st] == 'Area Diff.':
        DATA=CS_Size[iYY[st],:,:,2]
        Yrange=[-100,100]


    rgrPolygons=np.zeros((2,len(rgrGridSpacing),5)); rgrPolygons[:]=np.nan
    for dx in range(len(rgrGridSpacing)):
        iCTR=np.array([('_CTRL_' in rgsSimulations[ii]) & ('TH5PGW' not in rgsSimulations[ii])  for ii in range(len(rgsSimulations))])
        iPGW=np.array(['_PGW_' in rgsSimulations[ii]  for ii in range(len(rgsSimulations))])
        iCTRPGW=np.array(['TH5PGW' in rgsSimulations[ii]  for ii in range(len(rgsSimulations))])
        rgsSimCTR=np.array(rgsSimulations)[iCTR == True]; rgsSimPGW=np.array(rgsSimulations)[iPGW == True]; rgsSimCTRPGW=np.array(rgsSimulations)[iCTRPGW == True]
        for si in range(2):
            if si == 0:
                rgiACT=(iCTR == True)
                sNames=rgsSimCTR
            elif si == 1:
                rgiACT=(iPGW == True)
                sNames=rgsSimPGW
            elif si ==2:
                rgiACT=(iCTRPGW == True)
                sNames=rgsSimCTRPGW
            rgrDataAct=DATA[rgiACT,dx]
            # rgrDataAct=rgrDataAct*12000.**2
            # divide by area
            rgrPercentiles=np.nanpercentile(rgrDataAct, (10,25,50,75,90))
            rgrPolygons[si,dx,:]=rgrPercentiles
            rgrPolygons[np.isnan(rgrPolygons)]=0
            # # B-W plot representation
            # plt.plot([dx+0.25+0.4*si,dx+0.25+0.4*si], [rgrPercentiles[1],rgrPercentiles[3]], c=rgsColors[si] , lw=15, alpha=0.5,solid_capstyle="butt", zorder=2, label=rgsPeriods[si] if dx == 0 else "")
            # plt.plot([dx+0.25+0.4*si,dx+0.25+0.4*si], [rgrPercentiles[1],rgrPercentiles[0]], c=rgsColors[si] , lw=1, alpha=1,solid_capstyle="butt", zorder=2)
            # plt.plot([dx+0.25+0.4*si,dx+0.25+0.4*si], [rgrPercentiles[3],rgrPercentiles[4]], c=rgsColors[si] , lw=1, alpha=1,solid_capstyle="butt", zorder=2)
            # plt.plot(dx+0.25+0.4*si, rgrPercentiles[2], 'k_', zorder=2, markersize=12)
            # for ii in range(len(rgrDataAct)):
            #     plt.plot(dx+0.25+0.4*si, rgrDataAct[ii], 'wo', zorder=0, marker='o', color=rgsColors[si], markersize=2, alpha=0.6)
            

            if dx == (len(rgrGridSpacing)-1):
                # plot horizontal reference lines
                ax.axhline(y=rgrPercentiles[2], ls='-', c='k', zorder=-1, alpha=0.6, lw=0.2)
        ax.axvline(x=dx+0.5, ls='-', c='k', zorder=-1, lw=0.2)
    # Contour plot representation
    sInterp='slinear'
    sSteps=50
    for si in range(2):
        rgrXX=np.array(range(len(rgrGridSpacing)))+0.5
        x_new = np.linspace(rgrXX.min(), rgrXX.max(),sSteps)
        rgrYY=rgrPolygons[si,:,0]
        f = interp1d(rgrXX, rgrYY, kind=sInterp)
        y_smoothLO=f(x_new)
        rgrYY=rgrPolygons[si,:,4]
        f = interp1d(rgrXX, rgrYY, kind=sInterp)
        y_smoothHI=f(x_new)
        x_new=np.append(x_new,x_new[::-1])
        y_smooth=np.append(y_smoothLO,y_smoothHI[::-1])
        ax.fill(x_new, y_smooth, rgsColors[si], alpha=0.1)

        
        rgrXX=np.array(range(len(rgrGridSpacing)))+0.5
        x_new = np.linspace(rgrXX.min(), rgrXX.max(),sSteps)
        rgrYY=rgrPolygons[si,:,1]
        f = interp1d(rgrXX, rgrYY, kind=sInterp)
        y_smoothLO=f(x_new)
        rgrYY=rgrPolygons[si,:,3]
        f = interp1d(rgrXX, rgrYY, kind=sInterp)
        y_smoothHI=f(x_new)
        x_new=np.append(x_new,x_new[::-1])
        y_smooth=np.append(y_smoothLO,y_smoothHI[::-1])
        ax.fill(x_new, y_smooth, rgsColors[si], alpha=0.5, label=rgsPeriods[si])

        x_new = np.linspace(rgrXX.min(), rgrXX.max(),sSteps)
        rgrYY=rgrPolygons[si,:,2]
        f = interp1d(rgrXX, rgrYY, kind=sInterp)
        y_smooth=f(x_new)
        ax.plot(x_new ,y_smooth,c=rgsColors[si], lw=2)

    xx = [0.5,1.5, 2.5, 3.5,4.5,5.5,6.5]
    labels = [ '12 km C',  '12 km',  '4 km', '2 km','1 km','500 m', '250 m']
    plt.xticks(xx, labels, rotation=20)
    ax.set_ylabel(PRcharacteristic[iYY[st]]+'\n'+rgsStats[st]+' [%]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0, 7)
    plt.axhline(y=0, c='k', lw=0.5)
    plt.ylim(Yrange[0],Yrange[1])

    ax.text(0.03,1.03, rgsLableABC[st]+') ', ha='left',va='bottom', \
            transform = ax.transAxes, fontname="Times New Roman Bold", fontsize=14)
    if st == 0:
        plt.legend(loc="upper right",
                   ncol=1, prop={'size':12})

# Plot the figure
sPlotFile=sPlotDir
sPlotName= 'MCS-conv-vs-strat-PR_dx_'+sGrid+'.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
