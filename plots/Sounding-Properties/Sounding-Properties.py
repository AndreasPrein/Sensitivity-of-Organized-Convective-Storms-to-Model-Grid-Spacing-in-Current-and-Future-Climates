#!/usr/bin/env python
'''
    File name: Sounding-Properties.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 11.11.2016
    Date last modified: 03.04.2018

    ##############################################################
    Purpos:

    1) We read in the sounding data that was used as boundary condition
       to start the idealized WRF simulations

    2) We calculate sounding indices

    3) These indices are plotted


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
from mpl_toolkits import basemap
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
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
import shapely.geometry
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
import wrf
import SkewT
from scipy.interpolate import interp1d

########################################
#                            Settings

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

SimLatC=[40,41.2,35.5,42.6,34.5,41.0,39.1,41.8,35.5,36.7]
SimLonC=[-102.5,-94.6,-97.2,-98.2,-96.4,-95.3,-97.4,-100.0,-98.3,-86.3]

SimLatP=[35.5, 41.4, 36.2, 39.9, 38.7,  40.6, 34.9, 36.8, 40.0, 46.2]
SimLonP=[-92, -94.7,-88.9,-86.2,-101.7,-94.1,-93.4,-96.4,-98.6,-97.3]


sDataDir='/glade/u/home/prein/papers/Idealized-MCSs/data/Soundings/'
rgsDataSets=['CTR','PGW']

sPlotDir='/glade/u/home/prein/papers/Idealized-MCSs/plots/MCS-Soundings/'

# set the font size
plt.rcParams.update({'font.size': 18})
rgsLableABC=list(string.ascii_lowercase)

########################################
# load the WRF grid
sCoorFile='/glade/work/prein/CONUS/wrfout_conus04_constants.nc'
sLon='XLONG'
sLat='XLAT'
ncid=Dataset(sCoorFile, mode='r') # open the netcdf
rgrLonW=np.squeeze(ncid.variables[sLon][:])
rgrLatW=np.squeeze(ncid.variables[sLat][:])
rgrSufAlt=np.squeeze(ncid.variables['HGT'][:])
ncid.close()


########################################
# Read in the sounding data
rgrPWall=np.zeros((len(rgsSimulations))); rgrPWall
rgrCINall=np.copy(rgrPWall)
rgrVS3all=np.copy(rgrPWall)
rgrVS6all=np.copy(rgrPWall)
rgrRHall=np.zeros((len(rgsSimulations),50,2)); rgrRHall
rgrTKall=np.zeros((len(rgsSimulations),50,2)); rgrTKall
rgrcCAPEall=np.zeros((len(rgsSimulations),201,2)); rgrcCAPEall
for so in range(len(rgsSimulations)):
    print '    Read: '+rgsSimulations[so]
    iFile=open(sDataDir+rgsSimulations[so]+'_WRFdata.pkl')
    grDataAct = pickle.load(iFile)
    iFile.close()
    rgrEU=grDataAct['rgrEU']
    rgrEV=grDataAct['rgrEV']
    rgrP=grDataAct['rgrP']
    rgrQV=grDataAct['rgrQV']
    rgrZorig=grDataAct['rgrZorig']
    rgrQS=grDataAct['rgrQS']
    rgrTK=grDataAct['rgrTK']
    rgrPotT=grDataAct['rgrPotT']
    rgrTD=grDataAct['rgrTD']
    rgrPSFC=grDataAct['rgrPSFC']
    rgrQ2=grDataAct['rgrQ2']
    rgrT2=grDataAct['rgrT2']
    rgrT2Pot=grDataAct['rgrT2Pot']
    rgrPWall[so]=grDataAct['rgrPW']

    # Cumulative CAPE, CAPE, and CIN
    # from skewt import SkewT

    wind_abs=((rgrEU**2+rgrEV**2)**0.5)
    sknt=wind_abs*1.94384
    drct=(np.arctan2(rgrEU/wind_abs, rgrEV/wind_abs))* 180/pi+180
    mydata=dict(zip(('hght','pres','temp','dwpt','sknt','drct'),(rgrZorig,rgrP/100.,rgrTK-273.15,rgrTD-273.15,sknt, drct)))
    S=SkewT.Sounding(soundingdata=mydata)
    # parcel=S.get_parcel('ml',depth=300)
    parcel=S.mixed_layer_parcel(depth=300)
    # parcel=S.most_unstable_parcel(depth=300)
    P_lcl,P_lfc,P_el,CAPE,CIN,cCAPE,rgrHGT=S.get_cape(*parcel)
    # S.plot_skewt()
    # plt.show()
    rgrcCAPEall[so,:,0]=cCAPE
    rgrcCAPEall[so,:,1]=rgrHGT
    rgrCINall[so]=CIN

    # calculate rel hum.
    from thermodynamics import  DTtoRH
    rgrRHall[so,:,0]=DTtoRH(rgrTD,rgrTK)
    rgrRHall[so,:,1]=rgrZorig

    rgrTKall[so,:,0]=rgrTK
    rgrTKall[so,:,1]=rgrZorig

    # calculate wind shear at 2 levels
    rgrVSallLev=((rgrEU[0]-rgrEU[:])**2+(rgrEV[0]-rgrEV[:])**2)**0.5
    f = interp1d(rgrZorig-rgrZorig[0]+20, rgrVSallLev)
    rgrVS3all[so]=f(3000)
    rgrVS6all[so]=f(6000)


# grSave={'rgrcCAPEall':rgrcCAPEall,
#         'rgrCINall':rgrCINall,
#         'rgrRHall':rgrRHall,
#         'rgrVS3all':rgrVS3all,
#         'rgrVS6all':rgrVS6all,
#         'rgrPWall':rgrPWall}
# fh = open('SoundingStats.pkl',"w")
# grSaveDat=pickle.dump(grSave,fh)
# fh.close()


# stop()
# iFile=open('SoundingStats.pkl')
# grSave = pickle.load(iFile)
# iFile.close()


# ##################################################
#        PLOT THE SOUNDING STATISTICS

fig = plt.figure(figsize=(18,10))
plt.rcParams.update({'font.size': 14})


rgsCol=['#1f78b4','#e31a1c']
rgsPeriods=['CUR','FUT']

# plot the timing when the MCS occured
gs1 = gridspec.GridSpec(1,1)
gs1.update(left=0.05, right=0.13,
           bottom=0.42, top=0.93,
           wspace=0.15, hspace=0.3)
ax = plt.subplot(gs1[0,0])

DateAxis=pd.date_range(start='6/1/2018', end='8/31/2018')
for cf in range(2):
    if cf == 0:
        Year=np.array([rgsSimulations[ii][3:7] for ii in range(10)])
        Month=np.array([rgsSimulations[ii][8:10] for ii in range(10)])
        Day=np.array([rgsSimulations[ii][11:13] for ii in range(10)])
    if cf == 1:
        Year=np.array([rgsSimulations[ii][3:7] for ii in range(10,20,1)])
        Month=np.array([rgsSimulations[ii][8:10] for ii in range(10,20,1)])
        Day=np.array([rgsSimulations[ii][11:13] for ii in range(10,20,1)])
    for yy in range(10):
        Yloc=np.where((int(Month[yy]) == DateAxis.month) & (int(Day[yy]) == DateAxis.day))[0][0]
        ax.text(cf,Yloc,str(yy+1), fontsize=14, fontweight='bold', va='center',ha='center', color=rgsCol[cf])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(-0.5,1.5)
ax.set_ylabel('Days in JJA')
ax.set_ylim(0, len(DateAxis))

ax.set_yticks([0, 30, 61])
ax.set_yticklabels(['June','July','Aug.'])

ax.set_xticks([0, 1])
ax.set_xticklabels(['CUR','FUT'])

plt.title(rgsLableABC[0]+') ') #, fontsize=16)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import math
gs1 = gridspec.GridSpec(1,1)
gs1.update(left=0.16, right=0.45,
           bottom=0.42, top=0.93,
           wspace=0.15, hspace=0.3)
ax = plt.subplot(gs1[0,0])

map = Basemap(llcrnrlon=-105,llcrnrlat=30,urcrnrlon=-80,urcrnrlat=47,
        projection='lcc',lat_1=35,lat_2=45,lon_0=-95)
# load the shapefile, use the name 'states'
map.readshapefile('/glade/u/home/prein/ShapeFiles/US-States/cb_2013_us_state_5m', name='states', drawbounds=True)
# ax.axis('off')

# draw parallels.
parallels = np.arange(0.,90,5.)
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,5.)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

xx, yy = map(SimLonC, SimLatC)
for lo in range(len(SimLonC)):
    plt.text(xx[lo], yy[lo], str(lo+1),fontsize=17,fontweight='bold',ha='center',va='center',color=rgsCol[0])

xx, yy = map(SimLonP, SimLatP)
for lo in range(len(SimLonP)):
    plt.text(xx[lo], yy[lo], str(lo+1),fontsize=17,fontweight='bold',ha='center',va='center',color=rgsCol[1])
plt.title(rgsLableABC[1]+') ') #, fontsize=16)

# plot the soundings
gs1 = gridspec.GridSpec(1,3)
gs1.update(left=0.51, right=0.98,
           bottom=0.42, top=0.93,
           wspace=0.15, hspace=0.3)
rgrZcom=np.linspace(0,19,101)
# Start with the vertical profile of CAPE and RH
for va in range(3):
    ax = plt.subplot(gs1[0,va])
    if va == 0:
        rgrXXact=rgrcCAPEall[:,:,0]/1000.
        rgrYYact=rgrcCAPEall[:,:,1]/1000.
        sLabel='cCAPE [kJ kg$^{-1}$]'
    elif va == 1:
        rgrXXact=rgrRHall[:,:,0]
        rgrYYact=rgrRHall[:,:,1]/1000.
        sLabel='RH [%]'
    elif va == 2:
        rgrXXact=rgrTKall[:,:,0]-273.15
        rgrYYact=rgrTKall[:,:,1]/1000.
        sLabel='T diff. [$^{\circ}$C]'
    rgrYYact=rgrYYact[:,:]-rgrYYact[:,0][:,None]
    # remgrid the data to common high coordinate
    rgrYYremapp=np.zeros((rgrYYact.shape[0], len(rgrZcom)))
    for ex in range(rgrYYact.shape[0]):
        f = interp1d(rgrYYact[ex,:], rgrXXact[ex,:])
        rgrYYremapp[ex,:]=f(rgrZcom)

    for sim in range(2):
        if sim == 0:
            rgiSimact=['CTRL' in rgsSimulations[ii] for ii in range(len(rgsSimulations))]
        if sim == 1:
            rgiSimact=['PGW' in rgsSimulations[ii] for ii in range(len(rgsSimulations))]
        rgrXXact1=rgrYYremapp[rgiSimact,:]
        # rgrYYact1=rgrYYact[rgiSimact,:]
        
        # plot average sounding - needs interpolation!
        rgrPercentiles=np.percentile(rgrXXact1[:,:],(0,25,50,75,100),axis=0)
        if va <2:
            for ex in range(sum(rgiSimact)):
                plt.plot(rgrXXact1[ex,:],rgrZcom, c=rgsCol[sim],lw=1,alpha=0.2)
            rgrYYnew=np.append(rgrZcom,rgrZcom[::-1])
            rgrXXnew=np.append(rgrPercentiles[1,:],rgrPercentiles[3,:][::-1])
            ax.fill(rgrXXnew, rgrYYnew, rgsCol[sim], alpha=0.5, label=rgsPeriods[sim])
            plt.plot(np.mean(rgrXXact1[:,:], axis=0),rgrZcom, c=rgsCol[sim],lw=2,alpha=1)
        else:
            # for temperature show difference
            if sim == 0:
                rgrSortCUR=np.sort(rgrXXact1[:,:], axis=0)
            else:
                rgrSortPGW=np.sort(rgrXXact1[:,:], axis=0)
                rgrPercentiles=np.percentile(rgrSortPGW-rgrSortCUR,(0,25,50,75,100),axis=0)
                rgrYYnew=np.append(rgrZcom,rgrZcom[::-1])
                rgrXXnew=np.append(rgrPercentiles[1,:],rgrPercentiles[3,:][::-1])
                ax.fill(rgrXXnew, rgrYYnew, 'k', alpha=0.5, label=rgsPeriods[sim])
                plt.plot(np.mean(rgrSortPGW-rgrSortCUR, axis=0),rgrZcom, c='k',lw=2,alpha=1)
            
    plt.title(rgsLableABC[va+2]+') ') #, fontsize=16)
    ax.set_xlabel(sLabel)
    if va == 0:
        ax.set_ylabel('height above surface [km]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if va == 0:
        ax.set_xlim(0, 3.5)
    if va == 1:
        ax.set_xlim(50, 100)
    ax.set_ylim(0, 17)
    if va == 0:
        plt.legend(loc="lower right",\
                   ncol=1, prop={'size':12})


# PLOT THE HISTOGRAMS
gs1 = gridspec.GridSpec(1,4)
gs1.update(left=0.05, right=0.98,
           bottom=0.07, top=0.31,
           wspace=0.15, hspace=0.4)
for va in range(4):
    ax = plt.subplot(gs1[0,va])
    if va == 0:
        rgrXXact=rgrCINall
        sLabel='CIN [J kg$^{-1}$]'
    elif va ==1:
        rgrXXact=rgrPWall
        sLabel='PW [mm]'
    elif va ==2:
        rgrXXact=rgrVS3all
        sLabel='Shear 0-3 km [m s$^{-2}$]'
    elif va ==3:
        rgrXXact=rgrVS6all
        sLabel='Shear 0-6 km [m s$^{-2}$]'
    rgiBins=np.linspace(min(rgrXXact),max(rgrXXact),10)
    for sim in range(2):
        if sim == 0:
            rgiSimact=['CTRL' in rgsSimulations[ii] for ii in range(len(rgsSimulations))]
        if sim == 1:
            rgiSimact=['PGW' in rgsSimulations[ii] for ii in range(len(rgsSimulations))]
        rgrXXnew=rgrXXact[rgiSimact]
        if sim == 0:
            CTRL=rgrXXnew
        else:
            rMWU=scipy.stats.mannwhitneyu(CTRL, rgrXXnew)[1]
            # plt.title(rgsLableABC[va+3]+') '+str(rMWU)[:4]) #, fontsize=16)
            text(0.05, 1.,rgsLableABC[va+5]+') P='+str(rMWU)[:4],
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform = ax.transAxes, fontsize=14)


        n, bins, patches = plt.hist(rgrXXnew, rgiBins, normed=0, facecolor=rgsCol[sim], alpha=0.5, zorder=0, label=rgsPeriods[sim])

    ax.set_xlabel(sLabel)
    if va == 0:
        ax.set_ylabel('frequency [counts]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Plot the figure
sPlotFile=sPlotDir
sPlotName= 'Input-Sounding-Properties.pdf'
if os.path.isdir(sPlotFile) != 1:
    subprocess.call(["mkdir","-p",sPlotFile])
print '        Plot map to: '+sPlotFile+sPlotName
fig.savefig(sPlotFile+sPlotName)

stop()
