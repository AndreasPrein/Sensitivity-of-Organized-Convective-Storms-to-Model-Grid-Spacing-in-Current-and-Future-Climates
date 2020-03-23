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

fig = plt.figure(figsize=(18,6))
plt.rcParams.update({'font.size': 12})
gs1 = gridspec.GridSpec(1,3)
gs1.update(left=0.05, right=0.45,
           bottom=0.10, top=0.93,
           wspace=0.15, hspace=0.3)

rgsCol=['#1f78b4','#e31a1c']
rgsPeriods=['CUR','FUT']
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
            
    plt.title(rgsLableABC[va]+') ') #, fontsize=16)
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
gs1 = gridspec.GridSpec(2,2)
gs1.update(left=0.52, right=0.98,
           bottom=0.10, top=0.93,
           wspace=0.15, hspace=0.4)
XX=[0,1,0,1]
YY=[0,0,1,1]
for va in range(4):
    ax = plt.subplot(gs1[YY[va],XX[va]])
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
            text(0.05, 1.,rgsLableABC[va+3]+') P='+str(rMWU)[:4],
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform = ax.transAxes, fontsize=14)


        n, bins, patches = plt.hist(rgrXXnew, rgiBins, normed=0, facecolor=rgsCol[sim], alpha=0.5, zorder=0, label=rgsPeriods[sim])

    ax.set_xlabel(sLabel)
    if XX[va] == 0:
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
