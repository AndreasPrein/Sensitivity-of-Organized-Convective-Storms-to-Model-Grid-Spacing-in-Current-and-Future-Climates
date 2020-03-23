#!/usr/bin/env python
''' Extreme_PR_CCS.py

   1) coarsen the precipitation to 12 km grid spacing
   2) Idenify the MCS precipitation area
   3) Set all outside precipitation to zero
   4) calculate 99 % of precipitation for each MCS

'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from pdb import set_trace as stop
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle

#===================================================
sGrid='12km'  # can be 'native' for native grid or '12km' for 12 km model grid

sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/CCS/Extreme_PR/'+sGrid+'/'

#===============================================================
#         Loop over the simulations during the processing
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

rgrGridSpacing=['12000','12000nc','4000','2000','1000','500','250']
rgrDXnative=[12000,12000,4000,2000,1000,500,250]
if sGrid == 'native':
    rgrDX=[12000,12000,4000,2000,1000,500,250]
if sGrid == '12km':
    rgrDX=[12000]*len(rgrGridSpacing)

TimeAccumulation=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37]
Percentiles=[98,99,99.5,99.9,99.95]

iSmoothKM=36000 # smoothing filter lenth in m
iPR_threshold=5  # precipitation threshold in mm/h
rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1

P99_accumulation=np.zeros((len(rgsSimulations), len(rgrGridSpacing), len(TimeAccumulation), len(Percentiles))); P99_accumulation[:]=np.nan
RankedPRmax=np.zeros((len(rgsSimulations),len(rgrGridSpacing),51*51)); RankedPRmax[:]=np.nan
for si in range(len(rgsSimulations)):
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        # fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Smooth-'+str(iSmoothKM)+'_PR-TH-'+str(iPR_threshold)+'.py'
        # if os.path.isfile(fname) == 0:
        i1km=int(1000/rgrDXnative[dx])
        if i1km ==0:
            i1km=1

        print '    dx = '+rgrGridSpacing[dx]
        iSkipHours=3.  # hours to skip at the beginning of the simulation
        sTmpData='/glade/p/mmm/c3we/Idealized_MCSs/data/WRF/'+rgsSimulations[si]+'/'
        #=============================================================
        
        print ('---loading---')
                
        if rgrGridSpacing[dx] == '250':
            iSubDir='Combined/'
        else:
            iSubDir=''
        # get all output files
        rgsFiles=glob.glob(sTmpData+rgrGridSpacing[dx]+'/'+iSubDir+'wrfout_d01_*')
        rgsFiles=np.array(rgsFiles)
        rgsFiles=np.sort(rgsFiles)
        rgsFiles=rgsFiles[int(iSkipHours*12):]
        # get the size of the domain
        ncfile = Dataset(rgsFiles[0])
        Lat=np.squeeze(ncfile.variables["XLAT"])

        rgrPR=np.zeros((Lat.shape[0],Lat.shape[1],len(rgsFiles))); rgrPR[:]=np.nan
        for fi in range(len(rgsFiles)):
            print '        read file: '+rgsFiles[fi]
            try:
                ncfile = Dataset(rgsFiles[fi])
                rgrPR[:,:,fi] = np.squeeze(ncfile.variables["TOTAL_PRECIP"])
            except:
                rgrPR[:,:,fi]=np.squeeze(ncfile.variables["RAINNC"])+\
                               np.squeeze(ncfile.variables["HAILNC"])+\
                               np.squeeze(ncfile.variables["GRAUPELNC"])+\
                               np.squeeze(ncfile.variables["SNOWNC"])+\
                               np.squeeze(ncfile.variables["RAINC"])
        rgrPR=rgrPR*12.

        if sGrid == 'native':
            iSmooth=int(iSmoothKM/rgrDX[dx])
            rgrPR_smooth=scipy.ndimage.uniform_filter(rgrPR[:,:,:],[iSmooth,iSmooth,3])
        elif sGrid == '12km':
            if rgrDXnative[dx] != 12000:
                # bring data to 12 km grid
                iRatio=12000/rgrDXnative[dx]
                rgrPRcoarse=np.zeros((int(rgrPR.shape[0]/iRatio),int(rgrPR.shape[1]/iRatio),rgrPR.shape[2])); rgrPRcoarse[:]=np.nan
                for la in range(rgrPRcoarse.shape[0]):
                    for lo in range(rgrPRcoarse.shape[1]):
                        rgrPRcoarse[la,lo,:]=np.mean(rgrPR[la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio,:], axis=(0,1))
                rgrPR=rgrPRcoarse
            iSmooth=int(iSmoothKM/12000.)                    
            rgrPR_smooth=scipy.ndimage.uniform_filter(rgrPR[:,:,:],[iSmooth,iSmooth,3])
                
        # threshold the precipitation
        rgiTH_precip=(rgrPR_smooth >= 5)
        rgrPR_thresholded=rgrPR[rgiTH_precip == False]=0
        rgiObjectsUD, nr_objectsUD = ndimage.label(rgiTH_precip,structure=rgiObj_Struct)
        # sort the objects according to their size
        rgiVolObj=np.array([np.sum(rgiObjectsUD == ob+1) for ob in range(nr_objectsUD)])
        rgiObSize=np.array([np.where(np.sort(rgiVolObj)[::-1][ob] == rgiVolObj)[0][0] for ob in range(nr_objectsUD)])

        print '    Calculate object properties'
        if nr_objectsUD >= 1:
            grObject={}
            # only look at the largest object
            for ob in [rgiObSize[0]]:
                rgrObAct=np.copy(rgrPR)
                rgrObAct[rgiObjectsUD != (ob+1)]=0           

                # Does the object hit the boundary?
                rgiObjActSel=np.array(rgiObjectsUD == (ob+1)).astype('float')
                rgiBoundary=(np.sum(rgiObjActSel[0,:,:], axis=0)+np.sum(rgiObjActSel[-1,:,:], axis=0)+np.sum(rgiObjActSel[:,0,:], axis=0)+np.sum(rgiObjActSel[:,-1,:], axis=0) != 0)
                rgrObAct[:,:,rgiBoundary]=np.nan

                # loop over accumulations and calculate 99 percentile vaule
                for ac in range(len(TimeAccumulation)):
                    PR_accumulation=scipy.ndimage.uniform_filter(rgrObAct[:,:,:],[0,0,TimeAccumulation[ac]])*TimeAccumulation[ac]/12.
                    if np.nanpercentile(PR_accumulation, 99.9) == 0:
                        print('     !!! Zero Precipitation !!!')
                        continue
                    P99_accumulation[si,dx,ac,:]=np.nanpercentile(PR_accumulation, Percentiles)
                # search the maximum hourly precipitation timeslice
                HouyrlyPR=scipy.ndimage.uniform_filter(rgrObAct[:,:,:],[0,0,13])*13./12
                try:
                    TTmax=np.where(np.nanmax(HouyrlyPR) == HouyrlyPR)[2][0]
                    RankedPRmax[si,dx,:]=np.sort(HouyrlyPR[:,:,TTmax].flatten())
                except:
                    continue


SaveFile=sDataOut+'CCS-PR_Smooth-'+str(iSmoothKM)+'_PR-TH-'+str(iPR_threshold)+'_20191227.npz'
print('    Save: '+SaveFile)
np.savez(SaveFile,
         P99_accumulation=P99_accumulation,
         RankedPRmax=RankedPRmax,
         TimeAccumulation=TimeAccumulation,
         Percentiles=Percentiles,
         rgrGridSpacing=rgrGridSpacing,
         rgsSimulations=rgsSimulations)
stop()

# Plot the data
P99_accumulation[1,1,:,:]=np.nan
pe=1
CCS=((np.nanmean(P99_accumulation[-10:,:,:,pe], axis=0)-np.nanmean(P99_accumulation[:10,:,:,pe], axis=0))/np.nanmean(P99_accumulation[:10,:,:,pe], axis=0))*100
plt.contourf(CCS.transpose(), levels=np.linspace(-30,30,11), cmap='coolwarm', extend='both'); plt.show()

RankedPRmax[1,1,:]=np.nan
plt.plot(np.nanmean(RankedPRmax[:10,0,:], axis=0), np.nanmean(RankedPRmax[-10:,0,:], axis=0), c='r')
plt.plot(np.nanmean(RankedPRmax[:10,2,:], axis=0), np.nanmean(RankedPRmax[-10:,2,:], axis=0), c='b')
plt.plot(np.nanmean(RankedPRmax[:10,6,:], axis=0), np.nanmean(RankedPRmax[-10:,6,:], axis=0), c='k')
plt.plot([0,120],[0,120], c='k', ls='--'); plt.show()
