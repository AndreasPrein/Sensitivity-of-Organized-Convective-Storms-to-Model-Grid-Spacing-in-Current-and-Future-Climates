#!/usr/bin/env python
''' MCS-Tracking.py

   The goal of this program is to imitate MODE-TD in python.

   This program reads in 5 min. precipitation data.
   The data is smoothed in 3D and a precipitation threshold is applied.
   Then the resulting precipitation objects are tracked over space and time.
   The location and the track of the objects are stored.

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

sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/MCS-Characteristics/'+sGrid+'/'

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

iSmoothKM=36000 # smoothing filter lenth in m
iPR_threshold=5  # precipitation threshold in mm/h
rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1

for si in range(len(rgsSimulations)):
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Smooth-'+str(iSmoothKM)+'_PR-TH-'+str(iPR_threshold)+'.p'
        if os.path.isfile(fname) == 0:
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
                for ob in range(nr_objectsUD):
                    rgrObAct=np.copy(rgrPR)
                    rgrObAct[rgiObjectsUD != (ob+1)]=0

                    # Does the object hit the boundary?
                    rgiObjActSel=np.array(rgiObjectsUD == (ob+1)).astype('float')
                    rgiBoundary=(np.sum(rgiObjActSel[0,:,:], axis=0)+np.sum(rgiObjActSel[-1,:,:], axis=0)+np.sum(rgiObjActSel[:,0,:], axis=0)+np.sum(rgiObjActSel[:,-1,:], axis=0) != 0)
                    rgrObAct[:,:,rgiBoundary]=np.nan
                    rgiObjActSel[:,:,rgiBoundary]=np.nan
                    rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(rgrObAct[:,:,tt]) for tt in range(rgiObjectsUD.shape[2])])
                    rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(rgiObjectsUD.shape[2]-1)])*12.*(rgrDX[dx]/1000.)
                    rgrPR_Vol=(np.array([np.sum(rgrObAct[:,:,tt]) for tt in range(rgiObjectsUD.shape[2])])/(12.*60.*5.))*rgrDX[dx]**2
                    rgrPR_Max=np.array([np.max(rgrObAct[:,:,tt]) for tt in range(rgiObjectsUD.shape[2])])
                    for tt in range(rgiObjectsUD.shape[2]):
                        if np.sum(rgiObjectsUD[:,:,tt] == (ob+1)) >0:
                            PR_perc=np.percentile(rgrObAct[:,:,tt][rgiObjectsUD[:,:,tt] == (ob+1)], range(101))
                            PR_mean=np.mean(rgrObAct[:,:,tt][rgiObjectsUD[:,:,tt] == (ob+1)])
                        else:
                            PR_mean=np.nan
                            PR_perc=np.array([np.nan]*101)
                        if tt == 0:
                            rgrPR_Percentiles=PR_perc[None,:]
                            rgrPR_Mean=PR_mean
                        else:
                            rgrPR_Percentiles=np.append(rgrPR_Percentiles,PR_perc[None,:], axis=0)
                            rgrPR_Mean=np.append(rgrPR_Mean,PR_mean)

                    rgrSize=np.array([np.sum(rgiObjActSel[:,:,tt] == (ob+1)) for tt in range(rgiObjectsUD.shape[2])])*(rgrDX[dx]/1000.)**2
                    rgrSize[(rgrSize == 0)]=np.nan

                    rgrPRfootprint=np.nansum(rgrObAct, axis=2)

                    grAct={'rgrMassCent':rgrMassCent, 
                           'rgrObjSpeed':rgrObjSpeed,
                           'rgrPR_Vol':rgrPR_Vol,
                           'rgrPR_Percentiles':rgrPR_Percentiles,
                           'rgrPR_Max':rgrPR_Max,
                           'rgrPR_Mean':rgrPR_Mean,
                           'rgrSize':rgrSize,
                           'rgrPRfootprint':rgrPRfootprint}
                    try:
                        grObject[str(np.where(rgiObSize == ob)[0][0])]=grAct
                    except:
                        continue
                pickle.dump( grObject, open( fname, "wb" ) )

