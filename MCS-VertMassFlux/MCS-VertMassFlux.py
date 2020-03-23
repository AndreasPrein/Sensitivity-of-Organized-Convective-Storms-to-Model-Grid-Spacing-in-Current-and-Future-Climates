#!/usr/bin/env python
''' MCS-VertMassFlux.py

   This program calculates bulk vertical mass flux in an MCS at various grid spacings

   This program reads in 5 min.
      precipitation
      vertical windspeed
      temperature
      pressure

   We identify the MCS according to the precipitation > 0.1 mm/h

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
Stat='MEAN' # this is either 'mean' for mean flux or '' for sum flux
iSIM=>>SIM<<-1
sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/MCS-VertMassFlux/'+sGrid+'/'
if not os.path.exists(sDataOut):
    os.makedirs(sDataOut)

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
rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
iSkipHours=3.  # hours to skip at the beginning of the simulation

for si in [iSIM]: #range(len(rgsSimulations)):
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Smooth-'+str(iSmoothKM)+'_'+sGrid+'-grid'+Stat+'.pkl'
        if os.path.isfile(fname) == 0:
            i1km=int(1000/rgrDXnative[dx])
            if i1km ==0:
                i1km=1
    
            print '    dx = '+rgrGridSpacing[dx]
            
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
            rgrRho=np.zeros((95,Lat.shape[0],Lat.shape[1],len(rgsFiles))); rgrRho[:]=np.nan
            rgrW=np.zeros((96,Lat.shape[0],Lat.shape[1],len(rgsFiles))); rgrW[:]=np.nan
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
                rgrP=np.squeeze(ncfile.variables["PB"])+np.squeeze(ncfile.variables["P"])
                rgrT=(np.squeeze(ncfile.variables["T"])+300)* (rgrP/100000.)**0.2854
                rgrRho[:,:,:,fi] = rgrP/(287.05*rgrT)
                rgrW[:,:,:,fi] = np.squeeze(ncfile.variables["W"])
            rgrPR=rgrPR*12.
            if sGrid == 'native':
                iSmooth=int(iSmoothKM/rgrDX[dx])
                rgrPR_smooth=scipy.ndimage.uniform_filter(rgrPR[:,:,:],[iSmooth,iSmooth,3])
            elif sGrid == '12km':
                if rgrDXnative[dx] != 12000:
                    # bring data to 12 km grid
                    iRatio=12000/rgrDXnative[dx]
                    rgrPRcoarse=np.zeros((int(rgrPR.shape[0]/iRatio),int(rgrPR.shape[1]/iRatio),rgrPR.shape[2])); rgrPRcoarse[:]=np.nan
                    rgrRhocoarse=np.zeros((95,int(rgrPR.shape[0]/iRatio),int(rgrPR.shape[1]/iRatio),rgrPR.shape[2])); rgrRhocoarse[:]=np.nan
                    rgrWcoarse=np.zeros((96,int(rgrPR.shape[0]/iRatio),int(rgrPR.shape[1]/iRatio),rgrPR.shape[2])); rgrWcoarse[:]=np.nan
                    for la in range(rgrPRcoarse.shape[0]):
                        for lo in range(rgrPRcoarse.shape[1]):
                            rgrPRcoarse[la,lo,:]=np.mean(rgrPR[la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio,:], axis=(0,1))
                            rgrRhocoarse[:,la,lo,:]=np.mean(rgrRho[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio,:], axis=(1,2))
                            rgrWcoarse[:,la,lo,:]=np.mean(rgrW[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio,:], axis=(1,2))
                    rgrPR=rgrPRcoarse
                    rgrRho=rgrRhocoarse
                    rgrW=rgrWcoarse
                iSmooth=int(iSmoothKM/12000.)                    
                rgrPR_smooth=scipy.ndimage.uniform_filter(rgrPR[:,:,:],[iSmooth,iSmooth,3])
            rgrW=((rgrW[:-1,:]+rgrW[1:,:])/2.)
            # threshold the precipitation ratio
            rgiTH_PR=(rgrPR_smooth >= 0.1)
            rgrPR_thresholded=rgrPR[rgiTH_PR == False]=0
            rgiObjectsUD, nr_objectsUD = ndimage.label(rgiTH_PR,structure=rgiObj_Struct)
            # sort the objects according to their size
            rgiVolObj=np.array([np.sum(rgiObjectsUD == ob+1) for ob in range(nr_objectsUD)])
            rgiObSize=np.array([np.where(np.sort(rgiVolObj)[::-1][ob] == rgiVolObj)[0][0] for ob in range(nr_objectsUD)])

            print '    Calculate object properties'
            if nr_objectsUD >= 1:
                grObject={}
                for ob in range(nr_objectsUD):
                    rgrObAct=np.copy(rgrPR)
                    rgrObAct[rgiObjectsUD != (ob+1)]=0
                    rgrRho_obj=np.copy(rgrRho)
                    rgrRho_obj[:, (rgiObjectsUD != (ob+1))]=np.nan
                    rgrW_obj=np.copy(rgrW)
                    rgrW_obj[:, (rgiObjectsUD != (ob+1))]=np.nan

                    # Does the object hit the boundary?
                    rgiObjActSel=np.array(rgiObjectsUD == (ob+1)).astype('float')
                    rgiBoundary=(np.sum(rgiObjActSel[0,:,:], axis=0)+np.sum(rgiObjActSel[-1,:,:], axis=0)+np.sum(rgiObjActSel[:,0,:], axis=0)+np.sum(rgiObjActSel[:,-1,:], axis=0) != 0)
                    rgrRho_obj[:,:,:,rgiBoundary]=np.nan
                    rgrW_obj[:,:,:,rgiBoundary]=np.nan
                    rgrObAct[:,:,rgiBoundary]=np.nan
                    rgiObjActSel[:,:,rgiBoundary]=np.nan

                    # bulk mass flux at each level
                    MassFLX=rgrRho_obj*rgrW_obj

                    Upward=np.copy(MassFLX)
                    Upward[rgrW_obj < 0]=np.nan
                    if Stat == 'MEAN':
                        Upw_FLX=np.nanmean(Upward, axis=(1,2))*rgrDX[dx]**2
                    else:
                        Upw_FLX=np.nansum(Upward, axis=(1,2))*rgrDX[dx]**2

                    Downward=np.copy(MassFLX)
                    Downward[rgrW_obj > 0]=np.nan
                    if Stat == 'MEAN':
                        Dnw_FLX=np.nanmean(Downward, axis=(1,2))*rgrDX[dx]**2
                    else:
                        Dnw_FLX=np.nansum(Downward, axis=(1,2))*rgrDX[dx]**2

                    grAct={'Upw_FLX':Upw_FLX, 
                           'Dnw_FLX':Dnw_FLX,
                           'levels':range(125,95*250+125,250)}

                    try:
                        grObject[str(np.where(rgiObSize == ob)[0][0])]=grAct
                    except:
                        continue
                pickle.dump(grObject, open( fname, "wb" ) )

