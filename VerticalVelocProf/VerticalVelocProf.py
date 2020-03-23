#!/usr/bin/env python
''' VerticalVelocProf.py


   This program defines convective up- and downdrafts
   as objects and calculates statistics on these cores
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


#===================================================

def fnGetObjects2D(Data,Threshold,SmoothArea):
    # function to derive seperated objects in a 2D field
    rgiObj_Struct=np.zeros((3,3)); rgiObj_Struct[:,:]=1
    Data_smooth=scipy.ndimage.uniform_filter(Data[:,:],[SmoothArea,SmoothArea])
    rgiTH_Data=(Data_smooth >= Threshold)
    Data_thresholded=Data[rgiTH_Data == False]=0
    rgiObjectsUD, nr_objectsUD = ndimage.label(rgiTH_Data,structure=rgiObj_Struct)

    return rgiObjectsUD, nr_objectsUD

#===================================================
sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/Dynamics/'
iSIM=8 #>>SIM<<

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
rgrDX=[12000,12000,4000,2000,1000,500,250]

rgiMinVol= 4 # minimum number of gridcells in the core
SIGMA=0. # sigma used in Gausian Filter
iRan=50  # number of random objects per time steps
iSmooth=36000 # smoothing filter lenth in m
rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
rgsMetric=['Wmean','W05','W95','MassFLX','NrCores','CoreArea']
for si in [iSIM]: #range(len(rgsSimulations))[6:]:
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Sigm-'+str(SIGMA)+'_Dynamics.npz'
        if os.path.isfile(fname) == 0:
            iSmooth=int(iSmooth/rgrDX[dx])
            i1km=int(1000/rgrDX[dx])
            if i1km ==0:
                i1km=1
    
            print '    dx = '+rgrGridSpacing[dx]
            iSkipHours=4.  # hours to skip at the beginning of the simulation
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
            
            height_v = np.linspace(0,24000-250,96); height_v=height_v/1000.
            height_z = np.linspace(125,24000,95); height_z=height_z/1000.
            rgiLevSel=((height_v > 0) & (height_v < 16))
            rgiLevSelZ=((height_z >= 0) & (height_z <= 16))

            for fi in range(len(rgsFiles)):
                print '        read file: '+rgsFiles[fi]
                ncfile = Dataset(rgsFiles[fi])
                rgrVV     = np.squeeze(ncfile.variables["W"]) # vertical velocity [m/s]
                try:
                    rgrZ      = np.squeeze(ncfile.variables["DBZ"])
                except:
                    continue
                rgrP                = np.squeeze(ncfile.variables["PB"])+np.squeeze(ncfile.variables["P"])
                rgrT                = (np.squeeze(ncfile.variables["T"])+300)* (rgrP/100000.)**0.2854
                rgrRoh              = rgrP/(287.05*rgrT)
                rgrWflx   = rgrRoh*np.mean([rgrVV[1:,:,:],rgrVV[:-1,:,:]], axis=0)
                try:
                    rgrPR       = np.squeeze(ncfile.variables["TOTAL_PRECIP"])
                except:
                    rgrPR=np.squeeze(ncfile.variables["RAINNC"])+\
                           np.squeeze(ncfile.variables["HAILNC"])+\
                           np.squeeze(ncfile.variables["GRAUPELNC"])+\
                           np.squeeze(ncfile.variables["SNOWNC"])+\
                           np.squeeze(ncfile.variables["RAINC"])
                rgrPR=rgrPR*12.

                #=============================================================
                #  GET THE CONVECTIVE AREA FROM THE PRECIP FILE
                rPR_tresh=5.
                rgiObjectsUD, nr_objectsUD=fnGetObjects2D(rgrPR,rPR_tresh,iSmooth)
                rgiVolObj=np.array([np.sum(rgiObjectsUD == ob+1) for ob in range(nr_objectsUD)])
                rgiObSize=np.array([np.where(np.sort(rgiVolObj)[::-1][ob] == rgiVolObj)[0][0] for ob in range(nr_objectsUD)])

                # solely focus on the largest object
                try:
                    rgiPR=(rgiObjectsUD == np.where(rgiObSize == 0)[0][0]+1)
                except:
                    continue
                rgrVVact=rgrVV*rgiPR[None,:,:]
                rgrZact=rgrZ*rgiPR[None,:,:]
                rgrWflxact=rgrWflx*rgiPR[None,:,:]

                # cut out 12 km band around domain
                rgrVVact=rgrVVact[rgiLevSel,int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgrZact=rgrZact[rgiLevSelZ,int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgrWflxact=rgrWflxact[rgiLevSelZ,int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgrPRact=rgrPR[int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgiPR=rgiPR[int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                height_v_TMP=height_v[rgiLevSel]

                rgrObPorp=np.zeros((len(rgsMetric),3,sum(rgiLevSel)))
                if np.sum(rgiPR) == 0:
                    print '        No precipitation object found!'
                    print '        Skip this timestep'
                else:
                    # crop the the area with high PR
                    rgiAreaLon=[np.where(np.sum(rgiPR, axis=0) > 0)[0][0], np.where(np.sum(rgiPR, axis=0) > 0)[0][-1]]
                    rgiAreaLat=[np.where(np.sum(rgiPR, axis=1) > 0)[0][0], np.where(np.sum(rgiPR, axis=1) > 0)[0][-1]]
                    rgrVVact=rgrVVact[:,rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    rgrZact=rgrZact[:,rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    rgrWflxact=rgrWflxact[:,rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    rgrPRact=rgrPRact[rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    #=============================================================

                    print '        Smooth the W field and get objects'
                    # TEST=gaussian_filter(rgrVV,[0.25,2,2,0.05],0, truncate=3)
                    # https://stackoverflow.com/questions/16937158/extracting-connected-objects-from-an-image-in-python
                    if SIGMA == 0:
                        rgrSmoothed=np.copy(rgrVVact)
                    else:
                        rgrSmoothed=gaussian_filter(rgrVVact,[SIGMA,SIGMA,SIGMA,0],0, truncate=3)

                    # rgrUpdrafts=np.copy(rgrSmoothed); rgrUpdrafts[rgrSmoothed <= 1.5]=0
                    # rgrDowndrafts=np.copy(rgrSmoothed); rgrDowndrafts[rgrSmoothed >= -1.5]=0; rgrDowndrafts=np.abs(rgrDowndrafts)
                    # rgiObjectsUD, nr_objectsUD = ndimage.label(rgrUpdrafts,structure=rgiObj_Struct)
                    # rgiObjectsDD, nr_objectsDD = ndimage.label(rgrDowndrafts,structure=rgiObj_Struct)

                    for ud in range(3):
                        rgrData=np.copy(rgrSmoothed)
                        if ud == 0:
                            # mean stats
                            rgrData[np.abs(rgrData) < 1.5] = np.nan
                        elif ud == 1:
                            # Downdraft stats
                            rgrData[rgrData > -1.5] = np.nan
                        elif ud == 2:
                            # Updraft stats
                            rgrData[rgrData < 1.5] = np.nan
                        rgrObj=np.copy(rgrData)
                        rgrObj[~np.isnan(rgrObj)]=1; rgrObj[np.isnan(rgrObj)]=0
                        rgiObjectsUD, nr_objectsUD = ndimage.label(rgrObj,structure=rgiObj_Struct)
                        if nr_objectsUD == 1:
                            continue
                        for ll in range(rgrData.shape[0]):
                            rgrObPorp[rgsMetric.index('Wmean'),ud,ll]=np.nanmean(rgrData[ll,:,:])
                            rgrObPorp[rgsMetric.index('W05'),ud,ll]=np.nanpercentile(rgrData[ll,:,:], 5)
                            rgrObPorp[rgsMetric.index('W95'),ud,ll]=np.nanpercentile(rgrData[ll,:,:], 95)
                            rgrObPorp[rgsMetric.index('MassFLX'),ud,ll]=np.nansum(rgrData[ll,:,:]*rgrWflxact[ll,:,:])*rgrDX[dx]**2
                            rgrObPorp[rgsMetric.index('NrCores'),ud,ll]=len(np.unique(rgiObjectsUD[ll,:,:]))
                            rgrObPorp[rgsMetric.index('CoreArea'),ud,ll]=(np.sum(rgrObj[ll,:,:]))*(rgrDX[dx]/1000.)**2
                if fi ==0:
                    rgrObAll=np.copy(rgrObPorp[:,:,:,None])
                else:
                    rgrObAll=np.append(rgrObAll,rgrObPorp[:,:,:,None],axis=3)
            # Save the data
            np.savez(fname,rgrObAll=rgrObAll,rgsMetric=rgsMetric)
        # else:
        #     # print 'Load '+fname
        #     # DATA=np.load(fname)
        #     # stop()

            # plt.contourf(rgrDowndrafts[15,:,:,6], np.linspace(-10,10), extend='both', cmap=cm.coolwarm); plt.contour(rgiObjectsDD[15,:,:,6], levels=[1], colors='k',linewiths=0.4);plt.show()
            # plt.contourf(rgrDowndrafts[:,450,400,:], np.linspace(-3,3,100), extend='both', cmap=cm.coolwarm); plt.contour(rgiObjectsDD[:,450,400,:], levels=[0,1,100,100000], colors='k',linewidths=1);plt.show()
