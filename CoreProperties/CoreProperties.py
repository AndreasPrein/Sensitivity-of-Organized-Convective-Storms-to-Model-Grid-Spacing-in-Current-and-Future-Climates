#!/usr/bin/env python
''' CoreProperties.py


   This program defines convective up- and downdrafts
   as objects and saves them into a netcdf file.
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
sGrid='12km'  # can be 'native' for native grid or '12km' for 12 km model grid
sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/3D_Cores/'
iSIM=>>SIM<<-1

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
if sGrid == 'native':
    rgrDX=[12000,12000,4000,2000,1000,500,250]
if sGrid == '12km':
    rgrDX=[12000]*len(rgrGridSpacing)
rgsMetric=['Wmean','WP95','Wdepth','Wwidth','WwidthMax','Wvolume']

iSkipHours=3.  # hours to skip at the beginning of the simulation
rgiMinVol= 4 # minimum number of gridcells in the core
SIGMA=0. # sigma used in Gausian Filter
iSmooth=36000 # smoothing filter lenth in m
iRan=50  # number of random objects per time steps
DraftMinSpeed=3

rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
for si in [iSIM]: #range(len(rgsSimulations))[6:]:
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Sigm-'+str(SIGMA)+'_Random-'+str(iRan)+'_MinSpeed-'+str(DraftMinSpeed)+'.npz'
        if os.path.isfile(fname) == 0:
            iSmooth=int(iSmooth/rgrDX[dx])
            i1km=int(1000/rgrDX[dx])
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
            
            # rgrVV=np.zeros((96,Lat.shape[0],Lat.shape[1],len(rgsFiles)))# ; rgrVV[:]=np.nan
            # rgrZ=np.zeros((95,Lat.shape[0],Lat.shape[1],len(rgsFiles)))# ; rgrVV[:]=np.nan
            # rgrPR=np.zeros((Lat.shape[0],Lat.shape[1],len(rgsFiles))); rgrPR[:]=np.nan
            # for fi in range(len(rgsFiles)):

            height_v = np.linspace(0,24000-250,96); height_v=height_v/1000.
            height_z = np.linspace(125,24000,95); height_z=height_z/1000.

            rgiLevSel=((height_v > 0) & (height_v < 16))
            rgiLevSelZ=((height_z >= 0) & (height_z <= 16))
            for tt in range(len(rgsFiles)):
                print '        read file: '+rgsFiles[tt]
                ncfile = Dataset(rgsFiles[tt])
                rgrVV     = np.squeeze(ncfile.variables["W"]) # vertical velocity [m/s]
                try:
                    rgrZ     = np.squeeze(ncfile.variables["DBZ"])
                except:
                    continue
                rgrPR  = np.squeeze(ncfile.variables["TOTAL_PRECIP"])
                rgrPR=rgrPR*12.

                print '    time '+str(tt)
                #=============================================================
                #  GET THE CONVECTIVE AREA FROM THE PRECIP FILE
                rgrPR_smooth=scipy.ndimage.uniform_filter(rgrPR[:,:],[iSmooth,iSmooth])
                rgiPR=np.array((rgrPR_smooth > 2.5)).astype('int')
                rgrVVact=rgrVV[:,:,:]*rgiPR[None,:,:]
                rgrZact=rgrZ[:,:,:]*rgiPR[None,:,:]

                # cut out 12 km band around domain
                rgrVVact=rgrVVact[rgiLevSel,int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgrZact=rgrZact[rgiLevSelZ,int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgrPRact=rgrPR[int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                rgiPR=rgiPR[int(12000/rgrDX[dx]):-int(12000/rgrDX[dx]),int(12000/rgrDX[dx]):-int(12000/rgrDX[dx])]
                height_v_TMP=height_v[rgiLevSel]

                rgrObPorp=np.zeros((len(rgsMetric),2,iRan)); rgrObPorp[:]=np.nan
                if np.sum(rgiPR) == 0:
                    print '    No precipitation object found!'
                    print '    Skip this timestep'
                else:
                    # crop the the area with high PR
                    rgiAreaLon=[np.where(np.sum(rgiPR, axis=0) > 0)[0][0], np.where(np.sum(rgiPR, axis=0) > 0)[0][-1]]
                    rgiAreaLat=[np.where(np.sum(rgiPR, axis=1) > 0)[0][0], np.where(np.sum(rgiPR, axis=1) > 0)[0][-1]]
                    rgrVVact=rgrVVact[:,rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    rgrZact=rgrZact[:,rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    rgrPRact=rgrPRact[rgiAreaLat[0]:rgiAreaLat[1],rgiAreaLon[0]:rgiAreaLon[1]]
                    #=============================================================

                    print '    Smooth the W field and get objects'
                    # TEST=gaussian_filter(rgrVV,[0.25,2,2,0.05],0, truncate=3)
                    # https://stackoverflow.com/questions/16937158/extracting-connected-objects-from-an-image-in-python
                    if SIGMA == 0:
                        rgrSmoothed=np.copy(rgrVVact)
                    else:
                        rgrSmoothed=gaussian_filter(rgrVVact[:,:,:],[SIGMA,SIGMA,SIGMA],0, truncate=3)
                    rgrUpdrafts=np.copy(rgrSmoothed); rgrUpdrafts[rgrSmoothed <= DraftMinSpeed]=0
                    rgrDowndrafts=np.copy(rgrSmoothed); rgrDowndrafts[rgrSmoothed >= -DraftMinSpeed]=0; rgrDowndrafts=np.abs(rgrDowndrafts)
                    rgiObjectsUD, nr_objectsUD = ndimage.label(rgrUpdrafts,structure=rgiObj_Struct)
                    rgiObjectsDD, nr_objectsDD = ndimage.label(rgrDowndrafts,structure=rgiObj_Struct)

                    for ud in range(2):
                        if ud == 0:
                            rgrData=rgrUpdrafts[:,:,:]
                            rgiOb=rgiObjectsUD[:,:,:]
                            iOB=np.unique(rgiObjectsUD[:,:,:])[1:]
                        elif ud == 1:
                            rgrData=rgrDowndrafts[:,:,:]
                            rgiOb=rgiObjectsDD[:,:,:]
                            iOB=np.unique(rgiObjectsDD[:,:,:])[1:]
                        
                        if len(iOB) == 0:
                            continue
    
                        RANDOM=[random.randint(1,len(iOB)) for x in range(np.min([len(iOB),iRan*3]))]
                        try:
                            iOB=iOB[(np.array(RANDOM)-1)]
                        except:
                            iOB=iOB
        
                        jj=0
                        for ob in range(len(iOB)):
                            rgiOb_Act=(rgiOb == iOB[ob])
                            rgiOb2D=np.sum(rgiOb_Act, axis=0); rgiOb2D=np.where(rgiOb2D > 0)
                            if (np.sum(rgiOb_Act) >= rgiMinVol) & \
                               (np.mean(rgrPRact[:,:][rgiOb2D]) > 0.1) & \
                               (np.sum(np.sum(rgiOb_Act, axis=(1,2)) >= i1km) >=2) &\
                               (np.max(height_v_TMP[(np.sum(rgiOb_Act, axis=(1,2)) > 0)]) > 0) & \
                               (np.min(height_v_TMP[(np.sum(rgiOb_Act, axis=(1,2)) > 0)]) < 16) &\
                               (np.mean(rgrZact[:,:,:][rgiOb_Act]) >= 20):
                                rgrObPorp[rgsMetric.index('Wmean'),ud,jj]=np.mean(rgrData[rgiOb_Act])
                                rgrObPorp[rgsMetric.index('WP95'),ud,jj]=np.percentile(rgrData[rgiOb_Act],95)
                                rgiVert=np.sum(rgiOb_Act, axis=(1,2))
                                rgrObPorp[rgsMetric.index('Wdepth'),ud,jj]=np.sum(rgiVert >= i1km)*0.25
                                rgrObPorp[rgsMetric.index('Wwidth'),ud,jj]=np.mean(rgiVert[rgiVert >= i1km])*(rgrDX[dx]/1000.)**2
                                rgrObPorp[rgsMetric.index('WwidthMax'),ud,jj]=np.max(rgiVert)*(rgrDX[dx]/1000.)**2
                                rgrObPorp[rgsMetric.index('Wvolume'),ud,jj]=np.sum(rgiOb_Act)*(0.25*(rgrDX[dx]/1000.)**2)
                                jj=jj+1
                                if jj == iRan:
                                    break
                if tt ==0:
                    rgrObAll=np.copy(rgrObPorp)
                else:
                    rgrObAll=np.append(rgrObAll,rgrObPorp,axis=2)
            # Save the data
            np.savez(fname,rgrObAll=rgrObAll,rgsMetric=rgsMetric)
        else:
            print 'Load '+fname

            # plt.contourf(rgrDowndrafts[15,:,:,6], np.linspace(-10,10), extend='both', cmap=cm.coolwarm); plt.contour(rgiObjectsDD[15,:,:,6], levels=[1], colors='k',linewiths=0.4);plt.show()
            # plt.contourf(rgrDowndrafts[:,450,400,:], np.linspace(-3,3,100), extend='both', cmap=cm.coolwarm); plt.contour(rgiObjectsDD[:,450,400,:], levels=[0,1,100,100000], colors='k',linewidths=1);plt.show()
