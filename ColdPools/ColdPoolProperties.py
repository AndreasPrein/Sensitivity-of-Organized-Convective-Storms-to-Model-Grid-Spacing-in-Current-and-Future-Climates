#!/usr/bin/env python
''' ColdPoolProperties.py


   Here we read in idealized MCS simulation data and calculate coldpool
   properties simular to Feng et al.(2015)

   These are used to calculate coldpool depth, extend, volume dynamically

   The results are stored for later plotting
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
from thermodynamics import Theta
import pickle

#===================================================
sGrid='12km'  # can be 'native' for native grid or '12km' for 12 km model grid
sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/Coldpools/'+sGrid+'/'
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
rgrDXnative=[12000,12000,4000,2000,1000,500,250]
if sGrid == 'native':
    rgrDX=[12000,12000,4000,2000,1000,500,250]
if sGrid == '12km':
    rgrDX=[12000]*len(rgrGridSpacing)
rgsMetricCPall=['Volume','MeanDepth','P95Depth', 'Area']

iSkipHours=0  # hours to skip at the beginning of the simulation
rgiMinVol= 36000*36000*4  # minimum volume of cold pool size [36 km x 36 km * 20 min]
Smooth=12000 # smoothing filter lenth in m
DraftMinSpeed=3
Levels=16 # number of model levels that should be read in

rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1
for si in [iSIM]: # #si in [iSIM]: #range(len(rgsSimulations))[6:]:
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_Smooth-'+str(Smooth/1000)+'km.pkl'
        print fname
        DATA={}
        if os.path.isfile(fname) == 0:
            iSmooth=int(Smooth/rgrDX[dx])
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

            height_v = np.linspace(0,24000-250,96); height_v=height_v/1000.
            height_z = np.linspace(125,24000,95); height_z=height_z/1000.

            rgiLevSel=((height_v > 0) & (height_v < 16))
            rgiLevSelZ=((height_z >= 0) & (height_z <= 16))

            ncfile = Dataset(rgsFiles[0])
            Lat=np.squeeze(ncfile.variables["XLAT"])
            ncfile.close()
            if sGrid == 'native':
                Boyancy_Sim=np.zeros((len(rgsFiles),Levels,Lat.shape[0],Lat.shape[1]))  
                CP_Intensity=np.zeros((len(rgsFiles),Lat.shape[0],Lat.shape[1]))
            else:
                Boyancy_Sim=np.zeros((len(rgsFiles),Levels,51,51))  
                CP_Intensity=np.zeros((len(rgsFiles),51,51))
            for tt in range(len(rgsFiles)):
                print '        read file: '+rgsFiles[tt]
                ncfile = Dataset(rgsFiles[tt])

                rgrP                = np.squeeze(ncfile.variables["PB"][:,:Levels,:,:])+np.squeeze(ncfile.variables["P"][:,:Levels,:,:])
                rgrT                = (np.squeeze(ncfile.variables["T"][:,:Levels,:,:])+300)* (rgrP/100000.)**0.2854
                rgrQv               = np.squeeze(ncfile.variables["QVAPOR"][:,:Levels,:,:])
                rgrQc               = np.squeeze(ncfile.variables["QCLOUD"][:,:Levels,:,:])
                rgrQr               = np.squeeze(ncfile.variables["QRAIN"][:,:Levels,:,:])
                ncfile.close()

                if sGrid == '12km':
                    if rgrDXnative[dx] != 12000:
                        # bring data to 12 km grid
                        iRatio=12000/rgrDXnative[dx]

                        Pcoarse=np.zeros((rgrP.shape[0],int(rgrP.shape[1]/iRatio),int(rgrP.shape[2]/iRatio))); Pcoarse[:]=np.nan
                        Tcoarse=np.copy(Pcoarse)
                        Qvcoarse=np.copy(Pcoarse)
                        Qccoarse=np.copy(Pcoarse)
                        Qrcoarse=np.copy(Pcoarse)
                        for la in range(Pcoarse.shape[1]):
                            for lo in range(Pcoarse.shape[2]):
                                Pcoarse[:,la,lo]=np.mean(rgrP[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                                Tcoarse[:,la,lo]=np.mean(rgrT[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                                Qvcoarse[:,la,lo]=np.mean(rgrQv[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                                Qccoarse[:,la,lo]=np.mean(rgrQc[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                                Qrcoarse[:,la,lo]=np.mean(rgrQr[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                        rgrP=Pcoarse
                        rgrT=Tcoarse
                        rgrQv=Qvcoarse
                        rgrQc=Qccoarse
                        rgrQr=Qrcoarse
                # Calculate boyancy
                ThetaK=Theta(rgrT,rgrP)
                ThetaP=ThetaK*(1+0.608*rgrQv-rgrQc-rgrQr)
                AvArea=int(100000/rgrDX[dx])
                ThetaP_mean=scipy.ndimage.uniform_filter(ThetaP[:,:,:],[0,AvArea,AvArea])
                Boyancy=(9.81*(ThetaP-ThetaP_mean))/ThetaP_mean

                # Calculate cold pool intensity
                NAN=np.copy(Boyancy)
                for lev in range(Boyancy.shape[0]):
                    NAN[lev,:,:]=(Boyancy[lev,:,:] > -0.005)
                    NAN[lev,:,:][NAN[lev,:,:] == 1]=np.nan
                    if lev > 1:
                        NAN[lev,:,:]=NAN[lev,:,:]+NAN[lev-1,:,:]

                Boyancy_Low=np.copy(Boyancy)
                Boyancy_Low[np.isnan(NAN)]=np.nan
                Boyancy_Sim[tt,:,:,:]=Boyancy_Low
                Boyancy_Integral=np.nansum(Boyancy_Low, axis=0)*250.
                CP_Intensity[tt,:,:]=np.sqrt(-2*Boyancy_Integral)
                
            

            #=============================================================
            #  GET THE COLD POOL ELEMENENTS
            rgrCPPorpAll=np.zeros((len(rgsMetricCPall), CP_Intensity.shape[0] )); rgrCPPorpAll[:]=np.nan

            rgiBoy=np.array((Boyancy_Sim < -0.005)).astype('float')
            # statistics on all cold pools
            rgrCPPorpAll[rgsMetricCPall.index('Volume'),:]=np.nansum(rgiBoy,axis=(1,2,3))*(rgrDX[dx]/1000.)**2*0.25
            rgiBoy_NAN=np.copy(rgiBoy); rgiBoy_NAN[rgiBoy == 0]=np.nan
            DEBTH=np.nansum(rgiBoy_NAN,axis=(1)); DEBTH[rgiBoy[:,0,:,:] == 0]=np.nan
            rgrCPPorpAll[rgsMetricCPall.index('MeanDepth'),:]=np.nanmean(DEBTH*0.25,axis=(1,2))
            rgrCPPorpAll[rgsMetricCPall.index('Area'),:]=np.sum((rgiBoy[:,0,:,:] == 1), axis=(1,2))*(rgrDX[dx]/1000.)**2
            rgrCPPorpAll[rgsMetricCPall.index('P95Depth'),:]=np.array([np.mean(np.sort(np.nansum(rgiBoy_NAN[tt,:,:],axis=0).flatten())[-int((Lat.shape[0]**2)*0.05):]) for tt in range(rgiBoy_NAN.shape[0])])*0.25
            IntensityPercentiles=np.nanpercentile(CP_Intensity, (90,95,99,99.9, 99.99, 100), axis=(1,2))

            # calculate objects based on the most intense coldpools
            CPint_smooth=np.copy(CP_Intensity); CPint_smooth[np.isnan(CPint_smooth)]=0
            CPint_smooth=scipy.ndimage.uniform_filter(CPint_smooth,[iSmooth,iSmooth,iSmooth])
            CPint_TH=(CPint_smooth > 10)
            rgiObjectsUD, nr_objectsUD = ndimage.label(CPint_TH,structure=rgiObj_Struct)
            Object_index=np.array(range(nr_objectsUD-1))+1
            ObjectSize=np.array([np.sum(rgiObjectsUD == ii+1) for ii in range(nr_objectsUD-1)])*rgrDX[dx]**2
            Object_index=Object_index[ObjectSize >= rgiMinVol]

            IntenseCPs={}
            if len(Object_index) > 0:
                for ob in range(len(Object_index)):
                    grObject={}
                    rgrObAct=np.copy(CP_Intensity)
                    rgrObAct[rgiObjectsUD != Object_index[ob]]=0

                    # Does the object hit the boundary?
                    rgiObjActSel=np.array(rgiObjectsUD == Object_index[ob]).astype('float')
                    rgiBoundary=(np.sum(rgiObjActSel[:,0,:], axis=1)+np.sum(rgiObjActSel[:,-1,:], axis=1)+np.sum(rgiObjActSel[:,:,0], axis=1)+np.sum(rgiObjActSel[:,:,-1], axis=1) != 0)                    
                    rgrObAct[rgiBoundary,:,:]=np.nan
                    rgiObjActSel[rgiBoundary,:,:]=np.nan
                    rgrMassCent=np.array([scipy.ndimage.measurements.center_of_mass(rgrObAct[tt,:,:]) for tt in range(rgiObjectsUD.shape[0])])
                    rgrObjSpeed=np.array([((rgrMassCent[tt,0]-rgrMassCent[tt+1,0])**2 + (rgrMassCent[tt,1]-rgrMassCent[tt+1,1])**2)**0.5 for tt in range(rgiObjectsUD.shape[0]-1)])*12.*(rgrDX[dx]/1000.)

                    rgrCP_Vol=(np.array([np.nansum(rgrObAct[tt,:,:]) for tt in range(rgiObjectsUD.shape[0])])/(12.*60.*5.))*rgrDX[dx]**2
                    rgrCP_Max=np.array([np.max(rgrObAct[tt,:,:]) for tt in range(rgiObjectsUD.shape[0])])
                    for tt in range(rgiObjectsUD.shape[0]):
                        if np.sum(rgiObjectsUD[tt,:,:] == (ob+1)) >0:
                            CP_perc=np.percentile(rgrObAct[tt,:,:][rgiObjectsUD[tt,:,:] == (ob+1)], range(101))
                            CP_mean=np.mean(rgrObAct[tt,:,:][rgiObjectsUD[tt,:,:] == (ob+1)])
                        else:
                            CP_mean=np.nan
                            CP_perc=np.array([np.nan]*101)
                        if tt == 0:
                            rgrCP_Percentiles=CP_perc[None,:]
                            rgrCP_Mean=CP_mean
                        else:
                            rgrCP_Percentiles=np.append(rgrCP_Percentiles,CP_perc[None,:], axis=0)
                            rgrCP_Mean=np.append(rgrCP_Mean,CP_mean)

                    rgrSize=np.array([np.sum(rgiObjActSel[tt,:,:] == (ob+1)) for tt in range(rgiObjectsUD.shape[0])])*(rgrDX[dx]/1000.)**2
                    rgrSize[(rgrSize == 0)]=np.nan

                    grObject['rgrObjSpeed']=rgrObjSpeed
                    grObject['rgrCP_Vol']=rgrCP_Vol
                    grObject['rgrCP_Max']=rgrCP_Max
                    grObject['rgrCP_Percentiles']=rgrCP_Percentiles
                    grObject['Percentiles']=np.array(range(101))
                    grObject['rgrCP_Mean']=rgrCP_Mean

                    IntenseCPs[str(ob)]=grObject
                    
            # plt.contourf(rgiObjectsUD[50,:,:], levels=np.linspace(-1,7,100), cmap='coolwarm'); plt.show()

            DATA['rgrCPPorpAll']=rgrCPPorpAll
            DATA['rgsMetricCPall']=rgsMetricCPall
            DATA['IntensityPercentilesAllCPs']=IntensityPercentiles
            DATA['Iperc']=np.array([90,95,99,99.9, 99.99, 100])
            DATA['IntenseCPs']=IntenseCPs

            dbfile = open(fname, 'ab') 
            pickle.dump(DATA, dbfile)                      
            dbfile.close()


            # plt.contourf(CP_Intensity[45,:,:], levels=np.linspace(0,30,20), extend='both', cmap='coolwarm'); plt.show()
