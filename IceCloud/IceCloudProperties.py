#!/usr/bin/env python
''' IceCloudProperties.py


   Here we read in idealized MCS simulation data and calculate ice cloud 
   properties.

   These include size, spread rate, average and minimum temperature of top

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
sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/IceClouds/'+sGrid+'/'
iSIM=>>SIM<<-1
qTH=0.1 # ice cloud threshold in g/kg

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
rgsMetricCPall=['Horizontal_Extend','Spread_Rate','Average_CTT', 'Max_CTT']

iSkipHours=0  # hours to skip at the beginning of the simulation

for si in [iSIM]: #  #range(len(rgsSimulations))[6:]:
    print 'Start with '+rgsSimulations[si]
    CTHeightAll=np.zeros((85,2, len(rgrGridSpacing))); CTHeightAll[:]=np.nan
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m.pkl'
        print fname
        DATA={}
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

            height_v = np.linspace(0,24000-250,96); height_v=height_v/1000.
            height_z = np.linspace(125,24000,95); height_z=height_z/1000.

            rgiLevSel=((height_v > 0) & (height_v < 16))
            rgiLevSelZ=((height_z >= 0) & (height_z <= 16))

            ncfile = Dataset(rgsFiles[0])
            Lat=np.squeeze(ncfile.variables["XLAT"])
            ncfile.close()
            if sGrid == 'native':
                print '    Native grid evaluation is not implemented yet!'
                stop()
                Boyancy_Sim=np.zeros((len(rgsFiles),Levels,Lat.shape[0],Lat.shape[1]))  
                CP_Intensity=np.zeros((len(rgsFiles),Lat.shape[0],Lat.shape[1]))
            else:
                Temp=np.zeros((len(rgsFiles),2)); Temp[:]=np.nan
                CTHeight=np.zeros((len(rgsFiles),2)); CTHeight[:]=np.nan
                CTBoundary=np.zeros((len(rgsFiles)))
                IceCloudHE=np.zeros((len(rgsFiles))); IceCloudHE[:]=np.nan
            for tt in range(len(rgsFiles)):
                print '        read file: '+rgsFiles[tt]
                ncfile = Dataset(rgsFiles[tt])
                rgrP                = np.squeeze(ncfile.variables["PB"][:,:,:,:])+np.squeeze(ncfile.variables["P"][:,:,:,:])
                rgrT                = (np.squeeze(ncfile.variables["T"][:,:,:,:])+300)* (rgrP/100000.)**0.2854
                rgrSNOW             = np.squeeze(ncfile.variables["QSNOW"][:,:,:,:])
                rgrICE              = np.squeeze(ncfile.variables["QICE"][:,:,:,:])
                ncfile.close()
                rgrFrozen=(rgrSNOW+rgrICE)*1000.

                if sGrid == '12km':
                    if rgrDXnative[dx] != 12000:
                        # bring data to 12 km grid
                        iRatio=12000/rgrDXnative[dx]

                        Tcoarse=np.zeros((rgrT.shape[0],int(rgrT.shape[1]/iRatio),int(rgrT.shape[2]/iRatio))); Tcoarse[:]=np.nan
                        FrozenCoarse=np.copy(Tcoarse)
                        for la in range(Tcoarse.shape[1]):
                            for lo in range(Tcoarse.shape[2]):
                                Tcoarse[:,la,lo]=np.mean(rgrT[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                                FrozenCoarse[:,la,lo]=np.mean(rgrFrozen[:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(1,2))
                    else:
                        Tcoarse=rgrT
                        FrozenCoarse=rgrFrozen

                    # get ice cloud horrizontal extend
                    iCloud=np.array((FrozenCoarse >= 0.1))
                    HE_Cloud=np.sum(iCloud, axis=0)
                    HE_Cloud[HE_Cloud > 0]=1
                    # # does the ice cloud touch the boundary?
                    if np.sum(HE_Cloud[:,0])+np.sum(HE_Cloud[:,-1])+np.sum(HE_Cloud[0,:])+np.sum(HE_Cloud[-1,:]) > 0:
                        CTBoundary[tt]=1
                    # get cloud top heigth
                    try:
                        T_for_CTH=np.copy(Tcoarse)
                        T_for_CTH[FrozenCoarse < 0.01]=99999.
                        iCloudTop=np.nanargmin(T_for_CTH, axis=0)
                        CTH=iCloudTop*250.+125.
                        CTH[CTH <= 250]=np.nan
                    except:
                        continue
                    # get cloud top temperature
                    Tcoarse[FrozenCoarse < 0.01]=np.nan
                    TcoarseCTT=np.nanmin(Tcoarse, axis=0)
                    
                    # store data in common matrix
                    Temp[tt,0]=np.nanmean(TcoarseCTT)
                    Temp[tt,1]=np.nanmin(TcoarseCTT)
                    CTHeight[tt,0]=np.nanmean(CTH)
                    CTHeight[tt,1]=np.nanmax(CTH)
                    IceCloudHE[tt]=np.sum(HE_Cloud)*rgrDX[dx]**2

            DATA['TS']=Temp
            DATA['CTH']=CTHeight
            DATA['Boundary']=CTBoundary
            DATA['IceCloud_HE']=IceCloudHE
            DATA['time']=np.linspace(0,84*5,85)
            DATA['TempStats']=['Mean Cloud Top Temperature','Minimum Cloud Top Temperature']

            dbfile = open(fname, 'ab') 
            pickle.dump(DATA, dbfile)                      
            dbfile.close()

            CTHeightAll[:,:,dx]=CTHeight

