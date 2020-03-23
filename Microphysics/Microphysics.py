#!/usr/bin/env python
''' Microphysics.py


   Here we read in idealized MCS simulation data and calculate microphysic properties

   These include rain, cloud liquit, gaupel, snow, and ice mass mixing ratio

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
sDataOut='/glade/scratch/prein/Papers/Idealized_MCSs/data/Mikrophysics/'+sGrid+'/'
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

Variables=["QSNOW","QICE","QGRAUP","QRAIN","QCLOUD"] #,"QNRAIN","QNICE"]
rgiDomSize=[51,51,155,311,623,1247,2495]

iSkipHours=0  # hours to skip at the beginning of the simulation

for si in [iSIM]: #  #range(len(rgsSimulations))[6:]:
    print 'Start with '+rgsSimulations[si]
    for dx in range(len(rgrGridSpacing)):
        fname=sDataOut+rgsSimulations[si]+'_'+rgrGridSpacing[dx]+'m_MP.pkl'
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
                MR_mean=np.zeros((len(rgsFiles),5,len(height_z))); MR_mean[:]=np.nan
                # MassPerDrop=np.zeros((len(rgsFiles),2,len(height_z))); MassPerDrop[:]=np.nan
            for tt in range(len(rgsFiles)):
                print '        read file: '+rgsFiles[tt]
                ncfile = Dataset(rgsFiles[tt])
                DATA_all=np.zeros((len(Variables),len(height_z),rgiDomSize[dx],rgiDomSize[dx])); DATA_all[:]=np.nan
                for va in range(len(Variables)):
                    DATA_all[va,:,:,:]=np.squeeze(ncfile.variables[Variables[va]][:,:,:,:])
                ncfile.close()

                if sGrid == '12km':
                    if rgrDXnative[dx] != 12000:
                        # bring data to 12 km grid
                        iRatio=12000/rgrDXnative[dx]
                        DATA_coarse=np.zeros((len(Variables),len(height_z),int(rgiDomSize[0]),int(rgiDomSize[0]))); DATA_coarse[:]=np.nan
                        for la in range(rgiDomSize[0]):
                            for lo in range(rgiDomSize[0]):
                                DATA_coarse[:,:,la,lo]=np.mean(DATA_all[:,:,la*iRatio:la*iRatio+iRatio,lo*iRatio:lo*iRatio+iRatio], axis=(2,3))
                    else:
                        DATA_coarse=DATA_all
                    Cloud_MR=np.sum(DATA_coarse[(Variables.index("QSNOW"),Variables.index("QICE"),Variables.index("QGRAUP"), Variables.index("QCLOUD"), Variables.index("QRAIN")),:], axis=0)*1000.
                    Cloud_Extend=np.copy(Cloud_MR)
                    Cloud_Extend[:]=0; Cloud_Extend[(Cloud_MR > qTH)]=1
                    DATA_coarse[:,(Cloud_MR < qTH)]=np.nan
                    DATA_coarse[:,(np.sum(Cloud_Extend, axis=(1,2)) < 50),:,:]=np.nan
                    # # does the cloud touch the boundary?
                    # if np.sum(Cloud_Extend[:,:,0])+np.sum(Cloud_Extend[:,:,-1])+np.sum(Cloud_Extend[:,0,:])+np.sum(Cloud_Extend[:,-1,:]) > 100:
                    #     continue
                    # store data in common matrix
                    MR_mean[tt,:,:]=np.nanmean(DATA_coarse[(Variables.index("QSNOW"),Variables.index("QICE"),Variables.index("QGRAUP"), Variables.index("QCLOUD"), Variables.index("QRAIN")),:], axis=(2,3))
                    # MassPerDrop[tt,:,:]=MR_mean[tt,(Variables.index("QRAIN"),Variables.index("QICE")),:]/np.nanmean(DATA_coarse[(Variables.index("QNRAIN"),Variables.index("QNICE")),:], axis=(2,3))


            # plt.contourf(MR_mean[:,2,:].transpose()); plt.show()

            DATA['MR']=MR_mean
            DATA['Variables']=Variables
            DATA['time']=np.linspace(0,84*5,85)
            DATA['height_z']=height_z

            dbfile = open(fname, 'ab') 
            pickle.dump(DATA, dbfile)                      
            dbfile.close()

