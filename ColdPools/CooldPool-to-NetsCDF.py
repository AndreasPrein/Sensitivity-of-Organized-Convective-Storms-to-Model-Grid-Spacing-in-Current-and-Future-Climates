#!/usr/bin/env python
''' CooldPool-to-NetsCDF.py


   This program reads in an example MCS output, calculates the
   Cold pool and writes the result back to the original netcdf.

   Unlimately, this information is used in a Vapor visulaization.
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
DataFile='/glade/campaign/mmm/c3we/Idealized_MCSs/VAPOR-Visualizations/03_2011-07-16_CTRL_Midwest_-Loc2_MCS_Storm-Nr_JJA-8-TH5/12000/wrfout_d01_0001-01-01_04:30:00'

rgrGridSpacing=['250']
rgrDXnative=250


iSkipHours=0  # hours to skip at the beginning of the simulation
rgiMinVol= 36000*36000*4  # minimum volume of cold pool size [36 km x 36 km * 20 min]
Smooth=12000 # smoothing filter lenth in m
DraftMinSpeed=3
Levels=16 # number of model levels that should be read in

rgiObj_Struct=np.zeros((3,3,3)); rgiObj_Struct[:,:,:]=1


iSmooth=int(Smooth/rgrDXnative)
i1km=int(1000/rgrDXnative)
if i1km ==0:
    i1km=1

print '    dx = '+rgrGridSpacing[0]

#=============================================================

# get the size of the domain
ncfile = Dataset(DataFile)
Lat=np.squeeze(ncfile.variables["XLAT"])
ncfile.close()

height_v = np.linspace(0,24000-250,96); height_v=height_v/1000.
height_z = np.linspace(125,24000,95); height_z=height_z/1000.

rgiLevSel=((height_v > 0) & (height_v < 16))
rgiLevSelZ=((height_z >= 0) & (height_z <= 16))

ncfile = Dataset(DataFile)
Lat=np.squeeze(ncfile.variables["XLAT"])
ncfile.close()
Boyancy_Sim=np.zeros((len(DataFile),Levels,Lat.shape[0],Lat.shape[1]))  
CP_Intensity=np.zeros((len(DataFile),Lat.shape[0],Lat.shape[1]))

print '        read file: '+DataFile
ncfile = Dataset(DataFile)

rgrP                = np.squeeze(ncfile.variables["PB"][:,:Levels,:,:])+np.squeeze(ncfile.variables["P"][:,:Levels,:,:])
rgrT                = (np.squeeze(ncfile.variables["T"][:,:Levels,:,:])+300)* (rgrP/100000.)**0.2854
rgrQv               = np.squeeze(ncfile.variables["QVAPOR"][:,:Levels,:,:])
rgrQc               = np.squeeze(ncfile.variables["QCLOUD"][:,:Levels,:,:])
rgrQr               = np.squeeze(ncfile.variables["QRAIN"][:,:Levels,:,:])
# rgrQs               = np.squeeze(ncfile.variables["QSNOW"][:,:Levels,:,:])
ncfile.close()

# Qcloud=rgrQc+rgrQr+rgrQs

# Calculate boyancy
ThetaK=Theta(rgrT,rgrP)
ThetaP=ThetaK*(1+0.608*rgrQv-rgrQc-rgrQr)
AvArea=int(100000/rgrDXnative)
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
Boyancy_Low[np.isnan(NAN)]=0

Coldpools=np.zeros((len(height_z), Boyancy_Low.shape[1],Boyancy_Low.shape[2])); Coldpools[:]=0
Coldpools[:Boyancy_Low.shape[0],:,:]=Boyancy_Low; Coldpools[np.isnan(Coldpools)]=0

Boyancy_Integral=np.nansum(Boyancy_Low, axis=0)*250.
CP_Intensity=np.sqrt(-2*Boyancy_Integral)


# Write data to file
file=Dataset(DataFile,'r+')
CP=file.createVariable('CooldPool','float32',('Time', 'bottom_top', 'south_north', 'west_east'))
CP[:]=Coldpools[None,:]
CPint=file.createVariable('CP_Intensity','float32',('Time', 'south_north', 'west_east'))
CPint[:]=CP_Intensity[None,:]
file.close()

stop()
