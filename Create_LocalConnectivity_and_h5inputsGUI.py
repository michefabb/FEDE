#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:11:40 2024

@author: michelangelo
"""


import argparse
parser = argparse.ArgumentParser(description="Creates Local Connectivity files for TVB and h5 files for input to TVB GUI")
parser.add_argument('path_to_input_data', type=str, help='path to input data')
parser.add_argument('save_path_LocConn', type=str, help='path to save Local Connectivity files')
parser.add_argument('save_path_h5GUI', type=str, help='path to save h5 files for GUI')
parser.add_argument('subject_name', type=str, help='subject name')
parser.add_argument('sigma', type=float, help='sigma of local connectivity')
parser.add_argument('amp', type=float, help='amp of local connectivity')
parser.add_argument('midpoint', type=float, help='midpoint of local connectivity')
parser.add_argument('offset', type=float, help='offset of local connectivity')
args = parser.parse_args()

path_to_input_data                  = args.path_to_input_data
save_path_LocConn                   = args.save_path_LocConn
save_path_h5GUI                     = args.save_path_h5GUI
subject_name                        = args.subject_name
sigma                               = args.sigma
amp                                 = args.amp
midpoint                            = args.midpoint
offset                              = args.offset



#%pylab nbagg
import matplotlib
import numpy as np
import numpy
from tvb.simulator.lab import *
import multiprocessing as mp 
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.datatypes.sensors import SensorsEEG
from tvb.datatypes import connectivity
import json
########################################################################################################## 
# path_to_input_data = "/home/bbpnrsoa/TVB_output/"
# save_path='/home/bbpnrsoa/Results/'
# path_to_input_data = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/output_tvb_converter/TVB_output/"
# save_path_LocConn="/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/output_tvb_converter/TVB_output/"
# save_path_h5GUI="/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/h5_inputs_for_GUI/"
# subject_name="sub-01"
# sigma=0.5
# amp=1.0
# midpoint=0.0
# offset=0.0
########################################################################################################## 
# for Local Connectivity Matrix calculation the below values do not change the results and are not saved, but are needed to run the .configure parts; same for the h5 files
scale_weights=13366.0
local_coupling_strength = 0.1
white_matter_coupling_a=0.1
white_matter_speed=10.0




from tvb.basic.neotraits.api import List

white_matter = connectivity.Connectivity.from_file(path_to_input_data+'sub-01_Connectome.zip')
white_matter.speed = numpy.loadtxt(path_to_input_data+'sub-01_conduction_velocities.txt', dtype=float) #this uses the map of conduction velocity
white_matter.speed[white_matter.speed==0]=0.0000000001 # for elements with speed =0 replaces with 10^-10 to avoid possible issues
nr_streamlines = numpy.array([71631000]) # this is the nr of streamlines provided as input during Tractography
white_matter.weights = white_matter.weights * scale_weights/nr_streamlines
white_matter.configure()

white_matter_coupling = coupling.SigmoidalJansenRit(cmin=numpy.array([0.0]), cmax=numpy.array([0.005]), midpoint=numpy.array([6.0]), r=numpy.array([0.56]), a=numpy.array([white_matter_coupling_a], dtype=float))
white_matter_coupling.configure()

rm_f_name = (path_to_input_data+'sub-01_region_mapping.txt')
rm = RegionMapping.from_file(rm_f_name)
rm.connectivity=white_matter
sensorsEEG = SensorsEEG.from_file(path_to_input_data+'sub-01_EEG_Locations.txt')
sensorsEEG.configure()
prEEG = ProjectionSurfaceEEG.from_file(path_to_input_data+'sub-01_EEGProjection.mat', matlab_data_name="ProjectionMatrix")
prEEG.sensors=sensorsEEG
fsamp = 1e3/256.0       # sampling frequency as 1000ms / sampling_frequency_in_Hz
##########################################################################################################  
from tvb.datatypes.cortex import Cortex
from tvb.datatypes import surfaces, local_connectivity, equations
from scipy import io as sio

ctx_surface_name=(path_to_input_data+'sub-01_Cortex.zip')

ctx_surface = surfaces.CorticalSurface.from_file(ctx_surface_name)
ctx_surface.zero_based_triangles=True
ctx_surface.configure()
prEEG.sources=ctx_surface
prEEG.configure()
rm.surface=ctx_surface
rm.configure()

loc_conn = local_connectivity.LocalConnectivity(surface=ctx_surface,
                                                matrix=None, equation=equations.Gaussian(), cutoff=40.0,
                                               ) # set matrix = None if you wish to calculate a new local connectivity using the equation.parameters below (uncomment them and also the lines ctx.local_connectivity.matrix = None and ctx.compute_local_connectivity()), and also the lines between "ctx.local_connectivity.matrix = (ctx.local_connectivity.matrix+ctx.local_connectivity.matrix.T)/2" and "ctx.local_connectivity.configure()", instead of loading a local connectivity matlab file...remember to use also the final part of the script to save the matrix and the metadata

loc_conn.equation.parameters['sigma'] = sigma
loc_conn.equation.parameters['amp'] = amp
loc_conn.equation.parameters['midpoint'] = midpoint
loc_conn.equation.parameters['offset'] = offset
loc_conn.configure()

ctx = Cortex.from_file(source_file=ctx_surface_name, region_mapping_file=rm_f_name, local_connectivity_file=None) # if you wish to load an already existing local connectivity file, change from None to the file path
ctx.surface.configure()
ctx.region_mapping_data.connectivity = white_matter
ctx.region_mapping_data.connectivity.configure()
ctx.region_mapping_data.configure()
ctx.local_connectivity = loc_conn
ctx.coupling_strength = numpy.array([local_coupling_strength], dtype=float)
ctx.local_connectivity.matrix = None # needed? just to be sure it is recalculated
ctx.compute_local_connectivity() # needed? just to be sure it is recalculated
ctx.local_connectivity.matrix = (ctx.local_connectivity.matrix+ctx.local_connectivity.matrix.T)/2 # to make symmetric the distance i to j and j to i, see https://groups.google.com/g/tvb-users/c/E6VA7hCSFmc/m/jS-ggjyUAAAJ
# below to set to 0 the local connectivity elements of dummy regions...loc con matrix is a csc sparse matrix of scipy.sparse.csc_matrix, need to use its operations
from scipy.sparse import csc_matrix, eye
remove=eye(ctx.vertices.shape[0],dtype=float).tolil() # this creates an identity sparse matrix
for i in range(len(rm.array_data)-1):
    if rm.array_data[i]==white_matter.weights.shape[0]-1 or rm.array_data[i]==white_matter.weights.shape[0]-2:
        remove[i,i]=0.0 # this creates a identity sparse matrix with 0 on the diagonal at the index of vertices that in region map correspond to dummy regions
ctx.local_connectivity.matrix=remove.dot(ctx.local_connectivity.matrix.dot(remove)).tocsc() # this Imodified*(A*Imodified) matrix operation set to 0 the rows and columns of matrix A at the indices of vertices in dummy regions
local_connectivity_matrix_name=('local_connectivity_'+str(loc_conn.equation)[0:8:1]+'_'+str(loc_conn.cutoff)+'_'+str(loc_conn.equation.parameters['sigma'])+'_'+str(loc_conn.equation.parameters['amp'])+'_'+str(loc_conn.equation.parameters['midpoint'])+'_'+str(loc_conn.equation.parameters['offset'])+'_'+'matrix.mat')
local_connectivity_metadata_name=('local_connectivity_'+str(loc_conn.equation)[0:8:1]+'_'+str(loc_conn.cutoff)+'_'+str(loc_conn.equation.parameters['sigma'])+'_'+str(loc_conn.equation.parameters['amp'])+'_'+str(loc_conn.equation.parameters['midpoint'])+'_'+str(loc_conn.equation.parameters['offset'])+'_'+'metadata.json')
ctx.local_connectivity.configure()


ctx.configure()
##########################################################################################################  


mons = (
    monitors.EEG(sensors=sensorsEEG, projection=prEEG, region_mapping=rm, period=fsamp, reference=None, variables_of_interest=numpy.array([0])),
)
mons[0].configure()
# if reference=None it is the idealized reference-free EEG; if reference="average" it is used the average reference; if reference= name of EEG sensor, it is used that sensor as reference
# variables_of_interest= if not specified, it uses the variables_of_interest used in the model; 
##########################################################################################################
##########################################################################################################
# SAVE FILES
##########################################################################################################
from scipy import io as sio

if not os.path.exists(save_path_h5GUI):
        os.makedirs(save_path_h5GUI)


##########################################################################################################
# SAVE LOCAL CONNECTIVITY
##########################################################################################################
sio.savemat(save_path_LocConn+'/'+'local_connectivity_'+str(loc_conn.equation)[0:8:1]+'_'+str(loc_conn.cutoff)+'_'+str(loc_conn.equation.parameters['sigma'])+'_'+str(loc_conn.equation.parameters['amp'])+'_'+str(loc_conn.equation.parameters['midpoint'])+'_'+str(loc_conn.equation.parameters['offset'])+'_'+'matrix.mat', {'local_connectivity_matrix': ctx.local_connectivity.matrix})

local_connectivity_json = {}
local_connectivity_json["description"] = "Metadata of local connectivity"
local_connectivity_json["surface"] = ctx_surface_name
local_connectivity_json["equation"] = str(loc_conn.equation)[0:8:1]
local_connectivity_json["cutoff"] = loc_conn.cutoff
local_connectivity_json["sigma"] = loc_conn.equation.parameters['sigma']
local_connectivity_json["amp"] = loc_conn.equation.parameters['amp']
local_connectivity_json["midpoint"] = loc_conn.equation.parameters['midpoint']
local_connectivity_json["offset"] = loc_conn.equation.parameters['offset']

import json
with open(save_path_LocConn+'/'+'local_connectivity_'+str(loc_conn.equation)[0:8:1]+'_'+str(loc_conn.cutoff)+'_'+str(loc_conn.equation.parameters['sigma'])+'_'+str(loc_conn.equation.parameters['amp'])+'_'+str(loc_conn.equation.parameters['midpoint'])+'_'+str(loc_conn.equation.parameters['offset'])+'_'+'metadata.json', 'w') as outfile_loccon_json:
    json.dump(local_connectivity_json, outfile_loccon_json)
print("Local Connectivity files created!")


# ##########################################################################################################
# SAVE TO h5 FOR GUI IMPORT
# ##########################################################################################################
from tvb.core.neocom import h5
from tvb.basic.profile import TvbProfile


TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

import shutil
subfolder_h5=save_path_h5GUI+'/h5_local_conn_sigma_'+str(loc_conn.equation.parameters['sigma'])
os.makedirs(subfolder_h5)

h5.store_complete_to_dir(sensorsEEG, subfolder_h5)
h5.store_complete_to_dir(white_matter, subfolder_h5)
h5.store_complete_to_dir(loc_conn, subfolder_h5)
h5.store_complete_to_dir(ctx_surface, subfolder_h5)
h5.store_complete_to_dir(prEEG, subfolder_h5)
h5.store_complete_to_dir(rm, subfolder_h5)

skinair_surface_name=(path_to_input_data+'sub-01_outer_skin.zip')
skinair_surface = surfaces.SkinAir.from_file(skinair_surface_name)
skinair_surface.zero_based_triangles=True
skinair_surface.configure()
h5.store_complete_to_dir(skinair_surface, subfolder_h5)

# zip files
shutil.make_archive(subfolder_h5, 'zip', subfolder_h5)
print("h5 files created!")
shutil.rmtree(subfolder_h5)


