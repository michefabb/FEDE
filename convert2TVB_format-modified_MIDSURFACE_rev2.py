#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:56:40 2023

@author: michelangelo
"""
# rev1:
# - added conduction velocity map to TVB inputs files
# rev2:
# - modified in order to use Brainstorm or SimNIBS instead of MNE for computation of Leadfield Matrix



import argparse
parser = argparse.ArgumentParser(description="Convert data of MRtrix3_connectome and fmriprep pipelines to TVB format.")
parser.add_argument('recon_all_name', type=str, help='Name of subject in recon_all_dir')
parser.add_argument('recon_all_dir', type=str, help='Path to the recon-all results')
parser.add_argument('mrtrix_output_dir', type=str, help='Path to results of MRtrix3_connectome pipeline')
parser.add_argument('participant_label', type=str, help='Participant label as in BIDS dataset, i.e. sub-<participant_label>')
parser.add_argument('parcellation', type=str, help='Parcellation used in MRtrix3_connectome pipeline')
parser.add_argument('parc_image', type=str, help='Path to the parcellated image used for SC etxraction')
parser.add_argument('tvb_output', type=str, help='Path to results of TVBconverter pipeline')
parser.add_argument('tvb_workdir', type=str, help='Path to temp dir of TVBconverter pipeline')
parser.add_argument('n_cpus', type=int, help='How many cpus to run in parallel')
parser.add_argument('weights_path', type=str, help='Path to the SC weights file')
parser.add_argument('tracts_path', type=str, help='Path to the SC tract length file')
parser.add_argument('velocities_path', type=str, help='Path to the Conduction Velocity file')
parser.add_argument('input_dir', type=str, help='Path to the BIDS input dir')
parser.add_argument('leadfield_path', type=str, help='Path to the Leadfield calculation dir')
# fMRI not available, hence below is commented...if available uncomment
#parser.add_argument('fMRI_ROI_ts', type=str, help='Path to the fMRI ROI level time series')
#parser.add_argument('task_name', type=str, help='Name of the fMRI e.g. "rest" or some special task')
args = parser.parse_args()

#%%
import mne
import numpy as np
from mne.surface import _project_onto_surface
from mne.io.constants import FIFF
from scipy import spatial
import scipy.io as sio
import os
import shutil
import nibabel as nib
import sys
import glob
import json
import pandas as pd
from subprocess import Popen, PIPE
import nibabel.gifti as nbg

#%%

recon_all_name     = args.recon_all_name
recon_all_dir      = args.recon_all_dir
parcellation       = args.parcellation
mrtrix_output_dir  = args.mrtrix_output_dir
participant_label  = args.participant_label
parc_image         = args.parc_image
tvb_output         = args.tvb_output
tvb_workdir        = args.tvb_workdir
n_cpus             = args.n_cpus
weights_path       = args.weights_path
tracts_path        = args.tracts_path
velocities_path    = args.velocities_path
input_dir          = args.input_dir
leadfield_path     = args.leadfield_path
pipeline_name      = "TVB"
# fMRI not available, hence below is commented...if available uncomment
# fMRI_ROI_ts        = args.fMRI_ROI_ts
# task_name          = args.task_name


# recon_all_name     = "sub-01_recon"
# recon_all_dir      = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/"
# parcellation       = "hcpmmp1"
# mrtrix_output_dir  = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/output_tvb_converter/mrtrix_output"
# participant_label  = "01"
# parc_image         = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/sub-01_parc-hcpmmp1_indices.nii.gz"
# tvb_output         = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/output_tvb_converter/TVB_output"
# tvb_workdir        = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/output_tvb_converter/TVB_output/tmp"
# n_cpus             = 11
# weights_path       = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/sub-01_parcels_coreg_connectome.csv"
# tracts_path        = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/sub-01_parcels_coreg_meanlength.csv"
# velocities_path    = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/sub-01_parcels_coreg_meanvelocity.csv"
# input_dir          = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/input_dir"
# leadfield_path     = "/Users/michelangelo/Stella_Maris_Data/Stella_Maris_Patient_SS3T-CSD/SimNIBS/leadfield"
# #fMRI_ROI_ts        = args.fMRI_ROI_ts
# #task_name          = args.task_name
# pipeline_name      = "TVB"

# Set Freesurfer enviornment variable in the MNE config file, which can be found with the MNE command print(mne.get_config_path())
mne.set_config("FREESURFER_HOME", "/Applications/freesurfer/7.3.2", home_dir=None, set_env=True)
mne.set_config("FS_LICENSE", "/Users/michelangelo/license.txt", home_dir=None, set_env=True)
# do not set SUBJECTS_DIR...all the scripts below read it from the variable recon_all_dir
# os.environ['SUBJECTS_DIR']=recon_all_dir
# os.system("/Applications/freesurfer/7.3.2/SetUpFreeSurfer.sh")

#%%
# =============================================================================
# create cortical surface and region mapping 
# =============================================================================

# read surface, and convert units of the vertex positions
# CALCULATE THE VERTEX AT MIDTHICKNESS BETWEEN PIAL AND WHITE SURFACE AS DONE IN SIMNIBS
pial_lp = mne.read_surface(recon_all_dir + "/" + recon_all_name + "/surf/lh.pial", read_metadata=True, return_dict=True, verbose=True)
pial_lw = mne.read_surface(recon_all_dir + "/" + recon_all_name + "/surf/lh.white", read_metadata=True, return_dict=True, verbose=True)
pial_l=()
pial_l=[(np.array(pial_lp[0])+np.array(pial_lw[0]))/2]
pial_l.append(pial_lp[1])
pial_l.append(pial_lp[2])
pial_l.append(pial_lp[3])
pial_l=tuple(pial_l)    
pial_l[3]['rr']  = pial_l[0]/1000

pial_rp = mne.read_surface(recon_all_dir + "/" + recon_all_name + "/surf/rh.pial", read_metadata=True, return_dict=True, verbose=True)
pial_rw = mne.read_surface(recon_all_dir + "/" + recon_all_name + "/surf/rh.white", read_metadata=True, return_dict=True, verbose=True)
pial_r=()
pial_r=[(np.array(pial_rp[0])+np.array(pial_rw[0]))/2]
pial_r.append(pial_rp[1])
pial_r.append(pial_rp[2])
pial_r.append(pial_rp[3])
pial_r=tuple(pial_r)   
pial_r[3]['rr']  = pial_r[0]/1000

# merge surfaces, first left than right
n_vert_l = pial_l[3]['np']
n_vert_r = pial_r[3]['np']
n_vert   = n_vert_l + n_vert_r
pial_vertices  = np.concatenate((pial_l[3]['rr'],   pial_r[3]['rr']))
pial_tris      = np.concatenate((pial_l[3]['tris'], pial_r[3]['tris'] +  n_vert_l))

# for surface parcellations (i.e. desikan, destrieux, hcpmmp1) use the annot files from freesurfer to get the region mapping
# for volumetric parcellations use the label.gii files created with wb_command
if parcellation in ["desikan", "destrieux", "hcpmmp1"]: # i.e. surface parcellations
    
    # mrtrix lut, connectome ordering !!!!! 
    if parcellation == "desikan":
        parc="aparc"
        mrtrix_lut    = np.genfromtxt(mrtrix_output_dir+"/fs_default.txt", skip_header=4, dtype="str")
        region_names  = mrtrix_lut[:,2]
        region_names=np.delete(region_names, 0,0)
        region_names=np.delete(region_names, 36,0)
        region_names=np.delete(region_names, 43,0)
        prefix_len    = 7
        cortical = [ 1 if name[:3]=="ctx" else 0 for name in region_names ]
        hemisphere    = [ 1 if name[4:6] == "rh" or name[0:5] == "Right" else 0 for name in region_names] # 1 = right hemisphere, 0 = left hemisphere
        regexp_append=""
        
    elif parcellation =="destrieux": 
        parc="aparc.a2009s"
        mrtrix_lut    = np.genfromtxt(mrtrix_output_dir+"/fs_a2009s.txt", skip_header=4, dtype="str")
        #region_names  = mrtrix_lut[:,1] # need to jump nr 0 Unknown, 77 Left Thalamus Proper 85 Right Thalamus Proper 
        region_names  = mrtrix_lut[:,1]
        region_names=np.delete(region_names, 0,0)
        region_names=np.delete(region_names, 76,0)
        region_names=np.delete(region_names, 83,0)
        region_names  = [name.replace("&","_and_") for name in region_names] # in annot file name is written with "&" in stead of "_and_"
        prefix_len    = 7
        cortical = [ 1 if name[:3]=="ctx" else 0 for name in region_names ]
        hemisphere   = [ 1 if name[4:6] == "rh" or name[0:5] == "Right" else 0 for name in region_names]
        regexp_append=""
        
    elif parcellation =="hcpmmp1": 
        parc="HCPMMP1"
        mrtrix_lut    = np.genfromtxt(mrtrix_output_dir+"/hcpmmp1_ordered.txt", skip_header=3, dtype="str")
        region_names  = mrtrix_lut[:,1]
        # correct for typo in mrtrix lut, exchange R with L in some names
        #idx_names = np.array([20,21,25,26,39,42,46,48,70,76,91,92,95,124,159,173,174]) -1 + 180
        #correct_names = ["LO1", "LO2", "PSL", "SFL", "5L", "7AL", "7PL", "LIPv", "8BL", "47l","11l", "13l", "LIPd", "PBelt", "LO3", "MBelt", "LBelt"]
        #for i in range(len(idx_names)): region_names[idx_names[i]] = "R_"+correct_names[i]
        prefix_len    = 0
        cortical = np.zeros((len(region_names))) # regions are ordered, first cortical than subcortical ones
        cortical[:360] = 1
        hemisphere = [ 1 if name[0] == "R" else 0 for name in region_names] # this way brainstem is assigned to left hemisphere
        regexp_append="_ROI"
        
    
    # create region map of high_res pial surface
    region_map = np.zeros((n_vert))
    n_regions = mrtrix_lut.shape[0]+2 # MODIFICHE X CORTEX NOHOLES: AGGIUNTO +2
    
    # load labels according to mrtrix lut order !!!
    r = 1
    for i in range(len(region_names)):
        if cortical[i] :  # only cortical regions are represented in the region map 
            if  hemisphere[i] == 1: 
                add  = n_vert_l
                hemi = "rh"
            elif hemisphere[i] == 0: 
                add  = 0
                hemi = 'lh'
                
            label = region_names[i][prefix_len:]+regexp_append
            label = mne.read_labels_from_annot(subject=recon_all_name, subjects_dir=recon_all_dir, hemi=hemi, regexp ="^"+label, surf_name = 'pial', parc=parc) 
            region_map[label[0].vertices + add] = r
        r += 1
    
    # those entries which are still 0 were not labeled
    # they are corresponding to "subcortical" vertices, i.e. on the brain "inside" of the cortical surface
    # delete those from region map and cortical surface vertices and tris
    ind_sub_vert = np.where(region_map==0)[0]
    ## MODIFICHE X CORTEX NOHOLES: HO COMMENTATO LE DUE LINEE 180 E 181 E SCRITTO LE 182-185, POI COMMENTATO LE 189-203
    #pial_vertices = np.delete(pial_vertices, ind_sub_vert, 0)
    #region_map = region_map[region_map!=0] # remove "subcortical" vertices
    region_names=np.append(region_names, ['L_NonCortical_Dummy_Surfaces'])
    cortical=np.append(cortical, [1])
    hemisphere=np.append(hemisphere, [0])
    region_names=np.append(region_names, ['R_NonCortical_Dummy_Surfaces'])
    cortical=np.append(cortical, [1])
    hemisphere=np.append(hemisphere, [1])
    # MODIFICHE X CORTEX NOHOLES: PER DIVIDERE TRA DUMMY REGION LEFT E RIGHT
    for i in range(len(region_map)):
        if region_map[i]==0 :
            if  i <= n_vert_l-1 : # left hemisphere
                region_map[i]=len(region_names)-1
            elif i <= n_vert_l-1+n_vert_r : # right hemisphere
                region_map[i]=len(region_names)
                
    
            
    region_map -= 1 # reduce labels by 1, to start from 0 again
    
    # # get triangles which contains these "subcorical" vertices
    # mask = np.isin(pial_tris, ind_sub_vert)
    # rows, cols = np.nonzero(mask)
    # rows = np.unique(rows)
    
    # # delete tris
    # pial_tris = np.delete(pial_tris, rows,0)
    
    # # update tri indices
    # kk = []
    # for i in range(len(ind_sub_vert)):
    #     ind = ind_sub_vert[i]
    #     if pial_tris[pial_tris > ind ].sum() == 0: kk.append(ind)
    #     pial_tris[pial_tris > ind ] -= 1 
    #     ind_sub_vert -= 1
    
elif parcellation in ["aal", "aal2", "craddock200", "craddock400", "perry512"]: #i.e. volumetric parcellations
    
    # load label data from gifti files
    gii_r = nib.load(tvb_workdir+"/rh."+parcellation+".pial.label.gii")
    gii_l = nib.load(tvb_workdir+"/lh."+parcellation+".pial.label.gii")
    region_map = np.concatenate((gii_l.darrays[0].data, gii_r.darrays[0].data))
    
    # too identify "subcortical" voxel we still load e.g. desikan annot file and use it as a mask
    region_map_mask = np.zeros((n_vert))
    
    # load label for vertices from any freesurfer annot files
    labels = mne.read_labels_from_annot(subject=recon_all_name, subjects_dir=recon_all_dir, parc="aparc", surf_name = 'pial')
    for label in labels:
        if  label.hemi == "rh": 
            add  = n_vert_l
        elif label.hemi == "lh": 
            add  = 0
        region_map_mask[label.vertices + add] = 1
    
    # those entries which are still 0 were not labeled
    # they are corresponding to "subcortical" vertices, i.e. on the brain "inside" of the cortical surface
    # delete those from region map and cortical surface vertices and tris
    ind_sub_vert  = np.where(region_map_mask==0)[0]
    pial_vertices = np.delete(pial_vertices, ind_sub_vert, 0)
    region_map    = region_map[region_map_mask!=0] # remove "subcortical" vertices
    region_map -= 1 # reduce labels by 1, to start from 0 again
    
    # get triangles which contains these "subcorical" vertices
    mask = np.isin(pial_tris, ind_sub_vert)
    rows, cols = np.nonzero(mask)
    rows = np.unique(rows)
    
    # delete tris
    pial_tris = np.delete(pial_tris, rows,0)
    
    # update tri indices above ind
    kk = []
    for i in range(len(ind_sub_vert)):
        ind = ind_sub_vert[i]
        if pial_tris[pial_tris > ind ].sum() == 0: kk.append(ind)
        pial_tris[pial_tris > ind ] -= 1 
        ind_sub_vert -= 1
    
    # mrtrix lut, connectome ordering !!!!! 
    if parcellation in ["aal", "aal2"]:
        mrtrix_lut    = np.genfromtxt(mrtrix_output_dir+"/parc-"+parcellation+"_lookup.txt", skip_header=3, dtype="str")
        region_names  = mrtrix_lut[:,2]
        n_regions     = len(region_names)
        hemisphere    = [ 1 if name[-1] == "R" else 0 for name in region_names] # 1 = right hemisphere, 0 = left hemisphere, assigns the Vermis to Left hemisphere

    elif parcellation in ["craddock200", "craddock400", "perry512"]: 
        # These parcellations have no mrtrix_lut and no region names
        # see Craddock et al. 2012 regions are derived from clustering voxels FC pattern
        # Perry A. 2015 regions are derived via subdivision of AAL atlas
        
        # get number of regions by counting unique values in parc_image
        img = nib.load(parc_image) 
        img_data = img.get_fdata().astype('int')
        n_regions = len(np.unique(img_data)) - 1 # reduce by 1 because of 0 values in image background
        region_names = [str(i) for i in range(n_regions)]

        # all labels appearing in the gifti label files RIGHT are right hemisphere (i.e. 1 in hemisphere array)
        # NOTE: sometimes left and right cortical surfaces may overlap, this will assign some region to both hemispheres
        # can be crucial for surface simulations
        hemisphere = np.zeros((n_regions))
        index_hemisphere = np.unique(gii_r.darrays[0].data)[1:] -1 # skip the 0 for "subcortical" vertices and remove by 1 because region start counting at 0
        hemisphere[index_hemisphere] = 1
        
    # all labels appearing in region_map correspond to cortical regions
    cortical = np.zeros((n_regions))
    cortical[np.unique(region_map)] = 1
        
    
    

#%%
# =============================================================================
# compute source space 
# =============================================================================
# decimate surface
#pial_dec = mne.decimate_surface(pial_vertices, pial_tris, n_triangles=5120, method="sphere") # default was n_triangles=30000 and no method, TVB default cortex has 32760 triangles 
pial_dec=(pial_vertices, pial_tris)

# complete decimated surface (add normals + other parameters)
pial_dict = {'rr':pial_dec[0], 'tris':pial_dec[1]}
pial_complete = mne.surface.complete_surface_info(pial_dict)

# construct source space dictionary by hand
# use all point of the decimated surface as souce space
src =   {'rr':       pial_complete['rr'],
         'tris':     pial_complete['tris'],
         'ntri':     pial_complete['ntri'],
         'use_tris': pial_complete['tris'],
         'np':       pial_complete['np'],
         'nn':       pial_complete['nn'],
         'inuse':    np.ones(pial_complete['np']),
         'nuse_tri': pial_complete['ntri'],
         'nuse':     pial_complete['np'],
         'vertno':   np.arange(0,pial_complete['np']),
         'subject_his_id': recon_all_name,
         'dist': None,
         'dist_limit': None,
         'nearest': None,
         'type': 'surf',
         'nearest_dist': None,
         'pinfo': None,
         'patch_inds': None,
         'id': 101, # (FIFFV_MNE_SURF_LEFT_HEMI), # shouldn't matter, since we combined both hemispheres into one object
         'coord_frame': 5} # (FIFFV_COORD_MRI)}

src = mne.SourceSpaces([src])

#%% 
# =============================================================================
# save files for TVB
# =============================================================================
# get region map for source space (ie. downsampled pial), via nearest neighbour interpolation
n_vert   = pial_complete['np']

region_map_lores = np.zeros((n_vert))

vert_hires = pial_vertices
vert_lores = pial_complete['rr']

# serach nearest neighbour
idx = spatial.KDTree(vert_hires).query(vert_lores)[1]
region_map_lores = region_map[idx]

np.savetxt(tvb_output+"/sub-"+participant_label+"_region_mapping.txt", region_map_lores, fmt="%i")
print("Regionmap saved !")
# MODIFICHE X CORTEX NOHOLES: METTO 0 ALLA PROJECTION MATRIX DOVE CI SONO I VERTEX CORRISPONDENTI ALLE REGIONS DUMMY

leadfield_new=np.array(sio.loadmat(leadfield_path+"/sub-"+participant_label+"_EEGProjection_wdummy.mat")['ProjectionMatrix']) # load the file from Brainstorm
for i in range(len(region_map_lores)):
    if region_map_lores[i]==len(region_names)-1 or region_map_lores[i]==len(region_names)-2:
        leadfield_new[:,i]=0


# write leadfield to file
sio.savemat(tvb_output+"/sub-"+participant_label+"_EEGProjection.mat", mdict={'ProjectionMatrix':leadfield_new})

# leadfield for BIDS
BIDS_eeg_folder = input_dir+"/derivatives/"+pipeline_name+"/sub-"+participant_label+"/eeg"
if not os.path.exists(BIDS_eeg_folder):
    os.makedirs(BIDS_eeg_folder)
np.savetxt(BIDS_eeg_folder+"/sub-"+participant_label+"_desc-eeg_proj.tsv", leadfield_new, delimiter="\t")


# save in BIDS format
BIDS_anat_folder = input_dir+"/derivatives/"+pipeline_name+"/sub-"+participant_label+"/anat/"
if not os.path.exists(BIDS_anat_folder):
    os.makedirs(BIDS_anat_folder)

# create gii label table
gii_labeltb = nbg.GiftiLabelTable()

for i in range(len(region_names)):
    gii_label = nbg.GiftiLabel(key=i,alpha=1, 
                               red   = np.random.uniform(0,1,1)[0],
                               green = np.random.uniform(0,1,1)[0],
                               blue  = np.random.uniform(0,1,1)[0],
                              )
    gii_label.label = region_names[i]
    gii_labeltb.labels.append(gii_label)

darrays = [nbg.GiftiDataArray(region_map_lores.astype("int32"), intent="NIFTI_INTENT_LABEL", datatype=8)]
gii_image = nbg.GiftiImage(darrays=darrays, labeltable=gii_labeltb)
#nbg.giftiio.write(gii_image, BIDS_anat_folder+"/sub-"+participant_label+"_space-individual_dparc.label.gii") # giftiio.write deprecated by Nibabel 3.2.2, use nibabel.save() https://nipy.org/nibabel/changelog.html and https://neurostars.org/t/save-functional-data-as-surface-with-nilearn/21945/2
nib.save(gii_image, BIDS_anat_folder+"/sub-"+participant_label+"_space-individual_dparc.label.gii")    


# write cortical surface (i.e. source space) to file
cort_surf_path = tvb_output+"/sub-"+participant_label+"_Cortex/"
if not os.path.exists(cort_surf_path):
    os.makedirs(cort_surf_path)

# surface vertices are in ras-tkr coordinates used by freesurfer
# for them to allign with parc_image, use affine transform to bring them into ras-scanner
p = Popen(('mri_info --tkr2scanner '+recon_all_dir+"/"+recon_all_name+"/mri/aparc+aseg.mgz").split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate(b"input data that is passed to subprocess' stdin")
affine_xfm = np.array([ i.split() for i in str(output, "utf-8").splitlines()], dtype="float")

pial_vert_converted = affine_xfm.dot(np.concatenate((pial_complete['rr'] * 1000 ,np.ones((pial_complete['rr'].shape[0],1))), axis=1).T)[:3,:].T

# save
np.savetxt(cort_surf_path+"triangles.txt", pial_complete['tris'], fmt="%i")
np.savetxt(cort_surf_path+"vertices.txt", pial_vert_converted, fmt="%f")
np.savetxt(cort_surf_path+"normals.txt", pial_complete['nn'], fmt="%f")


# zip files
shutil.make_archive(cort_surf_path[:-1], 'zip', cort_surf_path)
print("Cortical surface zipped !")
shutil.rmtree(cort_surf_path)


# save in BIDS format
darrays = [nbg.GiftiDataArray(pial_vert_converted.astype("float32"), intent="NIFTI_INTENT_POINTSET")] + [nbg.GiftiDataArray(pial_complete['tris'].astype("int32"), intent="NIFTI_INTENT_TRIANGLE")]
gii_image = nbg.GiftiImage(darrays=darrays)
#nbg.giftiio.write(gii_image, BIDS_anat_folder+"/sub-"+participant_label+"_space-individual_pial.surf.gii")
nib.save(gii_image, BIDS_anat_folder+"/sub-"+participant_label+"_space-individual_pial.surf.gii")  # giftiio.write deprecated by Nibabel 3.2.2, use nibabel.save() https://nipy.org/nibabel/changelog.html and https://neurostars.org/t/save-functional-data-as-surface-with-nilearn/21945/2



# write BEM surfaces too file
#names = ["inner_skull_surface", "outer_skull_surface", "outer_skin_surface"] # "brain_surface",
names = ["outer_skin"] # # to read the surface from Brainstorm
BIDS_names = ["scalp"]
for i in range(len(names)) :
    name = names[i]
    BIDS_name = BIDS_names[i]
    # make dir
    bem_path = tvb_output+"/sub-"+participant_label+"_"+name+"/"
    if not os.path.exists(bem_path):
        os.makedirs(bem_path)
        
    #bem_surf = mne.read_surface(recon_all_dir+"/"+recon_all_name+"/bem/watershed/"+recon_all_name+"_"+name)
    #bem_surf = mne.read_surface(leadfield_path+"/"+name) # to read the surfaces modified in Blender in the /bem folder and not the original surfaces from FreeSurfer
    #bem_dict = {'rr':bem_surf[0], 'tris':bem_surf[1]}
    #bem_dict = {'rr':np.loadtxt("/Users/michelangelo/vertices.txt"), 'tris':np.loadtxt("/Users/michelangelo/triangles.txt", dtype=int)-1}
    bem_dict = {'rr':np.array(sio.loadmat(leadfield_path+"/sub-"+participant_label+"_outer_skin.mat")['vertices']), 'tris':np.array(sio.loadmat(leadfield_path+"/sub-"+participant_label+"_outer_skin.mat")['triangles'],dtype=int)-1}
    bem_complete = mne.surface.complete_surface_info(bem_dict)
    
    #bem_vert_converted = affine_xfm.dot(np.concatenate((bem_complete['rr'] ,np.ones((bem_complete['rr'].shape[0],1))), axis=1).T)[:3,:].T

    
    # save files
    np.savetxt(bem_path+"triangles.txt", bem_complete['tris'], fmt="%i")
    #np.savetxt(bem_path+"vertices.txt", bem_vert_converted, fmt="%f")
    np.savetxt(bem_path+"vertices.txt", bem_complete['rr'], fmt="%f")
    np.savetxt(bem_path+"normals.txt", bem_complete['nn'], fmt="%f")
    
    # zip folder
    shutil.make_archive(bem_path[:-1], 'zip', bem_path)
    shutil.rmtree(bem_path)

    # save for BIDS
    #darrays = [nbg.GiftiDataArray(bem_vert_converted.astype("float32"), intent="NIFTI_INTENT_POINTSET")] + [nbg.GiftiDataArray(bem_complete['tris'].astype("int32"), intent="NIFTI_INTENT_TRIANGLE")]
    darrays = [nbg.GiftiDataArray(bem_complete['rr'].astype("float32"), intent="NIFTI_INTENT_POINTSET")] + [nbg.GiftiDataArray(bem_complete['tris'].astype("int32"), intent="NIFTI_INTENT_TRIANGLE")]
    gii_image = nbg.GiftiImage(darrays=darrays)
    #nbg.giftiio.write(gii_image, BIDS_anat_folder+"/sub-"+participant_label+"_space-individual_" + BIDS_name + ".surf.gii") # giftiio.write deprecated by Nibabel 3.2.2, use nibabel.save() https://nipy.org/nibabel/changelog.html and https://neurostars.org/t/save-functional-data-as-surface-with-nilearn/21945/2
    nib.save(gii_image, BIDS_anat_folder+"/sub-"+participant_label+"_space-individual_" + BIDS_name + ".surf.gii")

print("BEM surfaces saved  !")

# save eeg_locations, are in ras-tkr coordinates used by freesurfer
# for them to allign with parc_image, use affine transform to bring them into ras-scanner
# eegp_loc_converted = affine_xfm.dot(np.concatenate((eegp_loc * 1000 ,np.ones((eegp_loc.shape[0],1))), axis=1).T)[:3,:].T

# f = open(tvb_output+"/sub-"+participant_label+"_EEG_Locations.txt", "w")
# f_bids = open(BIDS_eeg_folder+"/sub-"+participant_label+"_task-simulation_electrodes.tsv", "w")
# f_bids.write("name\tx\ty\tz\n")
# for i in range((np.array(ch_type)=="eeg").sum()): # write only "eeg" electrodes (not "misc")
#     f.write(mon.ch_names[i]+" "+"%.6f" %eegp_loc_converted[i,0]+" "+"%.6f" %eegp_loc_converted[i,1]+" "+"%.6f" %eegp_loc_converted[i,2]+"\n")
#     f_bids.write(mon.ch_names[i]+"\t"+"%.6f" %eegp_loc_converted[i,0]+"\t"+"%.6f" %eegp_loc_converted[i,1]+"\t"+"%.6f" %eegp_loc_converted[i,2]+"\n")
# f.close()
# f_bids.close()
shutil.copyfile(leadfield_path+"/sub-01_EEG_Locations.txt",tvb_output+"/sub-"+participant_label+"_EEG_Locations.txt")
if os.path.exists(leadfield_path+"/sub-01_task-simulation_electrodes.tsv"):
    shutil.copyfile(leadfield_path+"/sub-01_task-simulation_electrodes.tsv",BIDS_eeg_folder+"/sub-"+participant_label+"_task-simulation_electrodes.tsv")

print("EEG locations saved  !")


# create and save connectome.zip
tvb_connectome_path = tvb_output+"/sub-"+participant_label+"_Connectome/"
if not os.path.exists(tvb_connectome_path):
    os.makedirs(tvb_connectome_path)

# 1 weights, set diagonal to zero and make it symmetric
weights = np.genfromtxt(weights_path, delimiter=",")
# MODIFICHE X CORTEX NOHOLES: AGGIUNGE LE DUE DUMMY REGIONS
col_zero_w=np.zeros((weights.shape[0],2))
row_zero_w=np.zeros((2,weights.shape[1]+2))
weights = np.append(weights, col_zero_w, axis=1) 
weights = np.append(weights, row_zero_w, axis=0)
weights[np.diag_indices_from(weights)] = 0
i_lower = np.tril_indices_from(weights, -1)
weights[i_lower] = weights.T[i_lower]
n_regions = weights.shape[0] 
np.savetxt(tvb_connectome_path+"weights.txt", weights, delimiter="\t")

print("Weights saved  !")

# 2 tracts, set diagonal to zero and make it symmetric
tracts  = np.genfromtxt(tracts_path, delimiter=",")
# MODIFICHE X CORTEX NOHOLES: AGGIUNGE LE DUE DUMMY REGIONS
col_zero_t=np.zeros((tracts.shape[0],2))
row_zero_t=np.zeros((2,tracts.shape[1]+2))
tracts = np.append(tracts, col_zero_t, axis=1) 
tracts = np.append(tracts, row_zero_t, axis=0)
tracts[np.diag_indices_from(tracts)] = 0
i_lower = np.tril_indices_from(tracts, -1)
tracts[i_lower] = tracts.T[i_lower]
np.savetxt(tvb_connectome_path+"tract_lengths.txt", tracts, delimiter="\t")
print("Tracts saved !")

#3. centers
# create centroids
img = nib.load(parc_image) 
img_data = img.get_fdata().astype('int')

# get the right coordinate transform to align region centroids with the surfaces
f = open(tvb_connectome_path+"centres.txt","w")

# centers for BIDS
f_bids = open(BIDS_anat_folder+"/sub-" + participant_label + "_desc-centroid_morph.tsv","w")
f_bids.write("name\tcentroid-x\tcentroid-y\tcentroid-z\n")

for i in range(img_data.max()):
    rows, cols, slices = np.where(img_data==i+1)
    r = rows.mean()
    c = cols.mean()
    s = slices.mean()
    
    center = img.affine.dot(np.array([r,c,s,1]).T)[:3]
    f.write(region_names[i]+" %.6f" %center[0]+" %.6f" %center[1]+" %.6f" %center[2]+"\n")
    f_bids.write(region_names[i]+"\t%.6f" %center[0]+"\t%.6f" %center[1]+"\t%.6f" %center[2]+"\n")

# MODIFICHE X CORTEX NOHOLES: AGGIUNGE LE DUE DUMMY REGIONS CON COORDINATE LEFT -1,0,0 e RIGHT 1,0,0 NELLE PROSSIME 4 RIGHE
f.write(region_names[len(region_names)-2]+" %.6f" %-1+" %.6f" %0+" %.6f" %0+"\n")
f_bids.write(region_names[len(region_names)-2]+"\t%.6f" %-1+"\t%.6f" %0+"\t%.6f" %0+"\n")
f.write(region_names[len(region_names)-1]+" %.6f" %1+" %.6f" %0+" %.6f" %0+"\n")
f_bids.write(region_names[len(region_names)-1]+"\t%.6f" %1+"\t%.6f" %0+"\t%.6f" %0+"\n")
f.close()
f_bids.close()
print("Centers saved !")

# 4 orientation
# First get all Vertex-Normals corresponding to the Vertices of a Region 
# Now compute mean Vector and Normalize the Vector
# for subcortical regions set [0,0,1]
orientation = np.zeros((n_regions,3))

for i in range(n_regions):
    if cortical[i]: # cortical regions
        # select normal vertices of a region, average and normalize them
        nn  = pial_complete['nn'][ region_map_lores==i ,:]
        orientation[i,:] = nn.mean(axis=0)/np.linalg.norm(nn.mean(axis=0))
        
    elif not cortical[i]:  # subcortical regions
        orientation[i,:] = np.array([0,0,1])

np.savetxt(tvb_connectome_path+"average_orientations.txt", orientation, fmt="%f")
print("Orientations saved !")

# 5 area
# I'm not quite sure how to get to the exact value for the surface in mm^2
# so for now i just count the surface vertices corresponding to each region
# EDIT: According to the TVB Dokumentation, this attribute is not mandatory
# for the Input!
area = np.zeros((n_regions,1))
for i in range(n_regions):
    if cortical[i]: # cortical regions
        area[i] = np.sum(region_map_lores==i) 

    elif not cortical[i]:  # subcortical regions 
        area[i] = 0
    
np.savetxt(tvb_connectome_path+"areas.txt", area, fmt="%f")
print("Area saved !")

# 6 cortical
# connectivity cortical/non-cortical region flags; text file containing one boolean value on each line 
# (as 0 or 1 value) being 1 when corresponding region is cortical.
# due to error in configuring projection matrix in EEG, see monitors.py, class Projection, def config_for_sim
# this would need to get fixed, otherwise I don't know how to define the cortical variable or the lead field matrix
# therefor for now declare all regions as cortical
if parcellation == "desikan":
    np.savetxt(tvb_connectome_path+"cortical.txt", cortical, fmt="%i")
    
elif parcellation =="destrieux": 
    np.savetxt(tvb_connectome_path+"cortical.txt", cortical, fmt="%i")
    
elif parcellation=="hcpmmp1":
    np.savetxt(tvb_connectome_path+"cortical.txt", cortical, fmt="%i")
#cortical = np.ones((n_regions,1)).astype('int') 
#np.savetxt(tvb_connectome_path+"cortical.txt", cortical, fmt="%i")
print("Cortical saved !")


# 7 hemisphere
# text file containing one boolean value on each line 
# (as 0 or 1 value) being 1 when corresponding region is in the right hemisphere and 0 when in left hemisphere.
np.savetxt(tvb_connectome_path+"hemispheres.txt", hemisphere, fmt="%i")
print("Hemisphere saved !")

# zip all files
shutil.make_archive(tvb_connectome_path[:-1], 'zip', tvb_connectome_path)
print("Connectome zipped !")
shutil.rmtree(tvb_connectome_path)

# connectome for BIDS
BIDS_connectivity_folder = input_dir+"/derivatives/"+pipeline_name+"/sub-"+participant_label+"/connectivity"
if not os.path.exists(BIDS_connectivity_folder):
    os.makedirs(BIDS_connectivity_folder)
np.savetxt(BIDS_connectivity_folder+"/sub-"+participant_label+"_desc-weight_conndata-network_connectivity.tsv", weights, delimiter="\t")
np.savetxt(BIDS_connectivity_folder+"/sub-"+participant_label+"_desc-distance_conndata-network_connectivity.tsv", tracts, delimiter="\t")

# create additional json
conn_json = {}
conn_json["description"] = "Structural connectome, weights and distances (mm) derived from tractography"
conn_json["source_node_info"] = {"parcellation" : parcellation}

with open(BIDS_connectivity_folder+"/sub-"+participant_label+"_conndata-network_connectivity.json", 'w') as outfile:
    json.dump(conn_json, outfile)


# fMRI not available, hence below is commented...if available uncomment
# # fMRI ROI ts for BIDS
# BIDS_func_folder = input_dir+"/derivatives/"+pipeline_name+"/sub-"+participant_label+"/func"
# if not os.path.exists(BIDS_func_folder):
#     os.makedirs(BIDS_func_folder)

# ROI_ts = np.genfromtxt(fMRI_ROI_ts)
# df = pd.DataFrame(ROI_ts,columns=region_names)
# df.to_csv(BIDS_func_folder+"/sub-" + participant_label + "_task-" + task_name + "_atlas-" + parcellation + "_timeseries.tsv", sep = '\t', index=False)

# 8 conduction velocity, set diagonal to zero and make it symmetric
velocities  = np.genfromtxt(velocities_path, delimiter=",")
# MODIFICHE X CORTEX NOHOLES: AGGIUNGE LE DUE DUMMY REGIONS
col_zero_v=np.zeros((velocities.shape[0],2))
row_zero_v=np.zeros((2,velocities.shape[1]+2))
velocities = np.append(velocities, col_zero_v, axis=1) 
velocities = np.append(velocities, row_zero_v, axis=0)
velocities[np.diag_indices_from(velocities)] = 0
i_lower_v = np.tril_indices_from(velocities, -1)
velocities[i_lower_v] = velocities.T[i_lower_v]
np.savetxt(tvb_output+"/sub-"+participant_label+"_conduction_velocities.txt", velocities, delimiter="\t")
print("Velocities saved !")

print("TVBconverter has finished !")
