#!bin/sh

# INPUTS:
# sub-01_T1w.nii.gz (T1w image)
# sub-01_T2w_FLAIR_coreg.nii.gz (T2w image)
# wmfod_norm.mif (white matter FOD from Tractography)
# mask.mif (mask of wmfod from Tractography)
# mean_b0.nii.gz (mean B0 from preprocessed unbiased DWI sub-01_den_preproc_unbiased.mif)
# tracks_71631K.tck (streamlines of Tractography)
# sift_71631K.txt (SIFT weights from Tractography)
# sub-01_parcels_coreg.mif (parcellated image coregistered with DWI)
# 5tt_coreg.mif (to create the pictures of gratio and velocity in White Matter only)

## FSL Setup...change path as needed
FSLDIR=/Users/michelangelo/fsl
PATH=${FSLDIR}/share/fsl/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh
export PATH="/usr/local/bin:$PATH"

# path to input files
input_dir_velocity=`pwd`
mkdir -p ${input_dir_velocity}/conduction_velocity_map

# Create T2w in .nii.gz format using MRtrix from original DICOM file (change name as needed)
mrconvert 80578042_3D_Sag_T2_FLAIR_anon.dcm conduction_velocity_map/sub-01_T2w_FLAIR.nii.gz

# coregister T2w to T1w using FreeSurfer (to overlap position, voxel size etc)
mri_coreg --mov conduction_velocity_map/sub-01_T2w_FLAIR.nii.gz --ref sub-01_T1w.nii.gz --reg conduction_velocity_map/reg.lta --threads 11
mri_vol2vol --reg conduction_velocity_map/reg.lta --mov conduction_velocity_map/sub-01_T2w_FLAIR.nii.gz --targ sub-01_T1w.nii.gz --o conduction_velocity_map/sub-01_T2w_FLAIR_coreg.nii.gz

# extract only brain of T1w and T2w with their mask (assuming T2w is already coregistered to T1w)...USE SAME SETTINGS AS SYNB0-DISCO
bet sub-01_T1w.nii.gz conduction_velocity_map/T1.nii.gz -f 0.1 -g -0.2 -m
bet conduction_velocity_map/sub-01_T2w_FLAIR_coreg.nii.gz conduction_velocity_map/T2.nii.gz -f 0.1 -g -0.2 -m

# find min and max of T1 and T2 and store as variables, together with max-min
read T1min T1max <<< $(fslstats conduction_velocity_map/T1.nii.gz -R)
read T2min T2max <<< $(fslstats conduction_velocity_map/T2.nii.gz -R)
T1max_T1min=$(echo $T1max - $T1min | /usr/bin/bc ) 
T2max_T2min=$(echo $T2max - $T2min | /usr/bin/bc ) 


# rescale to 0-100 T1 and T2
fslmaths conduction_velocity_map/T1.nii.gz -sub $T1min -div $T1max_T1min -mul 100.0 conduction_velocity_map/T1_rescaled
fslmaths conduction_velocity_map/T2.nii.gz -sub $T2min -div $T2max_T2min -mul 100.0 conduction_velocity_map/T2_rescaled

# compute T1/T2 and store in a dedicated folder
mkdir conduction_velocity_map/T1T2_maps
fslmaths conduction_velocity_map/T1_rescaled.nii.gz -div conduction_velocity_map/T2_rescaled.nii.gz conduction_velocity_map/T1T2_maps/T1T2_map.nii.gz

#uses mrtrix and fsl commands at the command line or wrapped in a loop/script
#replace commands below with your ${subject} ID and preferred organization

mrconvert mask.mif conduction_velocity_map/mask.nii.gz

#generate fixels from single image
fod2fixel wmfod_norm.mif conduction_velocity_map/fixel_output -mask conduction_velocity_map/mask.nii.gz -peak peaks.mif -dirpeak -afd afd.mif -disp disp.mif

fixel2voxel conduction_velocity_map/fixel_output/afd.mif sum conduction_velocity_map/fixel_output/afd_sum.nii.gz

# COREGISTER AND REGRID afd_sum.nii.gz to T1/T2 image 
# use the flirt command to coregister the two datasets; This command uses the T1T2 map as the reference image, meaning that it stays stationary. 
# The ${subject}_afd_sum is then moved to find the best fit with the T1T2 map. The output of this command, “afd_sum2T1T2_fsl.mat ”, contains the transformation matrix that 
# was used to overlay the ${subject}_afd_sum on top of the T1T2 map
flirt -in conduction_velocity_map/fixel_output/afd_sum.nii.gz -ref conduction_velocity_map/T1T2_maps/T1T2_map.nii.gz -interp nearestneighbour -dof 6 -omat conduction_velocity_map/afd_sum2T1T2_fsl.mat
# Converting the transformation matrix to MRtrix format:
transformconvert conduction_velocity_map/afd_sum2T1T2_fsl.mat conduction_velocity_map/fixel_output/afd_sum.nii.gz conduction_velocity_map/T1T2_maps/T1T2_map.nii.gz flirt_import conduction_velocity_map/afd_sum2T1T2_mrtrix.txt 
# Applying the transformation matrix to the afd_sum image and regrid it to have same voxels of T1/T2:
mrtransform conduction_velocity_map/T1T2_maps/T1T2_map.nii.gz -linear conduction_velocity_map/afd_sum2T1T2_mrtrix.txt  -inverse conduction_velocity_map/T1T2_maps/T1T2_map_coreg.nii.gz
mrgrid conduction_velocity_map/fixel_output/afd_sum.nii.gz regrid -template conduction_velocity_map/T1T2_maps/T1T2_map_coreg.nii.gz conduction_velocity_map/fixel_output/afd_sum_resampled.nii.gz


# as per stikov et al 2015, by Mohammadi & Callaghan 2021 sqrt of 1 - mvf/avf+mvf
fslmaths conduction_velocity_map/T1T2_maps/T1T2_map_coreg.nii.gz -add conduction_velocity_map/fixel_output/afd_sum_resampled.nii.gz conduction_velocity_map/MVF_FVF.nii.gz
fslmaths conduction_velocity_map/T1T2_maps/T1T2_map_coreg.nii.gz -div conduction_velocity_map/MVF_FVF.nii.gz conduction_velocity_map/MVF_div_FVF.nii.gz

# this is just an easy way to make a voxel-wise map where each voxel has a value of 1 but the grid matches your image:
fslmaths conduction_velocity_map/MVF_div_FVF.nii.gz -mul 0 -add 1 conduction_velocity_map/1.nii.gz 

fslmaths conduction_velocity_map/1.nii.gz -sub conduction_velocity_map/MVF_div_FVF.nii.gz conduction_velocity_map/1_sub_MVFAVF.nii.gz
rm conduction_velocity_map/1.nii.gz

mkdir conduction_velocity_map/gratio
fslmaths conduction_velocity_map/1_sub_MVFAVF.nii.gz -sqrt conduction_velocity_map/gratio/gratio.nii.gz

# This cleans up any voxels (typically around the outside of the brain or on the periphery, you can just remask if you prefer or alter the value)
fslmaths conduction_velocity_map/gratio/gratio.nii.gz -uthr 0.95 conduction_velocity_map/gratio/gratio_masked.nii.gz

#calculate conduction velocity from gratio according AVF*sqrt(-ln(gratio)) Ruston via Berman, Filo & Mezer Modeling conduction delays in the corups callosum using MRI-measured g-ratio
mkdir conduction_velocity_map/conduction_velocity
fslmaths conduction_velocity_map/gratio/gratio_masked.nii.gz -log -mul -1 -sqrt -mul conduction_velocity_map/fixel_output/afd_sum_resampled.nii.gz conduction_velocity_map/conduction_velocity/conduction_velocity.nii.gz

# coregister conduction velocity to DWI (this in order to have velocity coregistered with tractography streamlines tracks.tck)
flirt -in mean_b0.nii.gz -ref conduction_velocity_map/conduction_velocity/conduction_velocity.nii.gz -interp nearestneighbour -dof 6 -omat conduction_velocity_map/diff2velocity_fsl.mat
transformconvert conduction_velocity_map/diff2velocity_fsl.mat mean_b0.nii.gz conduction_velocity_map/conduction_velocity/conduction_velocity.nii.gz flirt_import conduction_velocity_map/diff2velocity_mrtrix.txt
mrtransform conduction_velocity_map/conduction_velocity/conduction_velocity.nii.gz -linear conduction_velocity_map/diff2velocity_mrtrix.txt -inverse conduction_velocity_map/conduction_velocity/conduction_velocity_coreg.nii.gz

# calculate the matrix of conduction velocity
tcksample tracks_71631K.tck conduction_velocity_map/conduction_velocity/conduction_velocity_coreg.nii.gz conduction_velocity_map/mean_Velocity_per_streamline.csv -stat_tck mean
tck2connectome -symmetric -zero_diagonal -tck_weights_in sift_71631K.txt tracks_71631K.tck sub-01_parcels_coreg.mif sub-01_parcels_coreg_meanvelocity.csv -scale_file conduction_velocity_map/mean_Velocity_per_streamline.csv -stat_edge mean


# The below extracts the gratio and conduction velocity only in white matter to create pictures
# first extract the volume 2 of 5tt (white matter)
mkdir -p conduction_velocity_map/maps_with_only_white_matter/gratio
mkdir -p conduction_velocity_map/maps_with_only_white_matter/conduction_velocity
mrconvert 5tt_coreg.mif conduction_velocity_map/maps_with_only_white_matter/5tt_coreg.nii.gz
fslroi conduction_velocity_map/maps_with_only_white_matter/5tt_coreg.nii.gz conduction_velocity_map/maps_with_only_white_matter/5tt_vol2.nii.gz 2 1
# coreg gratio to DWI (this in order to have velocity coregistered with tractography streamlines tracks.tck)
mrtransform conduction_velocity_map/gratio/gratio_masked.nii.gz -linear conduction_velocity_map/diff2velocity_mrtrix.txt -inverse conduction_velocity_map/maps_with_only_white_matter/gratio/gratio_masked_coreg.nii.gz
# create map of voxels 0 for the -if operator in next command
fslmaths conduction_velocity_map/maps_with_only_white_matter/5tt_vol2.nii.gz -mul 0 conduction_velocity_map/maps_with_only_white_matter/5tt_zeros.nii.gz
# regrid voxels of 5tt to gratio to have same voxels
mrgrid conduction_velocity_map/maps_with_only_white_matter/5tt_vol2.nii.gz regrid -template conduction_velocity_map/maps_with_only_white_matter/gratio/gratio_masked_coreg.nii.gz conduction_velocity_map/maps_with_only_white_matter/5tt_vol2_resampled.nii.gz
mrgrid conduction_velocity_map/maps_with_only_white_matter/5tt_zeros.nii.gz regrid -template conduction_velocity_map/maps_with_only_white_matter/gratio/gratio_masked_coreg.nii.gz conduction_velocity_map/maps_with_only_white_matter/5tt_zeros_resampled.nii.gz
# then create a new image where the white matter is not zero (https://mrtrix.readthedocs.io/en/dev/reference/commands/mrcalc.html)...the -if operator 
# of mrcalc says to use the first image if the second is different from zero (the white matter part of 5tt) otherwise use the third image (the zero voxels created before)
mrcalc conduction_velocity_map/maps_with_only_white_matter/5tt_vol2_resampled.nii.gz conduction_velocity_map/maps_with_only_white_matter/gratio/gratio_masked_coreg.nii.gz conduction_velocity_map/maps_with_only_white_matter/5tt_zeros_resampled.nii.gz -if conduction_velocity_map/maps_with_only_white_matter/gratio/gratio_masked_coreg_WMonly.nii.gz
mrcalc conduction_velocity_map/maps_with_only_white_matter/5tt_vol2_resampled.nii.gz conduction_velocity_map/conduction_velocity/conduction_velocity_coreg.nii.gz conduction_velocity_map/maps_with_only_white_matter/5tt_zeros_resampled.nii.gz -if conduction_velocity_map/maps_with_only_white_matter/conduction_velocity/conduction_velocity_coreg_WMonly.nii.gz


