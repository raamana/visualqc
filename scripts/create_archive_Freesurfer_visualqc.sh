

# This simple script creates an archive of just the few files (for each subject) necessary for VisualQC
# from the Freesurfer Output structure

# Often Freesurfer processing is done on a HPC or another remote system, which may not have a display attached to it.
# So visualqc can't be run there directly. This may necessiate the copying the FS outputs down to the local computer.

# Copying the entire FS output is not only quite a hassle (huge size, slow transfer, not enough space locally etc),
# but also not necessary. Hence, this bash script helps by creating a tar or zip file of the just the files needed, which would help making that processing by saving time and disk space.

# Go to the Freesurfer SUBJECTS_DIR or where the all subjects are stored
# cd to it
# Run the following command, modifying it as necessary e.g. You can replace * with a more elaborate regex to select only a specific subset of subjects, and change the name of the output file to identify the source dataset, FS version etc


tar -cjvfh CompressedFileName.zip */{mri/orig.mgz,mri/aparc+aseg.mgz,surf/?h.pial}
