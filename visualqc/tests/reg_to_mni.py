import nibabel as nib
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (RigidTransform3D,
                                   AffineTransform3D)


path_mni_T1 = '/Users/Reddy/dev/visualqc/atlases/mni_icbm152_nl_VI_nifti/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii'
path_sub_T1 = '/Users/Reddy/dev/visualqc/example_datasets/id_001/mri/orig.mgz'

static_hdr = nib.load(path_mni_T1)
moving_hdr = nib.load(path_sub_T1)

static = static_hdr.get_data()
moving = moving_hdr.get_data()

static_grid2world = static_hdr.affine
moving_grid2world = moving_hdr.affine

nbins = 50
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [2000, 200, 20]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

rigid_transform = RigidTransform3D()
params0_rigid = None
starting_affine = None
rigid = affreg.optimize(static, moving,
                        rigid_transform, params0_rigid,
                        static_grid2world, moving_grid2world,
                        starting_affine=starting_affine)

affine_transform = AffineTransform3D()
params0_affine = None
affine = affreg.optimize(static, moving,
                         affine_transform, params0_affine,
                         static_grid2world, moving_grid2world,
                         starting_affine=rigid.affine)

transformed = affine.transform(moving)
print()
