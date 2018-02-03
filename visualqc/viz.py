
__all__ = ['aseg_on_mri']

from mrivis.utils import check_params, crop_to_seg_extents, read_image, pick_slices
from mrivis.color_maps import get_freesurfer_cmap
from visualqc.utils import get_axis

import numpy as np
from matplotlib import pyplot as plt, colors, cm
import matplotlib as mpl


def aseg_on_mri(mri_spec,
                aseg_spec,
                alpha_mri=0.7,
                alpha_seg=0.7,
                num_rows=2,
                num_cols=6,
                rescale_method='global',
                aseg_cmap='freesurfer',
                sub_cortical=False,
                annot=None,
                padding=5,
                bkground_thresh=0.05,
                output_path=None,
                figsize=None,
                **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    num_rows, num_cols, padding = check_params(num_rows, num_cols, padding)

    mri = read_image(mri_spec, bkground_thresh=bkground_thresh)
    seg = read_image(aseg_spec, bkground_thresh=0)
    mri, seg = crop_to_seg_extents(mri, seg, padding)

    slices = pick_slices(mri.shape, num_rows, num_cols)

    plt.style.use('dark_background')

    num_axes = 3
    if figsize is None:
        figsize = [5 * num_axes * num_rows, 5 * num_cols]
    fig, ax = plt.subplots(num_axes * num_rows, num_cols, figsize=figsize)

    # displaying some annotation text if provided
    if annot is not None:
        fig.suptitle(annot, backgroundcolor='black', color='g')

    display_params_mri = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_mri)
    display_params_seg = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_seg)

    normalize_labels = colors.Normalize(vmin=seg.min(), vmax=seg.max(), clip=True)
    fs_cmap = get_freesurfer_cmap(sub_cortical)
    seg_mapper = cm.ScalarMappable(norm=normalize_labels, cmap=fs_cmap)

    normalize_mri = colors.Normalize(vmin=mri.min(), vmax=mri.max(), clip=True)
    mri_mapper = cm.ScalarMappable(norm=normalize_mri, cmap='gray')

    axes_seg = list()
    axes_mri = list()

    ax = ax.flatten()
    ax_counter = 0
    for dim_index in range(3):
        for counter, slice_num in enumerate(slices[dim_index]):
            plt.sca(ax[ax_counter])
            ax_counter = ax_counter + 1

            slice_mri = get_axis(mri, dim_index, slice_num)
            slice_seg = get_axis(seg, dim_index, slice_num)

            # # masking data to set no-value pixels to transparent
            # seg_background = np.isclose(slice_seg, 0.0)
            # slice_seg = np.ma.masked_where(seg_background, slice_seg)
            # slice_mri = np.ma.masked_where(np.logical_not(seg_background), slice_mri)

            seg_rgb = seg_mapper.to_rgba(slice_seg)
            mri_rgb = mri_mapper.to_rgba(slice_mri)

            plt.imshow(seg_rgb, **display_params_seg)
            plt.imshow(mri_rgb, **display_params_mri)
            plt.axis('off')


    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.subplots_adjust(left  =0.01, right  =0.99,
                        bottom=0.01,   top  =0.99,
                        wspace=0.05 , hspace=0.02)
    # fig.tight_layout()

    if output_path is not None:
        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    # plt.close()

    return fig
