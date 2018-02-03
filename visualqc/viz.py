
__all__ = ['review_and_rate']

from mrivis.utils import check_params, crop_to_seg_extents, read_image, pick_slices
from mrivis.color_maps import get_freesurfer_cmap
from visualqc.utils import get_axis
from copy import copy, deepcopy

import numpy as np
from matplotlib import pyplot as plt, colors, cm
from matplotlib.widgets import RadioButtons, Slider
import matplotlib as mpl


def overlay_images(mri, seg, alpha_mri=0.7, alpha_seg=0.7,
                   num_rows=2, num_cols=6, figsize=None,
                   sub_cortical=False, annot=None, padding=5):
    """"""

    num_rows, num_cols, padding = check_params(num_rows, num_cols, padding)

    # mri = read_image(mri_spec, bkground_thresh=bkground_thresh)
    # seg = read_image(aseg_spec, bkground_thresh=0)
    mri, seg = crop_to_seg_extents(mri, seg, padding)

    slices = pick_slices(mri.shape, num_rows, num_cols)

    plt.style.use('dark_background')

    num_axes = 3
    if figsize is None:
        figsize = [4 * num_axes * num_rows, 4 * num_cols]
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

            seg_rgb = seg_mapper.to_rgba(slice_seg)
            mri_rgb = mri_mapper.to_rgba(slice_mri)

            handle_seg = plt.imshow(seg_rgb, **display_params_seg)
            handle_mri = plt.imshow(mri_rgb, **display_params_mri)
            plt.axis('off')

            axes_seg.append(handle_seg)
            axes_mri.append(handle_mri)

    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.subplots_adjust(left  =0.01, right  =0.9,
                        bottom=0.01,   top  =0.99,
                        wspace=0.05 , hspace=0.02)


    return fig, axes_mri, axes_seg
def review_and_rate(mri,
                    seg,
                    alpha_mri=0.7,
                    alpha_seg=0.7,
                    rating_list=('Good', 'Suspect', 'Bad', 'Failed', 'Later'),
                    num_rows=2,
                    num_cols=6,
                    rescale_method='global',
                    aseg_cmap='freesurfer',
                    sub_cortical=False,
                    annot=None,
                    padding=5,
                    output_path=None,
                    figsize=None,
                    **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    fig, axes_mri, axes_seg = overlay_images(mri, seg, alpha_mri=alpha_mri, alpha_seg=alpha_seg,
                                             figsize=figsize, num_rows=num_rows, num_cols=num_cols, padding=padding,
                                             sub_cortical=sub_cortical, annot=annot)

    def advance_to_next():
        """Callback to move to next image"""
        pass


    ax_radio = plt.axes([0.905, 0.8 , 0.085, 0.18], axisbg='#009b8c')
    radio_bt = RadioButtons(ax_radio, rating_list,
                            active=None, activecolor='orange')

    ax_quit  = plt.axes([0.905, 0.69, 0.085, 0.1], axisbg='#0084b4')
    quit_button = RadioButtons(ax_quit, ["Next", "Quit"],
                               active=None, activecolor='orange')

    for txt_lbl in quit_button.labels + radio_bt.labels:
        txt_lbl.set_color('#fff6da')
        txt_lbl.set_fontweight('bold')

    def save_and_advance(label):
        print('\t {}'.format(label))
        # TODO save rating
        plt.close()


    def quit_review(label):
        plt.close()
        if label.upper() == u'QUIT':
            print('User chosen to stop.')
            advance_to_next()

    radio_bt.on_clicked(save_and_advance)
    quit_button.on_clicked(quit_review)

    global rating, quit_now
    global latest_alpha_seg, prev_alpha_seg, axes_to_update
    latest_alpha_seg = deepcopy(alpha_seg)
    prev_alpha_seg = deepcopy(latest_alpha_seg)
    axes_to_update = copy(axes_seg)

    def on_mouse(event):
        """Callback for mouse events."""

        global latest_alpha_seg, prev_alpha_seg, axes_to_update

        print(event)
        print('alpha before action:\n prev {} latest {} global {}'.format(prev_alpha_seg, latest_alpha_seg, alpha_seg))
        if event.button in ['up', 'down', 3]:

            if event.button == 'up':
                latest_alpha_seg = latest_alpha_seg + event.step*0.025
            elif event.button == 'down':
                latest_alpha_seg = latest_alpha_seg - event.step*0.025
            elif event.button in [3]: # right click
                if latest_alpha_seg > 0.0:
                    prev_alpha_seg = copy(latest_alpha_seg)
                    latest_alpha_seg = 0.0
                else:
                    latest_alpha_seg = copy(prev_alpha_seg)

            # updating seg alpha for all axes
            for ax in axes_to_update:
                ax.set_alpha(latest_alpha_seg)

        elif event.dblclick:
            print('cue to move to next subject!')
        else:
            pass

        print('alpha AFTER action:\n prev {} latest {} global {}'.format(prev_alpha_seg, latest_alpha_seg, alpha_seg))

        return

    con_id_click  = fig.canvas.mpl_connect('button_press_event', on_mouse)
    # con_id_scroll = fig.canvas.mpl_connect('scroll_event', on_mouse)

    fig.set_size_inches(figsize)
    plt.show(block=True)

    if output_path is not None:
        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    fig.canvas.mpl_disconnect(con_id_click)
    # fig.canvas.mpl_disconnect(con_id_scroll)
    plt.close()

    return fig, rating, quit_now
