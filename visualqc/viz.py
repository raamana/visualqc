__all__ = ['review_and_rate']

import traceback
from os import makedirs
from os.path import join as pjoin, exists as pexists
from subprocess import check_call

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt, colors, cm
from matplotlib.widgets import RadioButtons, Slider
from mrivis.color_maps import get_freesurfer_cmap
from mrivis.utils import check_params, crop_to_seg_extents
from visualqc import config as cfg
from visualqc.config import zoomed_position, annot_vis_dir_name, default_vis_type, default_padding, \
    default_views, default_num_slices, default_num_rows, default_alpha_mri, default_alpha_seg, \
    default_rating_list, default_navigation_options
from visualqc.utils import get_axis, pick_slices, check_layout


def overlay_images(mri, seg, alpha_mri=default_alpha_seg, alpha_seg=default_alpha_seg,
                   vis_type=default_vis_type, contour_color=cfg.default_contour_face_color,
                   out_dir=None, fs_dir=None, subject_id=None,
                   views=default_views, num_slices_per_view=default_num_slices,
                   num_rows_per_view=default_num_rows, figsize=None,
                   annot=None, padding=default_padding,
                   output_path=None):
    """Backend engine for overlaying a given seg on MRI with freesurfer label."""

    num_rows_per_view, num_slices_per_view, padding = check_params(num_rows_per_view, num_slices_per_view, padding)
    mri, seg = crop_to_seg_extents(mri, seg, padding)

    surf_vis = dict()  # empty - no vis to include
    if 'cortical' in vis_type:
        if fs_dir is not None and subject_id is not None and out_dir is not None:
            surf_vis = make_vis_pial_surface(fs_dir, subject_id, out_dir)
    num_surf_vis = len(surf_vis)

    num_views = len(views)
    num_rows = num_rows_per_view * num_views
    slices = pick_slices(mri.shape, views, num_slices_per_view)
    num_volumetric_slices = len(slices)
    total_num_panels = num_volumetric_slices + num_surf_vis
    num_rows_for_surf_vis = 1 if num_surf_vis > 0 else 0
    num_rows = num_rows + num_rows_for_surf_vis
    num_cols = check_layout(total_num_panels, num_views, num_rows_per_view, num_rows_for_surf_vis)

    plt.style.use('dark_background')

    if figsize is None:
        # figsize = [min(15,4*num_rows), min(12,4*num_cols)] # max (15,12)
        figsize = [4 * num_rows, 2* num_cols]
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

    display_params_mri = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_mri)
    display_params_seg = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_seg)

    normalize_labels = colors.Normalize(vmin=seg.min(), vmax=seg.max(), clip=True)
    fs_cmap = get_freesurfer_cmap(vis_type)
    seg_mapper = cm.ScalarMappable(norm=normalize_labels, cmap=fs_cmap)

    normalize_mri = colors.Normalize(vmin=mri.min(), vmax=mri.max(), clip=True)
    mri_mapper = cm.ScalarMappable(norm=normalize_mri, cmap='gray')

    # deciding colors for the whole image
    unique_labels = np.unique(seg)
    # removing background - 0 stays 0
    unique_labels = np.delete(unique_labels, 0)
    if len(unique_labels) == 1:
        color4label = [contour_color]
    else:
        color4label = seg_mapper.to_rgba(unique_labels)

    handles_seg = list()
    handles_mri = list()

    ax = ax.flatten()
    # display surfaces
    for sf_counter, ((hemi, view), spath) in enumerate(surf_vis.items()):
        plt.sca(ax[sf_counter])
        img = mpimg.imread(spath)
        # img = crop_image(img)
        plt.imshow(img)
        ax[sf_counter].text(0, 0, '{} {}'.format(hemi, view))
        plt.axis('off')

    # display slices
    for ax_counter, (dim_index, slice_num) in enumerate(slices):
        plt.sca(ax[ax_counter + num_surf_vis])

        slice_mri = get_axis(mri, dim_index, slice_num)
        slice_seg = get_axis(seg, dim_index, slice_num)

        # display MRI
        mri_rgb = mri_mapper.to_rgba(slice_mri)
        h_mri = plt.imshow(mri_rgb, **display_params_mri)

        if 'volumetric' in vis_type:
            seg_rgb = seg_mapper.to_rgba(slice_seg)
            h_seg = plt.imshow(seg_rgb, **display_params_seg)
        elif 'contour' in vis_type:
            h_seg = plot_contours_in_slice(slice_seg, unique_labels, color4label)

        plt.axis('off')

        # # encoding the souce of the object (image/line) being displayed
        # handle_seg.set_label('seg {} {}'.format(dim_index, slice_num))
        # handle_mri.set_label('mri {} {}'.format(dim_index, slice_num))

        handles_mri.append(h_mri)
        if len(h_seg) >= 1:
            handles_seg.extend(h_seg)
        else:
            handles_seg.append(h_seg)

    # hiding unused axes
    for ua in range(total_num_panels, len(ax)):
        ax[ua].set_visible(False)

    # displaying some annotation text if provided
    if annot is not None:
        title_handle = fig.suptitle(annot, backgroundcolor='black', color='white', fontsize='large')
        title_handle.set_position(cfg.annot_position)

    fig.set_size_inches(figsize)

    if output_path is not None:
        # no space left unused
        plt.subplots_adjust(**cfg.no_blank_area)
        output_path = output_path.replace(' ', '_')
        layout_str = 'v{}_ns{}_{}x{}'.format(''.join([ str(v) for v in views]),num_slices_per_view,num_rows,num_cols)
        fig.savefig(output_path + '_{}.png'.format(layout_str), bbox_inches='tight')

    # leaving some space on the right for review elements
    plt.subplots_adjust(**cfg.review_area)

    return fig, handles_mri, handles_seg, figsize


def plot_contours_in_slice(slice_seg, unique_labels, color4label):
    """Returns a contour around the data in slice (after binarization)"""

    contour_handles = list()
    for index, label in enumerate(unique_labels):
        binary_slice_seg = slice_seg == label
        if not binary_slice_seg.any():
            continue

        # using pyplot-builtin contour
        ctr_h = plt.contour(binary_slice_seg, levels=[cfg.contour_level, ],
                          colors=(color4label[index],), linewidths=cfg.contour_line_width)
        contour_handles.append(ctr_h)

        # # skimage solution
        # contours = find_contours(binary_slice_seg, level=contour_level)
        # if len(contours) > 1:
        #     single_contour = join_contours(contours) # joining them, in case there are multiple
        # else:
        #     single_contour = contours[0]
        #
        # # display contours (notice the switch of x and y!)
        # ctr_h = plt.plot(single_contour[:, 1], single_contour[:, 0],
        #                           color=color4label[index], linewidth=contour_line_width)
        # contour_handles.append(ctr_h[0])

    return contour_handles


def join_contours(contour_list):
    """Joins multiple contour segments into a single object with line breaks"""

    clist_w_breaks = [cfg.line_break] * (2 * len(contour_list) - 1)
    clist_w_breaks[::2] = contour_list
    single_contour = np.vstack(clist_w_breaks)

    return single_contour


def make_vis_pial_surface(fs_dir, subject_id, out_dir, annot_file='aparc.annot'):
    """Generate screenshot for the pial surface in different views"""

    out_vis_dir = pjoin(out_dir, annot_vis_dir_name)
    makedirs(out_vis_dir, exist_ok=True)
    hemis = ('lh', 'rh')
    hemis_long = ('left', 'right')
    vis_list = dict()
    for hemi, hemi_l in zip(hemis, hemis_long):
        vis_list[hemi_l] = dict()
        script_file, vis_files = make_tcl_script_vis_annot(subject_id, hemi_l, out_vis_dir, annot_file)
        try:
            # run the script only if all the visualizations were not generated before
            all_vis_exist = all([pexists(vis_path) for vis_path in vis_files.values()])
            if not all_vis_exist:
                exit_code = run_tksurfer_script(fs_dir, subject_id, hemi, script_file)

            vis_list[hemi_l].update(vis_files)
        except:
            traceback.print_exc()
            print('unable to generate tksurfer visualizations for {} hemi - skipping'.format(hemi))

    # flattening it for easier use later on
    out_vis_list = dict()
    pref_order = [ ('right', 'lateral'), ('left', 'lateral'),
                   ('right', 'medial'), ('left', 'medial'),
                   ('right', 'transverse'), ('left', 'transverse')]
    for hemi_l, view in pref_order:
        if pexists(vis_list[hemi_l][view]):
            out_vis_list[(hemi_l, view)] = vis_list[hemi_l][view]

    return out_vis_list


def make_tcl_script_vis_annot(subject_id, hemi, out_vis_dir,
                              annot_file='aparc.annot'):
    """Generates a tksurfer script to make visualizations"""

    script_file = pjoin(out_vis_dir, 'vis_annot_{}.tcl'.format(hemi))
    vis = dict()
    for view in ['lateral', 'medial', 'transverse']:
        vis[view] = pjoin(out_vis_dir, '{}_{}_{}.tif'.format(subject_id, hemi, view))

    img_format = 'tiff'  # rgb does not work

    cmds = list()
    # cmds.append("resize_window 1000")
    cmds.append("labl_import_annotation {}".format(annot_file))
    cmds.append("scale_brain 1.37")
    cmds.append("redraw")
    cmds.append("save_{} {}".format(img_format, vis['lateral']))
    cmds.append("rotate_brain_y 180.0")
    cmds.append("redraw")
    cmds.append("save_{} {}".format(img_format, vis['medial']))
    cmds.append("rotate_brain_z -90.0")
    cmds.append("rotate_brain_y 135.0")
    cmds.append("redraw")
    cmds.append("save_{} {}".format(img_format, vis['transverse']))
    cmds.append("exit 0")

    try:
        with open(script_file, 'w') as sf:
            sf.write('\n'.join(cmds))
    except:
        raise IOError('Unable to write the script file to\n {}'.format(script_file))

    return script_file, vis


def run_tksurfer_script(fs_dir, subject_id, hemi, script_file):
    """Runs a given TCL script to generate visualizations"""

    exit_code = check_call(['tksurfer', '-sdir', fs_dir, subject_id, hemi, 'pial', '-tcl', script_file], shell=False)

    return exit_code


class ReviewInterface(object):
    """Class to layout interaction elements and define callbacks. """

    def __init__(self, fig, axes_seg, axes_mri, alpha_seg,
                 rating_list,
                 navig_options=default_navigation_options):
        "Constructor."

        self.fig = fig
        self.axes_seg = axes_seg
        self.axes_mri = axes_mri
        self.latest_alpha_seg = alpha_seg

        self.user_rating = None
        self.quit_now = False

        self.zoomed_in = False
        self.prev_axis = None
        self.prev_ax_pos = None

        self.rating_list = rating_list
        ax_radio = plt.axes(cfg.position_rating_axis, facecolor=cfg.color_rating_axis, aspect='equal')
        self.radio_bt_rating = RadioButtons(ax_radio, self.rating_list,
                                            active=None, activecolor='orange')

        ax_quit = plt.axes(cfg.position_navig_options, facecolor=cfg.color_quit_axis, aspect='equal')
        self.radio_bt_quit = RadioButtons(ax_quit, navig_options,
                                          active=None, activecolor='orange')

        ax_slider = plt.axes(cfg.position_slider_seg_alpha, facecolor=cfg.color_slider_axis)
        self.slider = Slider(ax_slider, label='transparency',
                             valmin=0.0, valmax=1.0, valinit=0.7, valfmt='%1.2f')
        self.slider.label.set_position((0.99, 1.5))
        self.slider.on_changed(self.set_alpha_value)

        for txt_lbl in self.radio_bt_quit.labels + self.radio_bt_rating.labels:
            txt_lbl.set(color=cfg.text_color_review, fontweight='normal')

        for circ in self.radio_bt_quit.circles + self.radio_bt_rating.circles:
            circ.set(radius=0.06)

        self.radio_bt_rating.on_clicked(self.save_rating)
        self.radio_bt_quit.on_clicked(self.advance_or_quit)

    # TODO implement key press handling e.g. ratings as letters (G, B etc) or numbers (1-5)

    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            if event.inaxes not in [self.slider.ax, self.radio_bt_rating.ax, self.radio_bt_quit.ax] \
                    and event.button not in [3]: # allowing toggling of overlay in zoomed-in state with right click
                self.prev_axis.set_position(self.prev_ax_pos)
                self.prev_axis.set_zorder(0)
                self.prev_axis.patch.set_alpha(0.5)
                self.zoomed_in = False

        # right click to toggle overlay
        # TODO another useful appl could be to use right click to record erroneous slices
        if event.button in [3]:
            if self.latest_alpha_seg != 0.0:
                self.prev_alpha_seg = self.latest_alpha_seg
                self.latest_alpha_seg = 0.0
            else:
                self.latest_alpha_seg = self.prev_alpha_seg
            self.update()

        # double click to zoom in to any axis
        elif event.dblclick and event.inaxes is not None:
            # zoom axes full-screen
            self.prev_ax_pos = event.inaxes.get_position()
            event.inaxes.set_position(zoomed_position)
            event.inaxes.set_zorder(1) # bring forth
            event.inaxes.set_facecolor('black') # black
            event.inaxes.patch.set_alpha(1.0)  # opaque
            self.zoomed_in = True
            self.prev_axis = event.inaxes

        else:
            pass

        plt.draw()

        return

    def set_alpha_value(self, latest_value):
        """" Use the slider to set alpha."""

        self.latest_alpha_seg = latest_value
        self.update()

    def save_rating(self, label):
        """Update the rating"""

        # print('  rating {}'.format(label))
        self.user_rating = label
        return

    def advance_or_quit(self, label):
        """Signal to quit"""

        if label.upper() == u'QUIT':
            self.quit_now = True
            plt.close(self.fig)
        else:
            self.quit_now = False
            plt.close(self.fig)

        return

    def update(self):

        # updating seg alpha for all axes
        for ax in self.axes_seg:
            ax.set_alpha(self.latest_alpha_seg)

        # self.fig.canvas.draw_idle()
        # plt.draw()
        return


def review_and_rate(mri,
                    seg,
                    alpha_mri=default_alpha_mri,
                    alpha_seg=default_alpha_seg,
                    contour_color=cfg.default_contour_face_color,
                    rating_list=default_rating_list,
                    views=default_views,
                    num_slices=default_num_slices,
                    num_rows=default_num_rows,
                    vis_type=default_vis_type,
                    fs_dir=None,
                    subject_id=None,
                    out_dir=None,
                    annot=None,
                    padding=default_padding,
                    output_path=None,
                    figsize=None,
                    **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    fig, axes_mri, axes_seg, figsize = overlay_images(mri, seg, alpha_mri=alpha_mri, alpha_seg=alpha_seg,
                                                      vis_type=vis_type, out_dir=out_dir,
                                                      contour_color=contour_color,
                                                      fs_dir=fs_dir, subject_id=subject_id,
                                                      views=views, num_slices_per_view=num_slices,
                                                      figsize=figsize, num_rows_per_view=num_rows, padding=padding,
                                                      annot=annot, output_path=output_path)

    interact_ui = ReviewInterface(fig, axes_seg, axes_mri, alpha_seg, rating_list)

    con_id_click = fig.canvas.mpl_connect('button_press_event', interact_ui.on_mouse)
    # con_id_scroll = fig.canvas.mpl_connect('scroll_event', on_mouse)

    fig.set_size_inches(figsize)
    plt.show()

    fig.canvas.mpl_disconnect(con_id_click)
    # fig.canvas.mpl_disconnect(con_id_scroll)
    plt.close()

    return interact_ui.user_rating, interact_ui.quit_now
