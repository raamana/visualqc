__all__ = ['review_and_rate']

import subprocess
import time
import traceback
from os import makedirs
from os.path import join as pjoin, exists as pexists
from subprocess import check_output

import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt, colors, cm
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, Slider, TextBox, Button
from mrivis.color_maps import get_freesurfer_cmap
from mrivis.utils import check_params, crop_to_seg_extents

from visualqc import config as cfg
from visualqc.config import zoomed_position, annot_vis_dir_name, default_padding, \
    default_navigation_options
from visualqc.utils import get_axis, pick_slices, check_layout


def generate_required_visualizations(qcw):
    """Method to pre-generate all the necessary visualizations, for the given workflow."""

    print('\n')
    if 'cortical' in qcw.vis_type and qcw.in_dir is not None and qcw.out_dir is not None:
        print('Pre-generating visualizations for {} ... Please wait!'.format(qcw.vis_type))
        start_time_vis_whole = time.time()
        vis_times = list()
        num_subjects = len(qcw.id_list)
        max_len = max([len(sid) for sid in qcw.id_list])+3
        for ii, subject_id in enumerate(qcw.id_list):
            print('processing {id:>{max_len}} ({ii}/{nn}) ... '.format(ii=ii, nn=num_subjects,
                                                                     id=subject_id, max_len=max_len), end='')
            start_time_vis_subject = time.time()
            make_vis_pial_surface(qcw.in_dir, subject_id, qcw.out_dir)
            print(' done.')
            vis_times.append(time.time()-start_time_vis_subject)

        # computing processing times
        end_time_vis_whole = time.time()
        total_vis_wf_time = end_time_vis_whole - start_time_vis_whole
        vis_times = np.array(vis_times).astype('float64')
        mean_vis_time = vis_times.mean()
        print('Time took per subject : {:.3f} seconds per subject, '
              'and {:3} seconds for {} subjects.'.format(mean_vis_time, total_vis_wf_time, num_subjects))

    else:
        print('Given {} vis_type does not need any visualizations to be pre-generated.'.format(qcw.vis_type))

    print('\n')

    return


def overlay_images(qcw, mri, seg,
                   subject_id=None,
                   annot=None,
                   figsize=None,
                   padding=default_padding,
                   output_path=None):
    """Backend engine for overlaying a given seg on MRI with freesurfer label."""

    num_rows_per_view, num_slices_per_view, padding = check_params(qcw.num_rows, qcw.num_slices, padding)
    mri, seg = crop_to_seg_extents(mri, seg, padding)

    surf_vis = dict()  # empty - no vis to include
    # TODO broaden this to include subcortical structures as well
    if 'cortical' in qcw.vis_type:
        if qcw.in_dir is not None and subject_id is not None and qcw.out_dir is not None:
            surf_vis = make_vis_pial_surface(qcw.in_dir, subject_id, qcw.out_dir)
    num_surf_vis = len(surf_vis)

    # TODO calculation below is redundant, if surf vis does not fail
    # i.e. if num_surf_vis is fixed, no need to recompute for every subject
    num_views = len(qcw.views)
    num_rows = num_rows_per_view * num_views
    slices = pick_slices(seg, qcw.views, num_slices_per_view)
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
                              alpha=qcw.alpha_mri)
    display_params_seg = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=qcw.alpha_seg)

    normalize_labels = colors.Normalize(vmin=seg.min(), vmax=seg.max(), clip=True)
    fs_cmap = get_freesurfer_cmap(qcw.vis_type)
    seg_mapper = cm.ScalarMappable(norm=normalize_labels, cmap=fs_cmap)

    normalize_mri = colors.Normalize(vmin=mri.min(), vmax=mri.max(), clip=True)
    mri_mapper = cm.ScalarMappable(norm=normalize_mri, cmap='gray')

    # deciding colors for the whole image
    unique_labels = np.unique(seg)
    # removing background - 0 stays 0
    unique_labels = np.delete(unique_labels, 0)
    if len(unique_labels) == 1:
        color4label = [qcw.contour_color]
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

        if 'volumetric' in qcw.vis_type:
            seg_rgb = seg_mapper.to_rgba(slice_seg)
            h_seg = plt.imshow(seg_rgb, **display_params_seg)
        elif 'contour' in qcw.vis_type:
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

    if annot is not None:
        h_annot = fig.suptitle(annot, **cfg.annot_text_props)
        h_annot.set_position(cfg.position_annot_text)

    fig.set_size_inches(figsize)

    if output_path is not None:
        # no space left unused
        plt.subplots_adjust(**cfg.no_blank_area)
        output_path = output_path.replace(' ', '_')
        layout_str = 'v{}_ns{}_{}x{}'.format(''.join([ str(v) for v in qcw.views]),num_slices_per_view,num_rows,num_cols)
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


def make_vis_pial_surface(in_dir, subject_id, out_dir, annot_file='aparc.annot'):
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
                _, _ = run_tksurfer_script(in_dir, subject_id, hemi, script_file)

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
    for view in cfg.surface_view_angles:
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


def run_tksurfer_script(in_dir, subject_id, hemi, script_file):
    """Runs a given TCL script to generate visualizations"""

    try:
        cmd_args = ['tksurfer', '-sdir', in_dir, subject_id, hemi, 'pial', '-tcl', script_file]
        txt_out = check_output(cmd_args, shell=False, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as tksurfer_exc:
        print('Error running tksurfer to generate 3d surface visualizations - skipping them.')
        exit_code = tksurfer_exc.returncode
        txt_out = tksurfer_exc.output
    else:
        exit_code = 0

    return txt_out, exit_code


class ReviewInterface(object):
    """Class to layout interaction elements and define callbacks. """

    def __init__(self, fig,
                 axes_seg, axes_mri,
                 qcw, subject_id,
                 flagged_as_outlier, outlier_alerts,
                 navig_options=default_navigation_options,
                 annot=None):
        "Constructor."

        self.fig = fig
        self.axes_seg = axes_seg
        self.axes_mri = axes_mri
        self.latest_alpha_seg = qcw.alpha_seg
        self.rating_list = qcw.rating_list
        self.flagged_as_outlier = flagged_as_outlier

        self.user_rating = None
        self.user_notes = None
        self.quit_now = False

        self.zoomed_in = False
        self.prev_axis = None
        self.prev_ax_pos = None

        # displaying some annotation text if provided
        if annot is not None:
            fig.text(cfg.position_annot_text[0], cfg.position_annot_text[1], **cfg.annot_text_props)

        # right above the rating area (blinking perhaps?)
        if self.flagged_as_outlier and outlier_alerts is not None:
            ax_alert = plt.axes(cfg.position_outlier_alert_box)
            ax_alert.axis('off')
            h_rect = Rectangle((0, 0), 1, 1, zorder=-1, facecolor='xkcd:coral', alpha=0.25)
            ax_alert.add_patch(h_rect)
            alert_msg = 'Flagged as an outlier'
            fig.text(cfg.position_outlier_alert[0], cfg.position_outlier_alert[1],
                     alert_msg, **cfg.annot_text_props)
            for idx, cause in enumerate(outlier_alerts):
                fig.text(cfg.position_outlier_alert[0], cfg.position_outlier_alert[1] - (idx+1) * 0.02,
                         cause, color=cfg.alert_colors_outlier[cause], **cfg.annot_text_props)

        ax_radio = plt.axes(cfg.position_rating_axis, facecolor=cfg.color_rating_axis, aspect='equal')
        self.radio_bt_rating = RadioButtons(ax_radio, self.rating_list,
                                            active=None, activecolor='orange')
        self.radio_bt_rating.on_clicked(self.save_rating)
        for txt_lbl in self.radio_bt_rating.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_rating.circles:
            circ.set(radius=0.06)


        # ax_quit = plt.axes(cfg.position_navig_options, facecolor=cfg.color_quit_axis, aspect='equal')
        # self.radio_bt_quit = RadioButtons(ax_quit, navig_options,
        #                                   active=None, activecolor='orange')
        # self.radio_bt_quit.on_clicked(self.advance_or_quit)
        # for txt_lbl in self.radio_bt_quit.labels:
        #     txt_lbl.set(color=cfg.text_option_color, fontweight='normal')
        #
        # for circ in self.radio_bt_quit.circles:
        #     circ.set(radius=0.06)


        # implementing two separate buttons for navigation (mulitple radio button has issues)
        ax_bt_quit = plt.axes(cfg.position_quit_button, facecolor=cfg.color_quit_axis, aspect='equal')
        ax_bt_next = plt.axes(cfg.position_next_button, facecolor=cfg.color_quit_axis, aspect='equal')
        self.bt_quit = Button(ax_bt_quit, 'Quit', hovercolor='red')
        self.bt_next = Button(ax_bt_next, 'Next', hovercolor='xkcd:greenish')
        self.bt_quit.on_clicked(self.quit)
        self.bt_next.on_clicked(self.next)
        self.bt_quit.label.set_color(cfg.color_navig_text)
        self.bt_next.label.set_color(cfg.color_navig_text)

        # # with qcw and subject id available here, we can add a button to
        # TODO open images directly in tkmedit with qcw.images_for_id[subject_id]['mri'] ... ['seg']

        # alpha slider
        ax_slider = plt.axes(cfg.position_slider_seg_alpha, facecolor=cfg.color_slider_axis)
        self.slider = Slider(ax_slider, label='transparency',
                             valmin=0.0, valmax=1.0, valinit=0.7, valfmt='%1.2f')
        self.slider.label.set_position((0.99, 1.5))
        self.slider.on_changed(self.set_alpha_value)

        # user notes
        ax_text = plt.axes(cfg.position_text_input) # , facecolor=cfg.color_textbox_input, aspect='equal'
        self.text_box = TextBox(ax_text, color=cfg.text_box_color, hovercolor=cfg.text_box_color,
                                label=cfg.textbox_title, initial=cfg.textbox_initial_text)
        self.text_box.label.update(dict(color=cfg.text_box_text_color, wrap=True,
                                   verticalalignment='top', horizontalalignment='left'))
        self.text_box.on_submit(self.save_user_notes)
        # matplotlib has issues if we connect two events to the same callback
        # self.text_box.on_text_change(self.save_user_notes_duplicate)


    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            # include all the non-data axes here (so they wont be zoomed-in)
            if event.inaxes not in [self.slider.ax, self.radio_bt_rating.ax,
                                    self.bt_next.ax, self.bt_quit.ax, self.text_box.ax] \
                    and event.button not in [3]: # allowing toggling of overlay in zoomed-in state with right click
                self.prev_axis.set_position(self.prev_ax_pos)
                self.prev_axis.set_zorder(0)
                self.prev_axis.patch.set_alpha(0.5)
                self.zoomed_in = False

        # right click to toggle overlay
        # TODO another useful appl could be to use right click to record erroneous slices
        if event.button in [3]:
            self.toggle_overlay()

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


    def do_shortcuts(self, key_in):
        """Callback to handle keyboard shortcuts to rate and advance."""

        # ignore keyboard key_in when mouse within Notes textbox
        if key_in.inaxes == self.text_box.ax:
            return

        key_pressed = key_in.key.lower()
        # print(key_pressed)
        if key_pressed in ['right', ' ', 'space']:
            self.user_rating = self.radio_bt_rating.value_selected
            self.next()
        if key_pressed in ['ctrl+q', 'q+ctrl']:
            self.user_rating = self.radio_bt_rating.value_selected
            self.quit()
        else:
            if key_pressed in cfg.default_rating_list_shortform:
                self.user_rating = cfg.map_short_rating[key_pressed]
                self.radio_bt_rating.set_active(cfg.default_rating_list.index(self.user_rating))
            elif key_pressed in ['t']:
                self.toggle_overlay()
            else:
                pass

    def toggle_overlay(self):
        """Toggles the overlay by setting alpha to 0 and back."""

        if self.latest_alpha_seg != 0.0:
            self.prev_alpha_seg = self.latest_alpha_seg
            self.latest_alpha_seg = 0.0
        else:
            self.latest_alpha_seg = self.prev_alpha_seg
        self.update()

    def set_alpha_value(self, latest_value):
        """" Use the slider to set alpha."""

        self.latest_alpha_seg = latest_value
        self.update()

    def save_rating(self, label):
        """Update the rating"""

        # print('  rating {}'.format(label))
        self.user_rating = label

    def save_user_notes(self, text_entered):
        """Saves user free-form notes from textbox."""

        self.user_notes = text_entered

    def save_user_notes_duplicate(self, text_entered):
        """Saves user free-form notes from textbox."""

        self.user_notes = text_entered

    # this callback for 2nd radio button is not getting executed properly
    # tracing revelas, some problems set_active callback
    def advance_or_quit(self, label):
        """Signal to quit"""

        if label.upper() == u'QUIT':
            self.quit()
        else:
            self.next()

    def quit(self, ignore_arg=None):
        "terminator"

        if self.user_rating in cfg.ratings_not_to_be_recorded:
            print('You have not rated the current subject! Please rate it before you can advance to next subject, or to quit.')
        else:
            self.quit_now = True
            plt.close(self.fig)

    def next(self, ignore_arg=None):
        "terminator"

        if self.user_rating in cfg.ratings_not_to_be_recorded:
            print('You have not rated the current subject! Please rate it before you can advance to next subject, or to quit.')
        else:
            self.quit_now = False
            plt.close(self.fig)

    def update(self):
        """updating seg alpha for all axes"""

        for ax in self.axes_seg:
            ax.set_alpha(self.latest_alpha_seg)

        # update figure
        plt.draw()



def review_and_rate(qcw,
                    mri,
                    seg,
                    subject_id=None,
                    flagged_as_outlier=False,
                    outlier_alerts=None,
                    output_path=None,
                    annot=None,
                    figsize=None,
                    **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    fig, axes_mri, axes_seg, figsize = overlay_images(qcw, mri, seg, subject_id=subject_id,
                                                      figsize=figsize, annot=annot, output_path=output_path)

    rating_ui = ReviewInterface(fig, axes_seg, axes_mri, qcw, subject_id, flagged_as_outlier, outlier_alerts, annot)

    con_id_click = fig.canvas.mpl_connect('button_press_event', rating_ui.on_mouse)
    con_id_keybd = fig.canvas.mpl_connect('key_press_event', rating_ui.do_shortcuts)
    # con_id_scroll = fig.canvas.mpl_connect('scroll_event', on_mouse)

    fig.set_size_inches(figsize)
    plt.show()

    fig.canvas.mpl_disconnect(con_id_click)
    fig.canvas.mpl_disconnect(con_id_keybd)
    # fig.canvas.mpl_disconnect(con_id_scroll)
    plt.close()

    return rating_ui.user_rating, rating_ui.user_notes, rating_ui.quit_now
