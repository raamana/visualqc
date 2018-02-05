
__all__ = ['review_and_rate']

from matplotlib import pyplot as plt, colors, cm
from matplotlib.widgets import RadioButtons, Slider
from mrivis.color_maps import get_freesurfer_cmap
from mrivis.utils import check_params, crop_to_seg_extents
from visualqc.utils import get_axis, pick_slices, check_layout

zoomed_position = [0.2, 0.2, 0.7, 0.7]

def overlay_images(mri, seg, alpha_mri=0.8, alpha_seg=0.7,
                   vis_type='cortical_volumetric',
                   views=(0, 1, 2), num_slices_per_view=12,
                   num_rows_per_view=2, figsize=None,
                   annot=None, padding=5,
                   output_path=None):
    """Backend engine for overlaying a given seg on MRI with freesurfer label."""

    num_rows_per_view, num_slices_per_view, padding = check_params(num_rows_per_view, num_slices_per_view, padding)

    # mri = read_image(mri_spec, bkground_thresh=bkground_thresh)
    # seg = read_image(aseg_spec, bkground_thresh=0)
    mri, seg = crop_to_seg_extents(mri, seg, padding)

    num_views = len(views)
    num_rows = num_rows_per_view*num_views
    slices = pick_slices(mri.shape, views, num_slices_per_view)
    num_cols = check_layout(len(slices), num_views, num_rows_per_view)

    plt.style.use('dark_background')

    if figsize is None:
        # figsize = [min(15,4*num_rows), min(12,4*num_cols)] # max (15,12)
        figsize = [ 4 * num_rows, 4 * num_cols]
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

    # displaying some annotation text if provided
    if annot is not None:
        title_handle = fig.suptitle(annot, backgroundcolor='black', color='white', fontsize='large')
        title_handle.set_position((0.95, 0.01))

    display_params_mri = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_mri)
    display_params_seg = dict(interpolation='none', aspect='equal', origin='lower',
                              alpha=alpha_seg)

    normalize_labels = colors.Normalize(vmin=seg.min(), vmax=seg.max(), clip=True)
    fs_cmap = get_freesurfer_cmap(vis_type)
    seg_mapper = cm.ScalarMappable(norm=normalize_labels, cmap=fs_cmap)

    normalize_mri = colors.Normalize(vmin=mri.min(), vmax=mri.max(), clip=True)
    mri_mapper = cm.ScalarMappable(norm=normalize_mri, cmap='gray')

    axes_seg = list()
    axes_mri = list()

    ax = ax.flatten()
    for ax_counter, (dim_index, slice_num) in enumerate(slices):
        plt.sca(ax[ax_counter])

        slice_mri = get_axis(mri, dim_index, slice_num)
        slice_seg = get_axis(seg, dim_index, slice_num)

        seg_rgb = seg_mapper.to_rgba(slice_seg)
        mri_rgb = mri_mapper.to_rgba(slice_mri)

        handle_seg = plt.imshow(seg_rgb, **display_params_seg)
        handle_mri = plt.imshow(mri_rgb, **display_params_mri)
        plt.axis('off')

        axes_seg.append(handle_seg)
        axes_mri.append(handle_mri)

    fig.set_size_inches(figsize)

    if output_path is not None:
        # no space left unused
        plt.subplots_adjust(left=0.01, right=0.99,
                            bottom=0.01, top=0.99,
                            wspace=0.05, hspace=0.02)

        output_path = output_path.replace(' ', '_')
        fig.savefig(output_path + '.png', bbox_inches='tight')

    # leaving some space on the right for review elements
    plt.subplots_adjust(left  =0.01, right  =0.9,
                        bottom=0.01,   top  =0.99,
                        wspace=0.05 , hspace=0.02)

    return fig, axes_mri, axes_seg, figsize


class ReviewInterface(object):
    """Class to layout interaction elements and define callbacks. """

    def __init__(self, fig, axes_seg, axes_mri, alpha_seg,
                 rating_list,
                 quit_elements=("Next", "Quit")):
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
        ax_radio = plt.axes([0.905, 0.8 , 0.085, 0.18], facecolor='#009b8c')
        self.radio_bt = RadioButtons(ax_radio, self.rating_list,
                                active=None, activecolor='orange')

        ax_quit  = plt.axes([0.905, 0.59, 0.065, 0.1], facecolor='#0084b4')
        self.quit_button = RadioButtons(ax_quit, quit_elements,
                                   active=None, activecolor='orange')

        ax_slider = plt.axes([0.905, 0.73, 0.07, 0.02], facecolor='#fa8072')
        self.slider = Slider(ax_slider, label='transparency',
                             valmin=0.0, valmax=1.0, valinit=0.7, valfmt='%1.2f')
        self.slider.label.set_position((0.99, 1.5))
        self.slider.on_changed(self.set_alpha_value)

        for txt_lbl in self.quit_button.labels + self.radio_bt.labels:
            txt_lbl.set_color('#fff6da')
            txt_lbl.set_fontweight('bold')

        self.radio_bt.on_clicked(self.save_rating)
        self.quit_button.on_clicked(self.advance_or_quit)

    def on_mouse(self, event):
        """Callback for mouse events."""

        if self.prev_axis is not None:
            self.prev_axis.set_position(self.prev_ax_pos)
            self.prev_axis.set_zorder(-1)
            self.zoomed_in = False

        # right click to toggle overlay
        if event.button in [3]:
            if self.latest_alpha_seg != 0.0:
                self.prev_alpha_seg = self.latest_alpha_seg
                self.latest_alpha_seg = 0.0
            else:
                self.latest_alpha_seg = self.prev_alpha_seg
            self.update()

        # double click to zoom in to any axis
        elif event.dblclick:
            # zoom axes full-screen
            self.prev_ax_pos = event.inaxes.get_position()
            event.inaxes.set_position(zoomed_position)
            event.inaxes.set_zorder(1)
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
                    alpha_mri=0.8,
                    alpha_seg=0.7,
                    rating_list=('Good', 'Suspect', 'Bad', 'Failed', 'Later'),
                    views=(0,1,2),
                    num_slices=12,
                    num_rows=6,
                    vis_type='cortical_volumetric',
                    annot=None,
                    padding=5,
                    output_path=None,
                    figsize=None,
                    **kwargs):
    "Produces a collage of various slices from different orientations in the given 3D image"

    fig, axes_mri, axes_seg, figsize = overlay_images(mri, seg, alpha_mri=alpha_mri, alpha_seg=alpha_seg,
                                                      vis_type=vis_type, views=views, num_slices_per_view=num_slices,
                                             figsize=figsize, num_rows_per_view=num_rows, padding=padding,
                                             annot=annot, output_path=output_path)

    interact_ui = ReviewInterface(fig, axes_seg, axes_mri, alpha_seg, rating_list)

    con_id_click  = fig.canvas.mpl_connect('button_press_event', interact_ui.on_mouse)
    # con_id_scroll = fig.canvas.mpl_connect('scroll_event', on_mouse)

    fig.set_size_inches(figsize)
    plt.show()

    fig.canvas.mpl_disconnect(con_id_click)
    # fig.canvas.mpl_disconnect(con_id_scroll)
    plt.close()

    return interact_ui.user_rating, interact_ui.quit_now
