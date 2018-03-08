"""
Module defining various interfaces, base and derived.

"""

from abc import ABC, abstractmethod
from visualqc import config as cfg
from matplotlib.widgets import RadioButtons, CheckButtons, Button, TextBox
from matplotlib import pyplot as plt

class BaseReviewInterface(ABC):
    """Class to layout interaction elements and define callbacks. """

    def __init__(self,
                 fig,
                 qcw,
                 axes_base,
                 axes_overlay,
                 ):
        "Constructor."

        self.fig = fig
        self.rating_list = qcw.rating_list

        self.axes_base = axes_base
        self.axes_overlay = axes_overlay

        self.latest_alpha_overlay = qcw.alpha_overlay

        self.user_rating = None
        self.user_notes = None
        self.quit_now = False

        self.zoomed_in = False
        self.prev_axis = None
        self.prev_ax_pos = None

    def add_annot(self, text=None, pos=None):
        """Text at top of UI """

        if text is not None and pos is not None:
            self.fig.text(cfg.position_annot_text[0],
                          cfg.position_annot_text[1],
                          **cfg.annot_text_props)

    def add_navigation(self):
        """Navigation elements"""

        ax_bt_quit = plt.axes(cfg.position_quit_button,
                              facecolor=cfg.color_quit_axis, aspect='equal')
        ax_bt_next = plt.axes(cfg.position_next_button,
                              facecolor=cfg.color_quit_axis, aspect='equal')
        self.bt_quit = Button(ax_bt_quit, 'Quit', hovercolor='red')
        self.bt_next = Button(ax_bt_next, 'Next', hovercolor='xkcd:greenish')
        self.bt_quit.on_clicked(self.quit)
        self.bt_next.on_clicked(self.next)
        self.bt_quit.label.set_color(cfg.color_navig_text)
        self.bt_next.label.set_color(cfg.color_navig_text)


    def add_notes_input(self):
        """Notes"""

        ax_text = plt.axes(cfg.position_text_input)
        self.text_box = TextBox(ax_text, color=cfg.text_box_color,
                                hovercolor=cfg.text_box_color,
                                label=cfg.textbox_title,
                                initial=cfg.textbox_initial_text)
        self.text_box.label.update(dict(color=cfg.text_box_text_color,
                                        wrap=True,
                                        verticalalignment='top',
                                        horizontalalignment='left'))
        self.text_box.on_submit(self.save_user_notes)


    def save_user_notes(self, text_entered):
        """Saves user free-form notes from textbox."""

        self.user_notes = text_entered

    def quit(self, ignore_arg=None):
        "terminator"

        if self.user_rating in cfg.ratings_not_to_be_recorded:
            print('You have not rated the current subject! '
                  'Please rate it before you can advance '
                  'to next subject, or to quit.')
        else:
            self.quit_now = True
            self.reset_figure()

    def next(self, ignore_arg=None):
        "advancer"

        if self.user_rating in cfg.ratings_not_to_be_recorded:
            print('You have not rated the current subject! '
                  'Please rate it before you can advance to next subject, '
                  'or to quit.')
        else:
            self.quit_now = False
            self.reset_figure()

    @abstractmethod
    def reset_figure(self):
        """ Resets the state of UI and clears the axes. """


class PialWhiteSurfReviewInterface(BaseReviewInterface):
    """Review interface to rate the quality of pial and white matter surfaces on T1 mri."""

    def __init__(self, fig, qcw, axes_base, axes_overlay):
        """Constructor"""

        super().__init__(fig, qcw, axes_base, axes_overlay)


    def add_rating_UI(self):
        """Rating"""

        ax_radio = plt.axes(cfg.position_rating_axis,
                            facecolor=cfg.color_rating_axis,
                            aspect='equal')
        self.radio_bt_rating = RadioButtons(ax_radio,
                                            self.rating_list,
                                            active=None,
                                            activecolor='orange')
        self.radio_bt_rating.on_clicked(self.save_rating)
        for txt_lbl in self.radio_bt_rating.labels:
            txt_lbl.set(color=cfg.text_option_color, fontweight='normal')

        for circ in self.radio_bt_rating.circles:
            circ.set(radius=0.06)


    def save_rating(self, label):
        """Update the rating"""

        # print('  rating {}'.format(label))
        self.user_rating = label
