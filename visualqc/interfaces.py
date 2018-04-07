"""
Module defining base interfaces.

"""

from abc import ABC, abstractmethod

from matplotlib.widgets import Button, TextBox
from matplotlib import pyplot as plt
from visualqc import config as cfg


class BaseReviewInterface(ABC):
    """Class to layout interaction elements and define callbacks. """


    def __init__(self, fig, axes,
                 next_button_callback=None,
                 quit_button_callback=None):
        "Constructor."

        self.fig = fig
        self.axes = axes

        self.user_rating = None
        self.user_notes = None
        self.quit_now = False

        self.zoomed_in = False
        self.prev_axis = None
        self.prev_ax_pos = None

        self.add_annot()
        self.add_navigation(next_button_callback, quit_button_callback)
        self.add_notes_input()

        # disabling all default keyboard shortcuts
        for key in [k for k in plt.rcParams.keys() if k.startswith('keymap')]:
            plt.rcParams[key] = ''


    def add_annot(self, annot_text=None):
        """Text at top of UI """

        if annot_text is not None:
            self.annot_text = self.fig.text(cfg.position_annot_text[0],
                                            cfg.position_annot_text[1],
                                            annot_text, **cfg.annot_text_props)


    def add_navigation(self, user_next_callback=None,
                       user_quit_callback=None):
        """Navigation elements"""

        ax_bt_quit = self.fig.add_axes(cfg.position_quit_button,
                              facecolor=cfg.color_quit_axis, aspect='equal')
        ax_bt_next = self.fig.add_axes(cfg.position_next_button,
                              facecolor=cfg.color_quit_axis, aspect='equal')
        self.bt_quit = Button(ax_bt_quit, 'Quit', hovercolor='red')
        self.bt_next = Button(ax_bt_next, 'Next', hovercolor='xkcd:greenish')
        #
        self.bt_quit.label.set_color(cfg.color_navig_text)
        self.bt_next.label.set_color(cfg.color_navig_text)
        # new impl to take control of blocking behav of plt.show()
        if user_next_callback is not None and user_quit_callback is not None:
            self.bt_next.on_clicked(user_next_callback)
            self.bt_quit.on_clicked(user_quit_callback)
        else:
            # previous impl - gives no control over blocking plt.show()
            self.bt_quit.on_clicked(self.builtin_quit)
            self.bt_next.on_clicked(self.builtin_next)


    def add_notes_input(self):
        """Notes"""

        ax_text = self.fig.add_axes(cfg.position_text_input)
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

        self.user_notes = str(text_entered).replace(cfg.delimiter,
                                                    cfg.delimiter_replacement)


    @abstractmethod
    def on_mouse(self, event):
        """Callback for mouse events."""


    @abstractmethod
    def on_keyboard(self, event):
        """Callback for keyboard events."""


    @abstractmethod
    def allowed_to_advance(self):
        """
        Method to ensure work is done for current iteration,
        before allowing the user to advance to next subject.
        Returns True if allowed, or False if not.
        """


    def builtin_quit(self, input_event_to_ignore=None):
        "terminator"

        if not self.allowed_to_advance():
            print('You have not rated the current subject! '
                  'Please rate it before you can advance '
                  'to next subject, or to quit.')
        else:
            self.quit_now = True
            self.reset_figure()


    def builtin_next(self, input_event_to_ignore=None):
        "advancer"

        if not self.allowed_to_advance():
            print('You have not rated the current subject! '
                  'Please rate it before you can advance to next subject, '
                  'or to quit.')
        else:
            self.quit_now = False
            self.reset_figure()


    @abstractmethod
    def reset_figure(self):
        """ Resets the state of UI and clears the axes. """

