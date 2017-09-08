#
# Author: Christopher Aicher <aicherc@uw.edu>
# License: BSD 3 clause
#

import numpy as np
from deep_rf.single_player_game import SinglePlayerGame
from matplotlib import cm
from PIL import Image


class GifMaker(object):
    """ Construct and Save Gif of SinglePlayerGame

    Args:
        game (SinglePlayerGame): game to track
        pixel_size (int): upscaling size of each pixel
        duration (int): duration of each frame in GIF

    Attributes:
        frames (list of np.ndarray): frames to save in GIF

    Methods:
        reset_gif() - resets gif
        add_frame() - adds current frame to gif
        save_gif() - save gif to file

    """
    def __init__(self, game, pixel_size=50, duration=0.25):
        self.game = game
        self.pixel_size = pixel_size
        self.max_pixel_value = 2.0
        self.duration = duration
        self.frames = []
        self._durations = []

    def reset_gif(self):
        # Reset Frames
        self.frames = []
        self._durations = []
        return

    def add_frame(self):
        # Get Frame of Game
        frame = self.game.get_frame()

        # Convert Frame to RGB array
        frame_array = cm.jet(np.kron(frame/self.max_pixel_value,
            np.ones((self.pixel_size, self.pixel_size))))
        frame_array = np.uint8(frame_array[:,:,0:3]*255)

        # If game is over, set frame black
        if self.game.is_game_over():
            black_frame_array = frame_array * 0
            for _ in xrange(0, 3):
                self.frames.append(frame_array)
                self._durations.append(0.5)
                self.frames.append(black_frame_array)
                self._durations.append(0.5)

        else:
            self.frames.append(frame_array)
            self._durations.append(1)

        return

    def save_gif(self, filename):
        """ Save Gif to filename """
        if filename[-3:] != "gif":
            raise ValueError("filename '{0}' must end in .gif".format(filename))
        if len(self.frames) == 0:
            raise ValueError("No Frames to Save!")

        # Convert frames to Image objects
        ims = [Image.fromarray(frame) for frame in self.frames]
        ims.append(Image.fromarray(self.frames[0]*0))

        # Calculte durations
        durations = [d * self.duration * 1000 for d in self._durations]
        durations.append(1000) # End on black for 1 second

        # Save as a gif
        ims[0].save(filename, save_all = True, append_images=ims[1:],
                duration = durations,
                loop = 0, # Loop forever
                )
        return







