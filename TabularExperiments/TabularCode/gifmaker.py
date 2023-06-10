import imageio
import numpy as np
import os


def make_gif(subfolder=None, folder=None, fps = 60):
    if subfolder is None:
        pass
    else:
        folder = os.path.join(folder, subfolder)

    images = []
    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith('.png'):
            file_path = os.path.join(folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(f'{folder}/animation.gif', images, fps=fps)

make_gif('BLBR_std_sweep', fps=4)