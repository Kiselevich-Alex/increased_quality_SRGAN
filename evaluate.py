
import os
import numpy as np
import tensorlayer as tl
from PIL import Image
from numpy import asarray
from model import get_G


checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)


def evaluate(lr_img_path):
    # load image
    image = Image.open(lr_img_path)

    # convert image to numpy array
    lr_image = asarray(image)

    # define model
    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
    G.eval()

    lr_image = (lr_image / 127.5) - 1  # rescale to ［－1, 1]
    lr_image = np.asarray(lr_image, dtype=np.float32)
    lr_image = lr_image[np.newaxis, :, :, :]
    size = [lr_image.shape[1], lr_image.shape[2]]

    # get upscale image
    out = G(lr_image).numpy()

    # save SRGAN upscale image
    tl.vis.save_image(out[0], './gen.png')

    image = Image.open('./gen.png')
    upscale_image = asarray(image)
    os.remove('./gen.png')

    return upscale_image
