
import os
import numpy as np
import scipy
import tensorlayer as tl
from PIL import Image
from numpy import asarray
from model import get_G

# create folders to save result images
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
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

    # save SRGAN upscale image to save_dir
    tl.vis.save_image(out[0], os.path.join(save_dir, 'gen.png'))

    # save bicubic upscale image to save_dir
    out_bicubic = scipy.misc.imresize(lr_image[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicubic, os.path.join(save_dir, 'bicubic.png'))
