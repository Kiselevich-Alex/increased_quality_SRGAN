import os
import time
import scipy
import random
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorlayer as tl
from config import config
from model import get_G, get_D
from tensorflow_core.python import device

# if learning on CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size 
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128

# create folders to save result images and trained models
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)

train_lr_img_list = sorted(
        tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
def get_train_data():
    # load dataset
    train_hr_img_list = sorted(
        tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))

    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)

    # dataset API and augmentation
    def generator_train():
        for imgid in range(len(train_lr_img_list)):
            lr_img = tf.constant(train_lr_imgs[imgid], dtype=tf.float32)
            lr_img.set_shape(train_lr_imgs[imgid].shape)

            hr_img = tf.constant(train_hr_imgs[imgid], dtype=tf.float32)
            hr_img.set_shape(train_hr_imgs[imgid].shape)

            rnd_x = random.randint(0, hr_img.shape[1] - 384 - 1)
            rnd_y = random.randint(0, hr_img.shape[0] - 384 - 1)

            yield lr_img, hr_img, rnd_x, rnd_y, int(rnd_x / 4), int(rnd_y / 4), random.randint(0, 1)

    def _map_fn_train(lr_img, hr_img, hr_rnd_x, hr_rnd_y, lr_rnd_x, lr_rnd_y, flip):
        hr_patch = tf.image.crop_to_bounding_box(hr_img, hr_rnd_y, hr_rnd_x, 384, 384)
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.

        lr_patch = tf.image.crop_to_bounding_box(lr_img, lr_rnd_y, lr_rnd_x, 96, 96)
        lr_patch = lr_patch / (255. / 2.)
        lr_patch = lr_patch - 1.

        if flip == 1:
            hr_patch = tf.image.flip_left_right(hr_patch)
            lr_patch = tf.image.flip_left_right(lr_patch)

        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(
        generator_train, output_types=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    return train_ds


def train():
    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data()

    # ## initialize learning (G)
    n_step_epoch = len(train_lr_img_list) / batch_size
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4],
                               os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))

    ## adversarial learning (G, D)
    n_step_epoch = len(train_lr_img_list) / batch_size
    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs + 1) / 2.)  # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs + 1) / 2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                    epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss,
                    d_loss))

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
            D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))


def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
    G.eval()

    for imid in range(len(valid_lr_img_list)):
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        valid_lr_img = (valid_lr_img / 127.5) - 1

        valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
        valid_lr_img = valid_lr_img[np.newaxis, :, :, :]
        size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

        out = G(valid_lr_img).numpy()

        print("LR size: %s /  generated HR size: %s" % (size, out.shape))
        print("[*] save images")
        tl.vis.save_image(out[0], os.path.join(save_dir, str(imid + 1) + '_valid_gen.png'))
        tl.vis.save_image(valid_hr_img, os.path.join(save_dir, str(imid + 1) + '_valid_hr.png'))

        out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, os.path.join(save_dir, str(imid + 1) + '_valid_bicubic.png'))
