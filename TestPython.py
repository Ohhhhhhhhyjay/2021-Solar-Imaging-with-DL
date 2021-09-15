import numpy as np
import tensorflow as tf
import sunpy.map
import os
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop

import datetime
import time

import matplotlib.pyplot as plt
import IPython.display as display

import flares as fl
import data_prep as prep

# Training
PATH = r'E:\Program Files\VSCode\2021_ImagingWithDeepLearning\my_test_data_6'
# Validation
# PATH = r'E:\Program Files\VSCode\2021_ImagingWithDeepLearning\my_val_data_2'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu

INPUT_CHANNELS = 9
OUTPUT_CHANNELS = 1
INPUT_SHAPE = [512, 512, 9]
OUTPUT_SHAPE = [512, 512, 1]
Scaler_aia = Rescaling(scale=1.0 / 64000)
Scaler_hmi = Rescaling(scale=1.0 / 76000)
Cropper = CenterCrop(height=512, width=512)

RAND_BUFFER_SIZE = 1000
BATCH_SIZE = 1
INDEX_BATCH_SIZE = 20 # As much as my memory can hold.

LAMBDA = 100



def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image
  
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, 512, 512, 9])

  return cropped_image[0], cropped_image[1]
  
@tf.function()
def random_jitter(input_image, real_image):
  # Resizing
  input_image, real_image = resize(input_image, real_image, 600, 600)

  # Random cropping back
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

# Make a multi-channels dataset
dataset_fits_list = []
for i in range(INPUT_CHANNELS + 1): # Walk all directories
    path = os.path.join(PATH, str(i))
    fits_list = fl.get_fits_list(path)
    dataset_fits_list.append(fits_list)
dataset_fits_list = np.array(dataset_fits_list)
dataset_fits_list = np.array(list(zip(*dataset_fits_list))) # 将其横向组合

index_dataset = tf.data.Dataset.from_tensor_slices(dataset_fits_list)

index_train_dataset = index_dataset.take(160)
index_test_dataset = index_dataset.skip(160)

index_train_dataset = index_train_dataset.shuffle(RAND_BUFFER_SIZE)
index_train_dataset = index_train_dataset.batch(INDEX_BATCH_SIZE)

index_test_dataset = index_test_dataset.batch(100) # Just to have access with 'for'...

def get_dataset(dataset_fits_list): # new
    """Generate dataset from a fits list. The list is like:
    [
    [output_dir//0//xxxx.fits, output_dir//1//xxxx.fit, ...]
    [output_dir//0//xxxx.fits, output_dir//1//xxxx.fit, ...]
    [output_dir//0//xxxx.fits, output_dir//1//xxxx.fit, ...]
    ...
    [output_dir//0//xxxx.fits, output_dir//1//xxxx.fit, ...]
    ]
    """
    examples = []
    labels = []
    for set in dataset_fits_list:
        all_channels = []
        for fits in set:
            fits = fits.numpy().decode() # fits 
            telescope = prep.TelescopeFits(fits)
            if 'AIA' in telescope:
                data = prep.AIAPrep(fits, Scaler_aia, Cropper)
            if 'HMI' in telescope:
                data = prep.HMIPrep(fits, Scaler_aia, Cropper)    
            all_channels.append(data)

        input_channel = tf.concat(all_channels[:-1], axis=2)
        examples.append(input_channel)

        zeros = tf.zeros([512, 512, 8])
        output_channel = tf.concat([all_channels[-1], zeros], axis=2)
        labels.append(output_channel)
    
    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))

    return dataset

def extract_dim(input_image, real_image):
    unstacked = tf.unstack(real_image, axis=2)
    real_image = unstacked[0]
    real_image = tf.expand_dims(real_image, 2)
    return input_image, real_image

def prep_train_dataset(dataset):
    # Apply random jitter
    train_dataset = dataset.map(random_jitter)

    # Eliminate redundant channels in output images
    train_dataset = train_dataset.map(extract_dim)

    # Separate into batches
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
  
def Generator():
  inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # 64 convolution kernels with size 4*4
    downsample(128, 4),  # 128 convolution kernels with size 4*4
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  # Max absolute error -maybe suitable for bright region
  linf_loss = tf.abs(tf.reduce_max(target) - tf.reduce_max(gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
  
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input_image')
  tar = tf.keras.layers.Input(shape=OUTPUT_SHAPE, name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 12))

  input_images = tf.unstack(test_input[0], axis=2)
  input_images.append(tar[0])
  input_images.append(prediction[0])
  display_list = input_images # unstack input. Not sure if this could work.
  title = ['Input Image(s)', 'Ground Truth', 'Predicted Image']

  for i in range(INPUT_CHANNELS + 2):
    plt.subplot(3, 4, i+1)# 3*4, 11 images in total
    if i < INPUT_CHANNELS:
      plt.title(title[0])
    else:
      plt.title(title[i - INPUT_CHANNELS + 1])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
    tf.summary.scalar('disc_loss', disc_loss, step=step) # 原step//1000
    
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      start = time.time()
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start} sec\n')

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

# Get test dataset
for batch_fits_list in index_test_dataset:
    test_dataset = get_dataset(batch_fits_list)
    test_dataset = test_dataset.map(extract_dim)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    break

# Get train dataset and train
for batch_fits_list in index_train_dataset:
    train_dataset = get_dataset(batch_fits_list)
    train_dataset = prep_train_dataset(train_dataset)
    fit(train_dataset, test_dataset, steps=10)