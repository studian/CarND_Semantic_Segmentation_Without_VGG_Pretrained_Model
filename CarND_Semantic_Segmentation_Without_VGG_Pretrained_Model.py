import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os.path
import tensorflow as tf
import warnings
from distutils.version import LooseVersion

import time

import re
import random
import numpy as np
import scipy.misc
#from scipy import misc
import shutil
import zipfile

from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def test_for_kitti_dataset(data_dir):
    kitti_dataset_path = os.path.join(data_dir, 'data_road')
    training_labels_count = len(glob(os.path.join(kitti_dataset_path, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(kitti_dataset_path, 'training/image_2/*.png')))
    testing_images_count = len(glob(os.path.join(kitti_dataset_path, 'testing/image_2/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0),\
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(kitti_dataset_path)
    assert training_images_count == 289, 'Expected 289 training images, found {} images.'.format(training_images_count)
    assert training_labels_count == 289, 'Expected 289 training labels, found {} labels.'.format(training_labels_count)
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def layers(num_classes, image_input):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    #.. VGG16 Architecture

    # Conv1
    W1_1 = tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.01))
    W1_2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    Conv1_1 = tf.nn.conv2d(image_input, W1_1, strides=[1, 1, 1, 1], padding="SAME")
    Conv1_1 = tf.nn.relu(Conv1_1)
    Conv1_2 = tf.nn.conv2d(Conv1_1, W1_2, strides=[1, 1, 1, 1], padding="SAME")
    Conv1_2 = tf.nn.relu(Conv1_2)
    Conv1_2 = tf.nn.max_pool(Conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Conv2
    W2_1 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    W2_2 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
    Conv2_1 = tf.nn.conv2d(Conv1_2, W2_1, strides=[1, 1, 1, 1], padding="SAME")
    Conv2_1 = tf.nn.relu(Conv2_1)
    Conv2_2 = tf.nn.conv2d(Conv2_1, W2_2, strides=[1, 1, 1, 1], padding="SAME")
    Conv2_2 = tf.nn.relu(Conv2_2)
    Conv2_2 = tf.nn.max_pool(Conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Conv3
    W3_1 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
    W3_2 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    W3_3 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
    Conv3_1 = tf.nn.conv2d(Conv2_2, W3_1, strides=[1, 1, 1, 1], padding="SAME")
    Conv3_1 = tf.nn.relu(Conv3_1)
    Conv3_2 = tf.nn.conv2d(Conv3_1, W3_2, strides=[1, 1, 1, 1], padding="SAME")
    Conv3_2 = tf.nn.relu(Conv3_2)
    Conv3_3 = tf.nn.conv2d(Conv3_2, W3_3, strides=[1, 1, 1, 1], padding="SAME")
    Conv3_3 = tf.nn.relu(Conv3_3)
    Conv3_Out = tf.nn.max_pool(Conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    vgg_layer3_out = Conv3_Out

    # Conv4
    W4_1 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01))
    W4_2 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    W4_3 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    Conv4_1 = tf.nn.conv2d(Conv3_Out, W4_1, strides=[1, 1, 1, 1], padding="SAME")
    Conv4_1 = tf.nn.relu(Conv4_1)
    Conv4_2 = tf.nn.conv2d(Conv4_1, W4_2, strides=[1, 1, 1, 1], padding="SAME")
    Conv4_2 = tf.nn.relu(Conv4_2)
    Conv4_3 = tf.nn.conv2d(Conv4_2, W4_3, strides=[1, 1, 1, 1], padding="SAME")
    Conv4_3 = tf.nn.relu(Conv4_3)
    Conv4_Out = tf.nn.max_pool(Conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    vgg_layer4_out = Conv4_Out

    # Conv5
    W5_1 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    W5_2 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    W5_3 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01))
    Conv5_1 = tf.nn.conv2d(Conv4_Out, W5_1, strides=[1, 1, 1, 1], padding="SAME")
    Conv5_1 = tf.nn.relu(Conv5_1)
    Conv5_2 = tf.nn.conv2d(Conv5_1, W5_2, strides=[1, 1, 1, 1], padding="SAME")
    Conv5_2 = tf.nn.relu(Conv5_2)
    Conv5_3 = tf.nn.conv2d(Conv5_2, W5_3, strides=[1, 1, 1, 1], padding="SAME")
    Conv5_3 = tf.nn.relu(Conv5_3)
    Conv5_Out = tf.nn.max_pool(Conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Conv_out = tf.reshape(Conv5_Out, [-1, 65536])

    # Conv6
    W6_1 = tf.Variable(tf.random_normal([3, 3, 512, 4096], stddev=0.01))
    W6_2 = tf.Variable(tf.random_normal([3, 3, 4096, 6], stddev=0.01))
    Conv6_1 = tf.nn.conv2d(Conv5_Out, W6_1, strides=[1,1,1,1], padding="SAME")
    Conv6_1 = tf.nn.relu(Conv6_1)
    Conv6_2 = tf.nn.conv2d(Conv6_1, W6_2, strides=[1,1,1,1], padding="SAME")
    Conv6_2 = tf.nn.relu(Conv6_2)
    vgg_layer7_out = Conv6_2

    #.. FCN8 Architecture
    # To build the decoder portion of FCN-8, we’ll upsample the input to the original image size.
    # The shape of the tensor after the final convolutional transpose layer will be 4-dimensional:
    # (batch_size, original_height, original_width, num_classes).

    # FCN-7
    fcn_layer_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1, 1),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_layer_7')

    # DCONV-7
    fcn_dconv_7 = tf.layers.conv2d_transpose(fcn_layer_7, num_classes, kernel_size=4, strides=(2, 2), padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='fcn_dconc_7')

    # FCN-4
    fcn_layer_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1, 1),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_layer_4')

    # SKIP LAYER-4
    skip_layer_4 = tf.add(fcn_dconv_7, fcn_layer_4)

    # DCONV-4
    fcn_dconv_4 = tf.layers.conv2d_transpose(skip_layer_4, num_classes, kernel_size=4, strides=(2, 2), padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='fcn_dconv_4')

    # FCN-3
    fcn_layer_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1, 1),
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='fcn_layer_3')

    # SKIP LAYER-3
    skip_layer_3 = tf.add(fcn_dconv_4, fcn_layer_3)

    # DCONV-3
    fcn_dconv_3 = tf.layers.conv2d_transpose(skip_layer_3, num_classes, kernel_size=16, strides=(8, 8), padding='SAME',
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name='fcn_dconv_3')

    return fcn_dconv_3

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # return None, None, None

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    outFile = open("output.txt", 'a')
    print("Training ..." + "\n")
    outFile.write("Training..." + "\n")

    start_time = time.clock()

    for epoch in range(epochs):
        batches = get_batches_fn(batch_size)

        for batch_input, batch_label in batches:
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: batch_input, correct_label: batch_label, keep_prob: 0.5,
                                          learning_rate: 1e-4})

        end_time = time.clock()
        train_time = end_time - start_time
        print("Epoch: {}/{} | Execution Time: {} sec | Loss: {}".format(epoch, epochs, train_time, loss))

        outFile.write("Epoch: {}/{} | Execution Time: {} sec | Loss: {}".format(epoch, epochs, train_time, loss))

    outFile.close()
    pass

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    test_for_kitti_dataset(data_dir)

    epochs = 25
    batch_size = 16

    # Download pretrained vgg model
    #helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/


    #input_image = tf.placeholder(tf.float32, [None, ])

    with tf.Session() as sess:
        # # tensorboard --logdir=./logs/vggload
        # merged_summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./logs/vggload/f1", sess.graph)

        # Path to vgg model
        #vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image = tf.placeholder(tf.float32, name='image_input')
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        #input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(num_classes, input_image)
        logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # # tensorboard --logdir=./logs/vggload
        # merged_summary = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("./logs/vggload/f1", sess.graph)


        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob,
                 learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        # tf.layers.conv2d()

if __name__ == '__main__':
    run()

