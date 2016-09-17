import numpy as np
import tensorflow as tf
import time
import os
import pprint as pp
import distriubted_model
import image_input
import scipy.misc

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("data_dir", "train", "dir for train set")
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_bool("log_device_placement", True, "whether to log the device placement")
tf.app.flags.DEFINE_integer('save_summaries_secs', 30,
                            'Save summaries interval seconds.')

FLAGS = flags.FLAGS

z_dim = 100
sample_size = 64
image_num = 17000
batch_size = 64
step_per_epoch = image_num / batch_size


def train():
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.Variable(0)
            images, _, _ = image_input.distorted_inputs(data_dir=FLAGS.data_dir)
            images = tf.cast(images, dtype=tf.float32) / 127.5 - 1
            if not images is None:
                print("image is None!!")
            real_images = tf.placeholder(dtype=tf.float32, shape=[batch_size, 50, 50, 3], name='real_image')
            z = tf.placeholder(tf.float32, [None, z_dim], name='z')
            # z = tf.random_uniform([batch_size, z_dim], -1, 1, dtype=tf.float32)
            # sample_z = tf.random_uniform([sample_size, z_dim], -1, 1, dtype=tf.float32)
            sample_z = np.random.uniform(-1, 1, size=(sample_size, z_dim)).astype(np.float32)
            # sample_z = tf.random_uniform([sample_size, z_dim], -1, 1, dtype=tf.float32)
            G = distriubted_model.generator(z)

            # D, D_logits = distriubted_model.discriminator(images)
            D, D_logits = distriubted_model.discriminator(real_images)
            sampler = distriubted_model.sampler(z)
            sample_images, _, _ = image_input.distorted_inputs(data_dir="sample_data")
            D_, D_logits_ = distriubted_model.discriminator(G, reuse=True)
            tf.histogram_summary("z", z)
            tf.image_summary("G", G)
            tf.histogram_summary("d", D)
            tf.histogram_summary("d_", D_)

            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.zeros_like(D_)))
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logits_, tf.ones_like(D_)))
            d_loss = d_loss_real + d_loss_fake

            tf.scalar_summary("d_loss_real", d_loss_real)
            tf.scalar_summary("d_loss_fake", d_loss_fake)
            tf.scalar_summary("g_loss", g_loss)
            tf.scalar_summary("d_loss", d_loss)

            saver = tf.train.Saver()

            t_vars = tf.trainable_variables()

            d_vars = [var for var in t_vars if 'd_' in var.name]
            g_vars = [var for var in t_vars if 'g_' in var.name]
            d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                .minimize(d_loss, var_list=d_vars)
            g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                .minimize(g_loss, var_list=g_vars, global_step=global_step)

            init_op = tf.initialize_all_variables()
            summary_op = tf.merge_all_summaries()
            # sample_z = np.random.uniform(-1, 1, size=(sample_size, z_dim))
            # sample_z = tf.random.uniform([sample_size, z_dim], -1, 1, dtype=tf.float32)

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=FLAGS.checkpoint_dir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)
        """
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        sv.start_queue_runners(sess, queue_runners)
        tf.logging.info('Started %d queues for processing input data.',
                        len(queue_runners))
        print('Started %d queues for processing input data.' %
              len(queue_runners))
        """
        is_chief = (FLAGS.task_index == 0)

        with sv.managed_session(server.target) as sess:
            step = 0
            start_time = time.time()
            next_summary_time = start_time + FLAGS.save_summaries_secs
            while not sv.should_stop() and step < 120000:
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]) \
                    .astype(np.float32)
                real_img = sess.run(images)
                # print(images.get_shape())
                if next_summary_time > time.time():
                    errD_fake, errD_real, errG, _, _, step = sess.run(
                        [d_loss_fake, d_loss_real, g_loss, d_optim, g_optim, global_step],
                        feed_dict={real_images: real_img, z: batch_z})

                    print("Epoch: [%2d] step: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (step / step_per_epoch, step,
                             (time.time() - start_time), (errD_fake + errD_real), errG))
                else:
                    errD_fake, errD_real, errG, _, _, summary_str, step = sess.run(
                        [d_loss_fake, d_loss_real, g_loss, d_optim, g_optim, summary_op, global_step],
                        feed_dict={real_images: real_img, z: batch_z})
                    if is_chief:
                        tf.logging.info('Running Summary operation on the chief.')
                        print('Running Summary operation on the chief.')

                        sv.summary_computed(sess, summary_str)
                        tf.logging.info('Finished running Summary operation.')
                        print('Finished running Summary operation.')

                    next_summary_time += FLAGS.save_summaries_secs
                if step % 100 == 1 and is_chief:
                    sample_image = sess.run(sample_images)
                    samples, d1_loss, g1_loss = sess.run(
                        [sampler, d_loss, g_loss],
                        feed_dict={z: sample_z, real_images: sample_image}
                    )

                    samples = np.array(samples).astype(np.float32)
                    # print(samples.shape)
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(step / step_per_epoch, step))
                    # print('samples images to execute....')
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d1_loss, g1_loss))

        sv.should_stop()


def save_images(images, size, image_path):
    return image_save(inverse_transform(images), size, image_path)


def image_save(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    images = images[0]
    for idx, image in enumerate(images):
        print len(image)
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def inverse_transform(images):
    return (images + 1.) / 2.


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    train()


if __name__ == '__main__':
    tf.app.run()
