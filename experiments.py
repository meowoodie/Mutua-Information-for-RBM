import timeit
import numpy
import arrow

from lib.rbm_ssd import RBM_SSD
from lib.rbm_sgd import RBM_SGD

try:
    import PIL.Image as Image
except ImportError:
    import Image

import theano
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams

from lib.utils import tile_raster_images
from lib.logistic_sgd import load_data

def Train_RBM(train_set, rbm_type="sgd",
              learning_rate=0.1,
              training_epochs=500,
              batch_size=20,
              output_folder='rbm_plots',
              n_hidden=500):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param train_set: Training data set

    :param rbm_type: The type of RBM object

    :param learning_rate: learning rate used for training the RBM (not used in ssd)

    :param training_epochs: number of epochs used for training

    :param batch_size: size of a batch used to train the RBM

    """

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = None
    if rbm_type == "ssd":
        rbm = RBM_SSD(input=x, n_visible=28 * 28, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)
    elif rbm_type == "sgd":
        rbm = RBM_SGD(input=x, n_visible=28 * 28, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    start_time = timeit.default_timer()

    training_data = []
    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        cur_time = str(arrow.utcnow())
        training_data.append((cur_time, epoch, numpy.mean(mean_cost)))
        print '%s Training epoch %d, cost is %f' % (cur_time, epoch, numpy.mean(mean_cost))

    end_time = timeit.default_timer()

    pretraining_time = end_time - start_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    return rbm

def Sampling_from_RBM(rbm, test_set,
                      n_chains=20,
                      n_samples=10,
                      output_folder="sampling_plots"):

    rng = numpy.random.RandomState(123)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')

    os.chdir('../')

if __name__ == "__main__":

    datasets = load_data("mnist.pkl.gz")

    train_set, _ = datasets[0]
    # test_set, _ = datasets[2]

    # rbm_ssd_hid_25  = Train_RBM(train_set, n_hidden=25, rbm_type="ssd", output_folder="ssd_hidden25_20160704")
    # rbm_ssd_hid_100 = Train_RBM(train_set, n_hidden=100, rbm_type="ssd", output_folder="ssd_hidden100_20160704")
    # rbm_ssd_hid_500 = Train_RBM(train_set, n_hidden=500, rbm_type="ssd", output_folder="ssd_hidden500_20160704")

    rbm_sgd_hid_25  = Train_RBM(train_set, n_hidden=25, rbm_type="ssd", output_folder="ssd_hidden25_20160706")
    # rbm_sgd_hid_100 = Train_RBM(train_set, n_hidden=100, rbm_type="sgd", output_folder="sgd_hidden100_20160704")
    # rbm_sgd_hid_500 = Train_RBM(train_set, n_hidden=500, rbm_type="sgd", output_folder="sgd_hidden500_20160704")

    # Sampling_from_RBM(rbm, test_set, output_folder="sampling_plots.2016_7_3")
