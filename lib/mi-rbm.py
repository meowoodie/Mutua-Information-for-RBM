"""Mutual Information for RBM

@Author: Woodie
@Date: July 12, 2016
@Description:
 It's a brand new version of RBM, using another kind of optimization
 method, named Stochastic Variational Training of Mutual Information.
 We have mainly revamped the get_cost_updates function in order to
 implement our new cost function.
@Contact: meowoodie@outlook.com
"""
import os
import arrow

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy
import theano
import theano.tensor as T
theano.config.exception_verbosity='high'

from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import load_data
from utils import tile_raster_images

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        # end-snippet-1

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1,
                         persistent=None, k=1,
                         hidden_sample_l=1,
                         visible_sample_m=1,
                         k1=0.9, k2=0.1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param hidden_sample_l (L): the number of hidden samples which are sampled
               according to one visible input.

        :param visible_sample_m (M): the number of visible input which are sampled
               from the training dataset, i.e. the size of
               mini-batch.

        The gradient of cost (R) is formulated as:

        grad(R)   = (1 / M) * sum_{n=1}^{M} {grad(R_n)}
        grad(R_n) = (1 / L) * sum_{l=1}^{L} {
                    log(p(v^n|h^l;\theta)) * grad(log(p(h^l|v^n;\theta))) +
                    grad(log(p(v^n|h^l;\theta)))
                    }
        """
        ##########################################
        # * Snippet-1: Multual Information Cost *
        ##########################################
        def R(input):
            # Calculate the gradient of R_n(\theta) for one v_input
            def calculate_Rn(v_input):

                # Calculate the gradient of R_l(\theta)
                # Equation:   R_n(\theta) = (1/L) * sum_{l=1}^{L} {R_l(\theta)}
                # Annotation: L = hidden_sample_l
                def calculate_Rl(v_input):
                    # Sample a h_sample according to one v_input
                    _, hl_mean, hl_sample = self.sample_h_given_v(v_input)
                    # Calculate the probability of visible output according to h_sample
                    _, vn_mean = self.propdown(hl_sample)
                    # - Part1.
                    #   Desc: Multiply each element in grad with T.log(vn_mean).sum()
                    #   Hint: [array(...), array(...), array(...)] = T.grad(..., self.params)
                    #         The number of elements in gradient is the number of params which are partial derivation.

                    # part1 = map(lambda x: x * T.log(vn_mean).sum(),
                    #             T.grad(T.log(hl_mean).sum(),
                    #                    self.params,
                    #                    disconnected_inputs='warn'))
                    part1 = [x * T.log(vn_mean).sum() for x in T.grad(
                        T.log(hl_mean).sum(),
                        self.params,
                        disconnected_inputs='warn')]

                    # - Part2.
                    part2 = T.grad((T.log(vn_mean).sum()),
                                    self.params,
                                    consider_constant=[vn_mean],
                                    disconnected_inputs='warn')
                    # Rl is the result that add corresponding elements in two gradient.
                    # Rl = log(p(v^n|h^l;\theta)) * grad(log(p(h^l|v^n;\theta))) + grad(log(p(v^n|h^l;\theta)))
                    # Rl = map(lambda p1, p2: p1 + p2, part1, part2)
                    Rl = [x + y for x, y in zip(part1, part2)]

                    mi_cost_xi = T.log(vn_mean).sum()

                    Rl.append(mi_cost_xi)
                    return Rl

                # Calculate the gradient of R_n(\theta) for one v_input, including:
                # - For L times:
                # - 1. Sample a h_sample with respect to current v_input
                # - 2. Calculate the gradient of R_l(\theta) with respect to current h_sample
                (
                    Rls,
                    updates
                ) = theano.scan(
                    calculate_Rl,
                    outputs_info=None,
                    non_sequences=v_input,
                    n_steps=hidden_sample_l
                )
                # - 3. Sum all R_l(\theta)
                #      Hint: the result of scan likes:
                #            [array([[...],
                #                    [...],
                #                     ... ,
                #                    [...]]),
                #             array(... ...),
                #             array(... ...)]
                #             One array(...) represent the result of a partial derivative params.
                # Warning: If do we need to calculate the total sum of the elements in the matrix? (.sum())
                #          or just the sum of corresponding elements in different array(...)? (.sum(0))
                # Rn = map(lambda x: x.sum(0) / hidden_sample_l, Rls)
                mi_cost_x = Rls.pop().sum()

                Rn = [x.sum(0) / hidden_sample_l for x in Rls]

                Rn.append(mi_cost_x)
                return Rn

            # Calculate the gradient of R_n(\theta) for each v_input
            # Annotation: self.input are a mini-batch of visible inputs which are sampled randomly from training dataset.
            (
                Rns,
                updates
            ) = theano.scan(
                calculate_Rn,
                outputs_info=None,
                sequences=[input]
            )
            # Get the gradient by summing all R_n(\theta)
            # Warning: If do we need to calculate the total sum of the elements in the matrix? (.sum())
            #          or just the sum of corresponding elements in different array(...)? (.sum(0))
            # R = map(lambda x: x.sum(0) / visible_sample_m, Rns)
            mi_cost = Rns.pop().sum() / (hidden_sample_l * visible_sample_m)

            R = [x.sum(0) / visible_sample_m for x in Rns]

            return (R, mi_cost), updates

        ##########################################
        # * Snippet-2:     Free Energy Cost     *
        ##########################################
        def G(input):
            # compute positive phase
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(input)

            # decide how to initialize persistent chain:
            # for CD, we use the newly generate hidden sample
            # for PCD, we initialize from the old state of the chain
            if persistent is None:
                chain_start = ph_sample
            else:
                chain_start = persistent
            # end-snippet-2
            # perform actual negative phase
            # in order to implement CD-k/PCD-k we need to scan over the
            # function that implements one gibbs step k times.
            # Read Theano tutorial on scan for more information :
            # http://deeplearning.net/software/theano/library/scan.html
            # the scan will return the entire Gibbs chain
            (
                [
                    pre_sigmoid_nvs,
                    nv_means,
                    nv_samples,
                    pre_sigmoid_nhs,
                    nh_means,
                    nh_samples
                ],
                updates_R
            ) = theano.scan(
                self.gibbs_hvh,
                # the None are place holders, saying that
                # chain_start is the initial state corresponding to the
                # 6th output
                outputs_info=[None, None, None, None, None, chain_start],
                n_steps=k
            )
            # start-snippet-3
            # determine gradients on RBM parameters
            # note that we only need the sample at the end of the chain
            chain_end = nv_samples[-1]

            cost = T.mean(self.free_energy(input)) - T.mean(
                self.free_energy(chain_end))

            G = T.grad(cost, self.params, consider_constant=[chain_end])

            return G, updates_R

        ##########################################
        # * Snippet-3:     Final Cost     *
        ##########################################
        g_G = g_R = gparams = mi_cost = updates = updates_R = updates_G = None
        if k1 > 0:
            g_G, updates_G = G(self.input)
            updates = updates_G
            gparams = g_G
        if k2 > 0:
            (g_R, mi_cost), updates_R = R(self.input)
            updates = updates_R
            gparams = g_R
        if k1 > 0 and k2 > 0:
            updates = updates_G.update(updates_R)
            gparams = [k1 * x - k2 * y for x, y in zip(g_G, g_R)]

        # Using SGD to constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )

        # TODO: need add a new function in order to change monitoring_cost to the real cost
        # monitoring_cost = self.get_pseudo_likelihood_cost(updates)

        return mi_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost


    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


def training(train_set, learning_rate=0.1, training_epochs=50,
             mini_batch_M=10, hidden_sample_L=10, n_hidden=50,
             K1=0.9, K2=0.1):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param mini_batch_M: size of a batch used to train the RBM

    :param hidden_sample_L: the number of hidden samples which are sampled
           according to one visible input.
    """
    print "Parameters:\nlr:\t%f\nepochs:\t%d\nmini_batch:\t%d\nhidden_sample:\t%d\nn_hidden:\t%d\nk1:\t%f\nk2:\t%f\n" \
          % (learning_rate, training_epochs, mini_batch_M, hidden_sample_L, n_hidden, K1, K2)
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_value(borrow=True).shape[0] / mini_batch_M

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((mini_batch_M, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient by using MI
    cost, updates = rbm.get_cost_updates(
        lr=learning_rate,
        persistent=persistent_chain,
        k=15,
        hidden_sample_l=hidden_sample_L,
        visible_sample_m=mini_batch_M,
        k1=K1,
        k2=K2
    )

    # Training RBM
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set[index * mini_batch_M: (index + 1) * mini_batch_M]
        },
        name='train_rbm'
    )

    start_time = arrow.now()
    
    print "[%s] Start training." % start_time
    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print '[%s] Training epoch %d, cost is %f' % (arrow.now(), epoch, numpy.mean(mean_cost))

    end_time = arrow.now()

    pretraining_time = end_time - start_time

    print "Training took %s" % pretraining_time

    return rbm

def generating(rbm, test_set, n_chains=20, n_samples=10, output_folder="rbm_plots"):
    #
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rng = numpy.random.RandomState(123)

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
    # end-snippet-6 start-snippet-7
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
    # end-snippet-7
    os.chdir('../')

if __name__ == '__main__':
    datasets = load_data("mnist.pkl.gz")

    train_set, _ = datasets[0]
    test_set, _ = datasets[2]

    rbm = training(train_set, learning_rate=0.01, training_epochs=20, mini_batch_M=100, hidden_sample_L=10, n_hidden=100, K1=0, K2=1)
    generating(rbm, test_set, output_folder="test_generated")


