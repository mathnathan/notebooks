import numpy as np
from numpy.random import multivariate_normal as multi_norm
from scipy.spatial import cKDTree as ckdt
from collections import defaultdict
from scipy.stats import norm
import warnings


class GMM: # gaussian mixture model (Only 1D)


    def __init__(self, pis, params):
        self.params = params # [[mu1, sig1], [mu2, sig2],...]
        self.components = params.shape[0]
        self.pis = pis


    def __call__(self, x):
        pis = self.pis
        p = self.params
        sz = self.components
        return np.array([pis[i]*norm.pdf(x,*(p[i])) for i in range(sz)]).sum(axis=0)


    def sample(self, n_samples, normalize=False):
        mode_id = np.random.choice(self.components, size=n_samples, replace=True, p=self.pis)
        return [np.array([norm.rvs(*(self.params[i])) for i in mode_id]), mode_id]


class Neuron():


    def __init__(self, shape, weights, bias, decay=1, lr_decay=-0.01):

        self.weights = np.array(weights)
        self.rows, self.cols = shape
        self.dim = self.rows*self.cols
        assert self.weights.size == self.dim, "Dimension of weights does not match input shape"
        self.bias = bias
        self.bias_inv = np.linalg.inv(bias)
        self.decay = decay
        self.tot_exp = 0
        self.avg_change = 0
        self.calls = 0
        self.neighbors = []
        self.lr = 1.0 # Learning Rate
        self.lr_decay = lr_decay
        self.k = -7


    def __call__(self, x, feedback=1, update=True):
        x = np.array(x)
        if len(x.shape) <= 1:
            x.shape = (1,x.size)
        assert len(x.shape) == 2, "Input shape must be (BATCH SIZE, WEIGHTS.SIZE)"
        assert x.shape[1] == self.weights.size, "x.shape[1]={}, MUST MATCH, self.weights.size={}".format(x.shape[1], self.weights.size)
        z = x-self.weights
        exp1 = np.dot(self.bias_inv,z.T)
        exp2 = (z.T*exp1).sum(0)
        output = np.squeeze(np.exp(-0.5*exp2))

        # Update
        if update: # Can only update batches of size 1 currently
            z = np.squeeze(z)
            assert len(z.shape) <= 1, "Can only update batches of size 1"
            lq = self.lr*output
            # update weights
            self.weights = np.maximum(np.minimum(self.weights + lq*z, 1), 0)
            # update bias
            z_dot_zT = np.outer(z,z)
            c = np.power(2*np.pi,self.dim/2)*np.sqrt(np.linalg.det(self.bias))
            #bias_update = self.lr*(np.maximum(output*(z_dot_zT-self.bias),-0.1*self.bias) + self.decay*self.bias)
            bias_update = self.lr*(np.maximum(output*(z_dot_zT-self.bias),-0.1*self.bias) + (c/self.k)*self.bias)
            #print("--bias_update--\n", bias_update)
            self.bias = np.maximum(np.minimum(self.bias + bias_update, 0.15), 1e-7)
            # calc bias inverse
            b = 1-output-c/self.k
            d = b/output
# I need to control the size of the update to the inverse bias calculation
            bias_inv_update = self.lr*np.minimum(2*(1/b-1)*self.bias_inv - 1/b*(np.outer(exp1, exp1)/(d+exp2)), 0.01*self.bias_inv)
            self.bias_inv = np.maximum(np.minimum(self.bias_inv + bias_inv_update, 1e5), 100)
            # decay the learning rate
            self.lr = np.maximum(0,self.lr-self.lr_decay)
            self.calls += 1

        return output


    def add_neighbors(self, neurons):
        self.neighbors.append(neurons)


    def get_weights(self):
        return self.weights


    def get_bias(self):
        return self.bias


    def sample(self, num_samps):
        return multi_norm(self.weights[0], np.diag([self.bias]*self.dim),num_samps)


class Net():


    def __init__(self, rows, cols, num_neurons, bias, decay, kernels, locs, sleep_cycle):
        """ rows - number of rows in the input
            cols - number of columns in the input
            num_neurons - number of neurons in the layers
            bias - the bias every neuron in the layer should be initialized with
            decay - the decay rate every neuron should be initialized with (could be list)
            kernels - the kernel sizes for every neuron. If only one, it is
            duplicated
            locs - location on the input for the neuron to listen
        """

        self.input_rows = rows
        self.input_cols = cols
        self.num_neurons = num_neurons
        self.bias = bias
        self.decay = decay if hasattr(decay, '__iter__') else [decay]*num_neurons
        self.sleep_cycle = sleep_cycle
        if len(kernels) != num_neurons:
            self.kernels = kernels*num_neurons
        else:
            self.kernels = kernels
        if len(locs) != num_neurons:
            self.locs = locs*num_neurons
        else:
            self.locs = locs

        self.num_calls = 0
        self.total_activity = 0
        self.neurons = defaultdict(list)
        #if isinstance(learning_params, dict):
            #self.learning_params = [learning_params]*num_neurons
        #elif isinstance(learning_params, list):
            #self.learning_params = learning_params
        #else:
            #sys.exit("Error: Learning params must be a dict or list")
        self.__build_network()


    def __build_network(self):

        pis = np.random.rand(self.num_neurons)
        pis /= pis.sum()
        for n in range(self.num_neurons):
            r,c = self.kernels[n]
            locx,locy = self.locs[n]
            # Create neuron
            weights = np.random.rand(r,c)
            self.neurons[(locx,locy)].append(Neuron([r,c], weights, self.bias,
                self.decay[n], pis[n]))

        # Calculate the nearest neighbors for the neurons
        locs = np.array(list(self.neurons.keys()))
        kdtree = ckdt(locs)
        neighbors = kdtree.query_ball_point(locs,7)

        # Give each neuron a pointer to its neighbors
        for loc, nbhrs in zip(locs, neighbors):
            neurons = self.neurons[tuple(loc)]
            for neuron in neurons:
                for nbhr_loc in locs[nbhrs[1:]]:
                    neuron.add_neighbors(self.neurons[tuple(nbhr_loc)])


    def __call__(self, xp, feedback=1, update=1):

        #print('xp = ', xp)
        output = []
        for loc, neurons in self.neurons.items():
            for neuron in neurons:
                x,y = loc
                r = neuron.rows//2
                c = neuron.cols//2
                y0 = int(np.ceil(y-r))
                y1 = int(np.floor(y+r+1))
                x0 = int(np.ceil(x-c))
                x1 = int(np.floor(x+c+1))
                try:
                    val = neuron(xp[:,y0:y1,x0:x1], feedback, update)
                    if update:
                        # Mult by normalizing factor now because only care about
                        # exp term
                        self.total_activity += val*np.sqrt(2*np.pi*neuron.bias)
                except ValueError:
                    print('loc = ', loc)
                    raise(ValueError)
                output.append(neuron.pi*val)

        if update:
            self.num_calls += 1
            if (self.num_calls+1) % self.sleep_cycle == 0:
                self.__sleep()
                self.num_calls = 0

        return np.array(output)


    def __sleep(self):
        print("SLEEPING!")
        for loc, neurons in self.neurons.items():
            print('neurons = ', neurons)
            for neuron in neurons:
                neuron.pi = neuron.tot_exp/self.total_activity
                print('pi = ', neuron.pi)
                neuron.tot_exp = 0
                neuron.calls = 0
                neuron.k = 1
                neuron.avg_output = 0

        self.total_activity = 0

