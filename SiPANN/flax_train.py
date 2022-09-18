import pickle
from tkinter import HIDDEN
from typing import Sequence
import pkg_resources
from flax import linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
import jax
import jax.numpy as numpy
import optax
import tensorflow as tf

from SiPANN import import_nn
import SiPANN.utils as ut


class History():
    def __init__(self, n_batch, n_layers, n_nodes, keep_rate, l_rate, a_func, s_data):
        self.epoch = []
        self.loss_tr = []
        self.loss_val = []
        self.r_tr = []
        self.r_val = []
        self.pred_tr = []
        self.pred_val =[]
        self.n_batch = n_batch
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.keep_rate = keep_rate
        self.l_rate = l_rate
        self.a_func = a_func
        self.normX = None
        self.normY = None
        self.s_data = s_data

    def add_norm(self, normX, normY=None):
        self.normX = normX
        self.normY = normY

    def epoch_end(self, loss_tr, loss_val, r_tr, r_val, epoch=None):
        if epoch == None:
            self.epoch.append(len(self.epoch))
        else:
            self.epoch.append(epoch)
        self.loss_tr.append(loss_tr)
        self.loss_val.append(loss_val)
        self.r_tr.append(r_tr)
        self.r_val.append(r_val)

    def import_values(self):
        dict = {'normX': self.normX,
                'normY': self.normY,
                's_data': self.s_data}
        return dict

    def values(self):
        idx = self.loss_val.index(min(self.loss_val))
        dict = {'loss': self.loss_val[idx],
                'r': self.r_val[idx],
                'n_batch': self.n_batch,
                'n_layers': self.n_layers,
                'keep_rate': self.keep_rate,
                'n_nodes': self.n_nodes,
                'l_rate': self.l_rate,
                'a_func': self.a_func,
                'b_epoch': self.epoch[idx]}
        return dict

n_batch, n_layers, n_nodes, keep_rate, l_rate, output_folder, data, a_func, norm = 8, [3, 6, 9], [128, 512, 1024], 0.6, 0.0001, '/SiPANN/FLAX_DATA/COUPLER_GAP/', "SiPANN/FLAX_DATA/COUPLER_GAP/data.npz", nn.leaky_relu, True

epochs = 500
beta = 0.0001

#import data
input_tf, output_tf, s_data = ut.get_data(data)
print("--Data Imported--")
hist = History(n_batch, n_layers, n_nodes, keep_rate, l_rate, a_func, s_data)

l_nodes = [int(
    n_nodes[0]/
    (2**(
        abs(
            int(
                n_layers[0]/2)-
                i)))) 
                for i in range(n_layers[0]+1)]

data = numpy.hstack((input_tf, output_tf))
train_size = int(len(data) * 0.2)
data_val = data[0:train_size].copy()
data_tr = data[train_size:].copy()
print(data_tr.shape, data_val.shape)

if norm:
    normX = import_nn.TensorMinMax(copy=True, feature_range=(-1,1))
    print(f'{data_tr[:, :s_data[0]].shape=}')
    normX.fit(data_tr[:,:s_data[0]])
    input_normed = normX.transform(input_tf)

    normY = import_nn.TensorMinMax(copy=True, feature_range=(-1,1))
    normY.fit(data_val[:,:s_data[1]])
    output_normed = normY.transform(output_tf)

    hist.add_norm(normX, normY)

# basis_all = weights_all = hidden_all = []
# b, W, input, regularizer = ut.make_weights_and_basis(1, input_normed, s_data[0], l_nodes[0], 0.5)
# basis_all.append(b), weights_all.append(W)
# hidden = input
# for i in range(n_layers[0]):
#     b, W, hidden, temp = ut.make_weights_and_basis(a_func, hidden, l_nodes[i], l_nodes[i+1], 0.5)
#     basis_all.append(b), weights_all.append(W), hidden_all.append(hidden)
#     regularizer += temp
# b, W, output1, temp = ut.make_weights_and_basis(1, hidden, l_nodes[-1], s_data[1], 0.5)
# basis_all.append(b), weights_all.append(W)
# regularizer += temp

def loss_fn(params):
    loss = 0
    output = params
    for i in range(s_data[1]):
        loss += numpy.mean(numpy.square(output_normed[:,i:i+1] - output[:,i:i+1]), axis=-1)
    #loss = tf.losses.mean_squared_error(output_normed, output)
    loss_total = numpy.mean(loss + regularizer*beta)
    return loss_total
# # train = tf.train.AdamOptimizer(l_rate).minimize(loss_total)
train = optax.adam(l_rate)
# r = ut.make_r(output_normed, output1)

def quadratic_task(theta, opt_fn, opt_state, steps=100):
    # @jax.jit
    # def f(theta):
    #     product = jax.vmap(numpy.matmul)(w, theta)
    #     return numpy.mean(numpy.sum((product - y) ** 2, axis=1))

    losses = []
    for _ in range(steps):
        loss, grads = jax.value_and_grad(loss_fn)(theta)
        updates, opt_state = opt_fn(grads, opt_state)
        theta += updates
        losses.append(loss)

    return numpy.stack(losses), theta, opt_state

batch_size = n_batch
keep_prob = 0.5
count = 0
min_loss = float('inf')
gen_loss_cur = 0
gen_loss_old = float('inf')

# initial_params = {
#     'hidden': hidden,
#     'output': output1,
# }
sgd = train.update
# with open("output.txt", "wt") as fd:
#     try:
#         losses, *_ = quadratic_task(output1, opt_fn=sgd, opt_state=train.init(output1))
#     except ValueError as e:
#         print(e, file=fd)
# fd.close()
# def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
#     opt_state = optimizer.init(params)

#     @jax.jit
#     def step(params, opt_state):
#         loss_value, grads = jax.value_and_grad(loss)(params)
#         updates, opt_state = optimizer.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss_value

#     for i in range(len(data_tr)):
#         params, opt_state, loss_value = step(params, opt_state)
#     print(f'step {i}, loss: {loss_value}')
#     return params

# Finally, we can fit our parametrized function using the Adam optimizer
# provided by optax.
# optimizer = train
# params = fit(initial_params, optimizer)

# plt.plot(numpy.arange(len(losses)), losses)
plt.show()

class CNN(nn.Module):
    """A simple CNN model."""

    def setup(self):
        self.basis_all = self.weights_all = self.hidden_all = numpy.array([])

    @nn.compact
    def __call__(self, input_normed):
        len_features = len(numpy.unique(data_tr))
        b, W, input, regularizer = ut.make_weights_and_basis(1, input_normed, s_data[0], l_nodes[0], 0.5, len_features)
        numpy.append(self.basis_all, b), numpy.append(self.weights_all, W)
        hidden = input
        for i in range(n_layers[0]):
            b, W, hidden, temp = ut.make_weights_and_basis(a_func, hidden, l_nodes[i], l_nodes[i+1], 0.5, len_features)
            numpy.append(self.basis_all, b), numpy.append(self.weights_all, W), numpy.append(self.hidden_all, hidden)
            regularizer += temp
        b, W, output1, temp = ut.make_weights_and_basis(1, hidden, l_nodes[-1], s_data[1], 0.5, len_features)
        numpy.append(self.basis_all, b), numpy.append(self.weights_all, W)
        regularizer += temp
        return output1
        # x = input_normed
        # x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        # x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # x = x.reshape((x.shape[0], -1))  # flatten
        # x = nn.Dense(features=256)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=10)(x)
        return x

def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = numpy.mean(numpy.argmax(logits, -1) == labels)
    return {
        'loss': loss,
        'accuracy': accuracy,
    }

def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, data_tr)
    print(params)
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = CNN().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics

@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])

def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = train_size
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        print(f'{perm=}')
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: numpy.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state

def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']

train_ds, test_ds = data_tr, data_val
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

momentum = 0.9

state = create_train_state(init_rng, l_rate, momentum)
del init_rng  # Must not be used anymore.

for epoch in range(1, epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
    # Evaluate on the test set after each training epoch 
    test_loss, test_accuracy = eval_model(state.params, test_ds)
    print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
        epoch, test_loss, test_accuracy * 100))

#output data
# with open(output_folder + 'History.pkl', 'wb') as output:
#     pickle.dump(hist, output, protocol=pickle.HIGHEST_PROTOCOL)
print("--Done--")
