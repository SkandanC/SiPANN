from flax import linen as nn
import jax
from jax import random
import jax.numpy as np
import tensorflow as tf
import argparse, os, math, h5py

from SiPANN.nn import cartesian_product

linear = lambda x: x

#Get all the input arguments and prep them
def loadInputArgs():
    parser = argparse.ArgumentParser(description="NN Testing")
    parser.add_argument("--data",type=str,default='fake') # Where the data file is.
    parser.add_argument("--n_batch",type=int,default=32) # Batch Size
    parser.add_argument("--n_layers",type=int,default=4) # Number of layers in the network.
    parser.add_argument("--n_nodes",type=int,default=256) # Number of neurons per layer. Fully connected layers.
    parser.add_argument("--keep_rate",type=float,default=1) # The dropout rate between 0 and 1
    parser.add_argument("--l_rate",type=float,default=.001) # Learning Rate.
    parser.add_argument("--a_func",type=str,default='tf.nn.leaky_relu') #The activation func to use
    parser.add_argument("--output_folder",type=str,default='results/') #Where to output the results to. Note: Must have / at the end.
    parser.add_argument("--job_id",type=int,default='0') #what number of a batch it is
    parser.add_argument("--norm",type=str,default='y') #whether to use the norm or not
    parser.add_argument("--rand",type=str,default='n') #whether to do a random search or not

    args = parser.parse_args()
    dict = vars(args)

    # print(dict['norm'])
    for attribute, value in dict.items():
        # print('{} : \t {}'.format(attribute, value))
        pass

    # Clean input
    for key,value in dict.items():
        if dict[key] in ('no', 'false', 'f', 'n', '0'):
            dict[key] = False
        elif dict[key] in ('yes', 'true', 't', 'y', '1'):
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass

    if dict['rand']:
        dict['n_batch'] = np.random.randint(8,257)
        dict['n_layers'] = np.random.randint(2,13)
        dict['n_nodes'] = np.random.randint(64,1025)
        dict['l_rate'] = np.random.uniform(.0001, .005) 

    # Rename dictionary items if necesarry
    kwargs = {
            'data': dict['data'],
            'n_batch':dict['n_batch'],
            'n_layers':dict['n_layers'],
            'keep_rate':dict['keep_rate'],
            'n_nodes':dict['n_nodes'],
            'l_rate':dict['l_rate'],
            'output_folder':dict['output_folder'],
            'a_func':eval(dict['a_func']),
            'job_id':dict['job_id'],
            'norm':dict['norm']
            }

    output_folder = kwargs['output_folder'] + "job_" + str(kwargs['job_id']) + "/"
    kwargs['output_folder'] = output_folder
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return kwargs

def make_weights_and_basis(act_func, input_val, input_num, output_num, keep_prob, len_features):
    sigma = 0.03
    key = random.PRNGKey(1701)
    b = sigma * random.normal(key, shape=[output_num])

    # b = tf.Variable(tf.random.normal([output_num], stddev=0.03), dtype=tf.float32)
    W = sigma * random.normal(key, [input_num, output_num])
    # W = tf.Variable(tf.random_normal([input_num, output_num], stddev=0.03), dtype=tf.float32)

    if act_func == 1:
        layer = np.matmul(input_val, W) + b
    else:
        layer = np.matmul(act_func(input_val),W) + b
    layer = nn.Dense(len_features)(layer)

    def l2_loss(x, alpha):
        return alpha * (x ** 2).mean()

    def loss_fn(W):
        loss = 0
        loss += sum(
            l2_loss(w, alpha=0.001) 
            for w in W
        )
        return loss
    regularization = loss_fn(W)

    # class Dropout2(nn.Module):
    #     """Transformer Model for sequence tagging."""
    #     dropout = nn.Dropout
    #     rate: float = 1-keep_prob
    #     @nn.compact
    #     def __call__(self, dummy):
    #         """Applies Transformer model on the inputs.
    #         Args:
    #         inputs: input data
    #         train: if it is training.
    #         Returns:
    #         output of a transformer encoder.
    #         """
    #         x = layer
    #         # self.dropout = nn.Dropout(1-keep_prob)
    #         x = self.dropout(self.rate)(x, deterministic=False)
    #         # logits = nn.Dense(
    #         #     cfg.output_vocab_size,
    #         #     kernel_init=cfg.kernel_init,
    #         #     bias_init=cfg.bias_init)(x)
    #         return x

    # init_rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    # variables = Dropout2().init(init_rngs, layer)
    # layer = Dropout2().apply(variables, layer, rngs={'dropout': random.PRNGKey(2)})
    # variables = layer.init(init_rngs, layer, deterministic=False)
    # layer = layer.apply(variables, layer, rngs={'dropout': random.PRNGKey(2)})

    return b, W, layer, regularization

def make_r(y, pred):
    total_error = 0
    unexplained_error = 0
    for i in range(y.shape[1]):
        total_error += np.sum(np.square(np.subtract(y[:,i:i+1], np.mean(y[:,i:i+1]))))
        unexplained_error += np.sum(np.square(np.subtract(y[:,i:i+1], pred[:,i:i+1])))
    R_squared = np.subtract(1.0, np.divide(unexplained_error, total_error))
    return R_squared

def get_data(data):
    if data == 'fake':
        num_points = 5000
        x = np.linspace(0, 2.5*np.pi, num_points).reshape(-1,1)
        y1 = 10000*np.sin(x)
        y2 = .000001*np.sin(x)
        data = np.hstack((x,y1,y2))
        s_data = (1,2)
    else:
        if data[-2:] == 'h5':
            with h5py.File(data, 'r') as f:
                #make sure it's labeled as expected
                if 'INPUT' in f and 'OUTPUT' in f:
                    input = np.array(f['INPUT'])
                    output = np.array(f['OUTPUT'])
                elif 'INPUT_LRG' in f and 'OUTPUT_AGL' in f and 'OUTPUT_MAG' in f:
                    input = f['INPUT_LRG']
                    output = np.hstack((f['OUTPUT_AGL'], f['OUTPUT_MAG']))
                else:
                    raise TypeError("File doesn't have INPUT/OUTPUT groups")

                #make sure they're column arrays
                if len(input.shape) == 1:
                    input = np.expand_dims(input, axis=1)
                if len(output.shape) == 1:
                    output = np.expand_dims(output, axis=1)

                s_data = (input.shape[1], output.shape[1])
                data = np.hstack((input,output))
        elif data[-3:] == "npz":
            f = np.load(data)
            #make sure it's labeled as expected
            if 'INPUT' in f and 'OUTPUT' in f:
                input = np.array(f['INPUT'])
                output = np.array(f['OUTPUT'])
            elif 'INPUT_LRG' in f and 'OUTPUT_AGL' in f and 'OUTPUT_MAG' in f:
                input = f['INPUT_LRG']
                output = np.hstack((f['OUTPUT_AGL'], f['OUTPUT_MAG']))
            elif 'TE0' in f and 'TE1' in f:
                output = np.asarray(f['TE0'])
                input = ([np.linspace(1.45, 1.65, 20), np.linspace(0.4, 0.6, 10), np.linspace(0.18, 0.24, 10), np.linspace(80, 90, 10), np.linspace(0.05,0.3,10)])
                input = cartesian_product(input)[:, 0].reshape(output.shape)
                print(input.shape)
            else:
                raise TypeError("File doesn't have INPUT/OUTPUT groups")

            #make sure they're column arrays
            if len(input.shape) == 1:
                input = np.expand_dims(input, axis=1)
            if len(output.shape) == 1:
                output = np.expand_dims(output, axis=1)

            #check for complex numbers
            if output.dtype == 'complex128':
                output = np.hstack((output.real, output.imag))

            s_data = (input.shape[1], output.shape[1])
            data = np.hstack((input, output))

    return input, output, s_data