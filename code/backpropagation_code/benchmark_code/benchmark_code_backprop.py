import argparse
import numpy as np
import sys, getopt
import os
from os.path import expanduser
home_dir = expanduser("~")
result_dir = 'results/'
import datetime
import codecs, json
import time
import random
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
np.random.seed(999)
random.seed(999)

def csv_list(string):
	lst_of_strings = string.split(',')
	lst_of_ints = []
	for string in lst_of_strings:
		lst_of_ints.append(int(lst_of_strings[i]))
	return lst_of_ints

### Create input options
parser = argparse.ArgumentParser(description='Do benchmarks')
parser.add_argument("--cpu", action='store_true', 
	help="Run only on the CPU.")
parser.add_argument("-t", "--train_size", default=10000, 
	help="The size of the training set used. Max is 600000.", type=int)
parser.add_argument("-e", "--nr_epochs", default=100, 
	help="The number of epochs to train for.", type=int)
parser.add_argument("-b", "--batch_size", default=10000, 
	help="The batch size to use when it is not the iterated variable", type=int)
parser.add_argument("-l", "--nr_layers", default=10, 
	help="The number of hidden layers to use when it is not the iterated variable", type=int)
parser.add_argument("-n", "--neur_per_lay", default=10, 
	help="The number of neurons per hidden layer to use when it is not the iterated variable", type=int)
parser.add_argument("-a", "--actfun", default='tanh', 
	help="The activation function to use when it is not the iterated variable")
#parser.add_argument("-B", "--batch_sizes_list", default='100,500,1000,5000,10000', 
#	help="The list of batch sizes to use when it is the iterated variable. Provide as \"val1,val2,...,valn\""
#parser.add_argument("-L", "--nr_layers_list", default='2,4,6', 
#	help="The list of nr of hidden layers to use when it is the iterated variable. Provide as \"val1,val2,...,valn\""
#parser.add_argument("-N", "--neur_per_lay_list", default='2,6,10', 
#	help="The list of neurons per hidden layer to use when it is the iterated variable. Provide as \"val1,val2,...,valn\""
#parser.add_argument("-A", "--actfun_list", default='tanh,relu,sigmoid', 
#	help="The list of activation functions to use when it is the iterated variable. Provide as \"val1,val2,...,valn\""
#parser.add_argument("-I", "--input_feats", default='Halo_mass,Halo_mass_peak,Concentration,Halo_spin', 
#	help="The list of input features to use. Provide as \"val1,val2,...,valn\""
#parser.add_argument("-O", "--output_feats", default='Stellar_mass', 
#	help="The list of output features to use. Provide as \"val1,val2,...,valn\""

args = parser.parse_args()

### General parameters
run_on_cpu = args.cpu
nEpochs = [args.nr_epochs]
batch_sizes = args.batch_sizes 
train_set_size = [args.train_size] # how many examples will be used for training+validation+testing
input_features = args.input_feats
output_features = output_feats

### Network parameters
nLayers = args.nr_layers_list # nLayers + 1 = nHiddenLayers
activationFunctions = args.actfun_list
neuronsPerLayer = args.neur_per_lay_list

data_dict = {'X_pos': 0, 'Y_pos': 1, 'Z_pos': 2, 'X_vel': 3, 'Y_vel': 4, 
             'Z_vel': 5, 'Halo_mass': 6, 'Stellar_mass': 7, 'SFR': 8, 
             'Intra_cluster_mass': 9, 'Halo_mass_peak': 10, 
             'Stellar_mass_obs': 11, 'SFR_obs': 12, 'Halo_radius': 13, 
             'Concentration': 14, 'Halo_spin': 15, 'Type': 16}


json_info_dict = {
    'On_CPU_only': run_on_cpu,
    'number_of_epochs': nEpochs,
    'training_set_size': train_set_size,
    'input_features': input_features,
    'output_features': output_features,
    'Parameter_order': ['batch_sizes', 'nr_of_layers', 'activation_functions', 'neurons_per_layer'],
    'batch_sizes': batch_sizes,
    'nr_of_layers': nLayers,
    'activation_functions': activationFunctions,
    'neurons_per_layer': neuronsPerLayer
}

if run_on_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

galfile = pd.read_hdf('/scratch/data/galcats/P200/galaxies.Z01.h5')
galaxies = galfile.as_matrix()
gal_header = galfile.keys().tolist()

### Remove data points with halo mass below 10.5
galaxies = galaxies[galaxies[:,6] > 10.5, :]


n_data_points = galaxies.shape[0]
train_indices = np.random.choice(n_data_points, int(train_set_size[0]), replace=False)

x_train = np.zeros((len(train_indices), len(input_features)))

y_train = np.zeros((len(train_indices), len(output_features)))


for i in range(len(input_features)):
    x_train[:,i] = galaxies[train_indices, data_dict[input_features[i]]]
    
for i in range(len(output_features)):
    y_train[:,i] = galaxies[train_indices, data_dict[output_features[i]]]


if run_on_cpu:
    pu_string = 'CPU'
else:
    pu_string = 'GPU'

timing_grid = np.zeros((len(batch_sizes), len(nLayers), len(activationFunctions), len(neuronsPerLayer)))
tot_nr_comb = np.size(timing_grid)
comb_tried = 0
glob_start = time.clock()
date_time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
with open(result_dir + pu_string + date_time_string + '.txt', 'w+') as f:
    
    date_string_proper = datetime.datetime.now().strftime("%H:%M, %Y-%m-%d")
    f.write('Benchmark done on ' + pu_string + ' at ' + date_string_proper + '\n')
    f.write('Parameters checked are batch sizes, nLayers, activationFunctions and neuronsPerLayer\n\n')
    f.flush()
    
    for i_bSize, bSize in enumerate(batch_sizes):
        for i_nLay, nLay in enumerate(nLayers):
            for i_actFun, actFun in enumerate(activationFunctions):
                for i_neurPerLay, neurPerLay in enumerate(neuronsPerLayer):

                    comb_tried += 1

                    # create model
                    model = Sequential()
                    model.add(Dense(neurPerLay, input_dim = len(input_features), activation = actFun))

                    for i in range(0, nLay):
                        model.add(Dense(neurPerLay, activation = actFun))

                    model.add(Dense(len(output_features), activation = None))

                    # Compile model
                    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

                    start = time.clock()
                    # Fit the model
                    history = model.fit(x_train , y_train, epochs=int(nEpochs[0]), 
                            batch_size=int(bSize), verbose=0)
                    end = time.clock()

                    timing_grid[i_bSize, i_nLay, i_actFun, i_neurPerLay] = (end - start)/60

                    progress_end = time.clock()
                    elapsed_so_far = (progress_end - glob_start) / 60
                    time_remaining = elapsed_so_far / comb_tried * (tot_nr_comb - comb_tried)

                    f.write('Combinations tried: %d/%d     ' % (comb_tried, tot_nr_comb))
                    f.write('Elapsed time: %dmin     ' % (elapsed_so_far))
                    f.write('Time remaining: %dmin.\n' % (time_remaining))
                    f.flush()

                    
f.close()
                
json_data = timing_grid.tolist()
json_data = [json_data]
json_data.append(json_info_dict)

with open(result_dir + pu_string + date_string + '.json', 'w+') as f:
    json.dump(json_data, f)
f.close()

