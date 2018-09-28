from data_processing import *
from keras.callbacks import EarlyStopping, TerminateOnNaN
from model_setup import *

def train_net(parameters):

    (
        galaxies, network_args, data_keys, redshifts, tot_nr_points, train_frac, val_frac, test_frac, outputs_to_weigh, 
        weigh_by_redshift, norm, nr_epochs, batch_size, progress_file, verb, early_stop_min_delta, early_stop_patience, param_id
    ) = parameters
    
#     print(nr_epochs, early_stop_patience)

    # prepare the training data
    training_data_dict = divide_train_data(galaxies, data_keys, network_args, redshifts, outputs_to_weigh=outputs_to_weigh, 
                                           weigh_by_redshift=weigh_by_redshift, total_set_size=tot_nr_points, train_frac=train_frac, 
                                           val_frac=val_frac, test_frac=test_frac, emerge_targets=True)
    # galaxies = None
    training_data_dict = normalise_data(training_data_dict, norm)
    
    model = standard_network(network_args['input_features'], network_args['output_features'], network_args['nr_neurons_per_lay'], 
                         network_args['nr_hidden_layers'], network_args['activation_function'], 
                         network_args['output_activation'], network_args['reg_strength'], clipvalue=.001)
    

    earlystop = EarlyStopping(monitor='val_loss', min_delta=early_stop_min_delta, 
                              patience=early_stop_patience, verbose=1, mode='auto')
    callbacks_list = [earlystop]

    history = model.fit(x = training_data_dict['input_train_dict'], 
                        y = training_data_dict['output_train_dict'], 
                        validation_data = (training_data_dict['input_val_dict'], 
                        training_data_dict['output_val_dict'], training_data_dict['val_weights']), 
                        epochs=int(nr_epochs), batch_size=int(batch_size), 
                        callbacks=callbacks_list, sample_weight=training_data_dict['train_weights'], verbose=verb)

    train_history = None
    val_history = None
    test_score = None
    
    
    
    if 'loss' in history.history:
        train_history = history.history['loss']
    if 'val_loss' in history.history:
        val_history = history.history['val_loss']
        norm_scores = model.evaluate(x=training_data_dict['input_test_dict'], y=training_data_dict['output_test_dict'],
                                               sample_weight=training_data_dict['test_weights'], verbose=verb)
        test_score = norm_scores[0]
    
        

    parameters = {'nr_lay': network_args['nr_hidden_layers'], 
                  'neur_per_lay': network_args['nr_neurons_per_lay'], 
                  'act_fun': network_args['activation_function'], 'batch_size': batch_size, 
                  'reg_strength': network_args['reg_strength']}
    results = [parameters, test_score, train_history, val_history, param_id]
    
    # lock write lock
    lock.acquire()
    with open(progress_file, 'r') as f:
        data = f.readlines()
    lastline = data[-1]
    progress, tot_jobs = lastline.split('/')
    progress = int(progress)
    progress += 1
    progress = '{:d}'.format(progress)
    with open(progress_file, 'a') as f:
        f.write('\n{}/{}'.format(progress, tot_jobs))
    #unlock write lock
    lock.release()
    
    return results

def init(l):
    global lock
    lock = l