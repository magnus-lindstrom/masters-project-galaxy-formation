from data_processing import *
from keras.callbacks import EarlyStopping, TerminateOnNaN
from model_setup import *

def train_net(parameters):

    (galaxies, data_keys, redshifts, total_set_size, train_size, val_size, test_size, 
     input_features, output_features, outputs_to_weigh, weigh_by_redshift, neur_per_lay, nr_lay, act_fun, 
     output_activation, norm, reg_strength, nr_epochs, batch_size, 
     progress_file, verb, early_stop_min_delta, early_stop_patience, param_id) = parameters
    
    training_data_dict = divide_train_data(galaxies, data_keys, input_features, 
                                           output_features, redshifts, int(total_set_size), 
                                           int(train_size), int(val_size), int(test_size))
    training_data_dict = normalise_data(training_data_dict, norm)

    train_weights, val_weights, test_weights = get_weights(training_data_dict, output_features, outputs_to_weigh, 
                                                           weigh_by_redshift = weigh_by_redshift)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=early_stop_min_delta, 
                              patience=early_stop_patience, verbose=1, mode='auto')
    nan_termination = TerminateOnNaN()
    callbacks_list = [earlystop, nan_termination]

    model = standard_network(input_features, output_features, neur_per_lay, nr_lay, 
                                                     act_fun, output_activation, reg_strength, clipvalue=.1)
    history = model.fit(x = training_data_dict['input_train_dict'], 
                        y = training_data_dict['output_train_dict'], 
                        validation_data = (training_data_dict['input_val_dict'], 
                        training_data_dict['output_val_dict'], val_weights), 
                        epochs=int(nr_epochs), batch_size=int(batch_size), 
                        callbacks=callbacks_list, sample_weight=train_weights, verbose=verb)

    train_history = None
    val_history = None
    test_score = None
    
    if 'loss' in history.history:
        train_history = history.history['loss']
    if 'val_loss' in history.history:
        val_history = history.history['val_loss']
        norm_scores = model.evaluate(x=training_data_dict['input_test_dict'], y=training_data_dict['output_test_dict'],
                                               sample_weight=test_weights, verbose=verb)
        test_score = norm_scores[0]
    
        

    parameters = {'inp_norm': norm['input'], 'nr_lay': nr_lay, 'neur_per_lay': neur_per_lay, 
                 'act_fun': act_fun, 'batch_size': batch_size, 'reg_strength': reg_strength}
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