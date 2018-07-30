import numpy as np


def standard_network(input_features, output_features, neurons_per_layer, nr_layers, act_fun, output_activation, 
                     reg_strength, clipvalue=0.5, alpha=0.01, loss='mse'):
    
    from keras import regularizers
    from keras.optimizers import Adam
    from keras.models import Sequential, Model, load_model
    from keras.layers import Input, Dense, LeakyReLU, concatenate
    
    main_input = Input(shape=(len(input_features),), name = 'main_input')

    x = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(main_input)
    if act_fun == 'relu':
        x = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(main_input)
     #   x = relu(alpha=alpha)(x)
        x = LeakyReLU(alpha=alpha)(x)
    else:
        x = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength), 
                  activation=act_fun)(main_input)
        
    for i in range(0, nr_layers-1): # -1 because one is added automatically
        if act_fun == 'relu':
            x = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(x)
      #      x = relu(alpha=alpha)(x)
            x = LeakyReLU(alpha=alpha)(x)

        else:
            x = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength), 
                      activation=act_fun)(x)

    output_layers = []
    
    for feature in output_features:
        output_layers.append(Dense(1, kernel_regularizer=regularizers.l2(reg_strength), name = feature,
                                  activation = output_activation[feature])(x))
        
    if clipvalue > 0:
        optimiser = Adam(clipvalue = clipvalue)
    else:
        optimiser = Adam()


    model = Model(main_input, output_layers)
    model.compile(optimizer = optimiser, loss = loss)
    
    return model


def standard_network_get_nr_variables_weight_shapes(input_features, output_features, neurons_per_layer, nr_layers):
    
    weight_shapes = []
    
    weight_shapes.append((len(input_features), neurons_per_layer))
    weight_shapes.append((neurons_per_layer, ))
    
    for i_lay in range(nr_layers - 1):
        weight_shapes.append((neurons_per_layer, neurons_per_layer))
        weight_shapes.append((neurons_per_layer, ))
        
    for i_output in range(len(output_features)):
        weight_shapes.append((neurons_per_layer, 1))
        weight_shapes.append((1, ))
        
    nr_parameters = (nr_layers-1)*neurons_per_layer**2 + (len(input_features)+len(output_features)+nr_layers)*neurons_per_layer \
                     + len(output_features)
    
    return [nr_parameters, weight_shapes]


def split_network(input_features, output_features, neurons_per_layer, nr_layers, act_fun, output_activation, reg_strength,
                  alpha=0.1, optimiser='adam', loss='mse'):
    
    from keras import regularizers
    from keras.optimizers import Adam
    from keras.models import Sequential, Model, load_model
    from keras.layers import Input, Dense, LeakyReLU, concatenate
    
    if output_features.sort() is not ['SFR', 'Stellar_mass']:
        print('This network is intended to have two outputs: SFR and Stellar mass')
        return
    
    main_input = Input(shape=(len(input_features),), name = 'main_input')

    if act_fun == 'leaky_relu':
        x1 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(main_input)
        x2 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(main_input)
        x1 = LeakyReLU(alpha)(x1)
        x2 = LeakyReLU(alpha)(x2)
    else:
        x1 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength), 
                  activation=act_fun)(main_input)
        x2 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength), 
                  activation=act_fun)(main_input)

    for i in range(0, nr_layers-1): # -1 because one is added automatically
        if act_fun == 'leaky_relu':
            x1 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(x1)
            x2 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength))(x2)
            x1 = LeakyReLU(alpha)(x1)
            x2 = LeakyReLU(alpha)(x2)

        else:
            x1 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength), 
                      activation=act_fun)(x1)
            x2 = Dense(neurons_per_layer, kernel_regularizer=regularizers.l2(reg_strength), 
                      activation=act_fun)(x2)

    output_layers = []
    output_layers.append(Dense(1, kernel_regularizer=regularizers.l2(reg_strength), 
                               name = 'Stellar_mass',
                               activation = output_activation['Stellar_mass'])(x1))
    output_layers.append(Dense(1, kernel_regularizer=regularizers.l2(reg_strength), 
                               name = 'SFR',
                               activation = output_activation['SFR'])(x2))

    model = Model(main_input, output_layers)
    model.compile(optimizer = optimiser, loss = loss)
    
    return model





















