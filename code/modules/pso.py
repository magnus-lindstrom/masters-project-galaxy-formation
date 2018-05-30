import numpy as np
import time
import datetime
from keras.models import Sequential
from keras.layers import Dense

class Feed_Forward_Neural_Network():
    
    def __init__(self, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function):
        
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.input_features = input_features
        self.output_features = output_features
        self.activation_function = activation_function
        
        self.model = None
        
    def pso_setup(self, pso_param_dict={}):
        
        self.pso_swarm = PSO_Swarm(self, self.nr_hidden_layers, self.nr_neurons_per_lay, self.input_features, 
                                   self.output_features, self.activation_function, pso_param_dict=pso_param_dict)
        
    def pso_train(self, nr_iterations, training_data_dict, loss_func_dict, nr_iters_before_restart_check=None, 
                  speed_check=False):
        
        self.pso_swarm.train_network(nr_iterations, training_data_dict, loss_func_dict, nr_iters_before_restart_check, speed_check)


class PSO_Swarm(Feed_Forward_Neural_Network):
    
    def __init__(self, parent, nr_hidden_layers, nr_neurons_per_lay, input_features, output_features, 
                 activation_function, pso_param_dict=None):
        self.pso_param_dict = {
            'nr_particles': 40,
            'xMin': -10,
            'xMax': 10,
            'alpha': 1,
            'deltaT': 1,
            'c1': 2,
            'c2': 2,
            'inertiaWeightStart': 1.4,
            'inertiaWeightMin': 0.3,
            'explorationFraction': 0.8,
            'min_std_tol': 0.01
        }
    
        if pso_param_dict is not None:
            for key in pso_param_dict:
                if key in self.pso_param_dict:
                    self.pso_param_dict[key] = pso_param_dict[key]
                else:
                    print('\'%s\ is not a valid key. Choose between:' % (key), self.pso_param_dict.keys())
                    break
        
        self.parent = parent
        self.nr_variables = (nr_hidden_layers-1)*nr_neurons_per_lay**2 + \
            (len(input_features)+len(output_features)+nr_hidden_layers)*nr_neurons_per_lay + len(output_features)
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons_per_lay = nr_neurons_per_lay
        self.activation_function = activation_function
        self.input_features = input_features
        self.output_features = output_features
        
        self.best_weights = None
        
        self.inertia_weight = self.pso_param_dict['inertiaWeightStart']
        self.vMax = (self.pso_param_dict['xMax']-self.pso_param_dict['xMin']) / self.pso_param_dict['deltaT']
        
        self.set_up_model()
        
        self.initialise_swarm()
        
    def set_up_model(self):
        
        self.model = Sequential()
        self.model.add(Dense(self.nr_neurons_per_lay, input_dim = len(self.input_features), 
                             activation = self.activation_function))
    
        for i in range(0, self.nr_hidden_layers-1):
            self.model.add(Dense(self.nr_neurons_per_lay, activation = self.activation_function))

        self.model.add(Dense(len(self.output_features), activation = None))
                
        # always give the model mse loss. When evaluating the particles, the actual loss will be taken into account
        self.model.compile(loss='mse', optimizer='adam')
        
    def train_network(self, nr_iterations, training_data_dict, loss_function, nr_iters_before_restart_check, speed_check):
        
        if loss_function == 'halo_mass_weighted_loss':
        
            self.real_halo_masses = {}
            log_halo_masses = training_data_dict['x_train'][:, training_data_dict['x_data_keys']['Halo_mass']]
            self.real_halo_masses['train'] = np.power(10, log_halo_masses)
            log_halo_masses = training_data_dict['x_val'][:, training_data_dict['x_data_keys']['Halo_mass']]
            self.real_halo_masses['val'] = np.power(10, log_halo_masses)
            log_halo_masses = training_data_dict['x_test'][:, training_data_dict['x_data_keys']['Halo_mass']]
            self.real_halo_masses['test'] = np.power(10, log_halo_masses)

        self.training_data_dict = training_data_dict
        self.loss_function = loss_function
        self.nr_iterations_trained = nr_iterations
        
        with open('progress.txt', 'w+') as f:

            # make sure the output isn't just the same
            shouldStartFresh = 1
            while shouldStartFresh:
                shouldStartFresh = 0

                inertia_weight_reduction = np.exp(np.log(self.pso_param_dict['inertiaWeightMin'] / 
                                            self.pso_param_dict['inertiaWeightStart']) / 
                                            self.pso_param_dict['explorationFraction'] / nr_iterations)
                inertia_weight = self.pso_param_dict['inertiaWeightStart']

                self.validationScoreHistory = []
                self.trainingScoreHistory = []
                
                self.avg_speed_before_history = []
                self.avg_speed_after_history = []
                    
                lastTimeSwarmBest = 0
                
                self.initialise_swarm()

                glob_start = time.time()
                for iteration in range(nr_iterations):

                    if (int(iteration/10) == iteration/10) and (iteration > 0):
                        # see if network has run into a local minima   
                        if nr_iters_before_restart_check is not None:
                            if (iteration - lastTimeSwarmBest) > nr_iters_before_restart_check:
                                self.set_weights(self.best_weights)
                                y_pred = self.predict_output('val')

                                stds = np.std(y_pred, axis=0)
                                print('standard deviations of predicted parameters: ', stds)
                                shouldStartFresh = np.any(stds < self.pso_param_dict['min_std_tol'])
                                if shouldStartFresh:
                                    break

                        progress_end = time.time()
                        elapsed_so_far = (progress_end - glob_start) / 60
                        time_remaining = elapsed_so_far / iteration * (self.nr_iterations_trained - iteration)

                        print('%s, Iteration %d' % (datetime.datetime.now().strftime("%H:%M:%S"), iteration))
                        f.write('%s      ' % (datetime.datetime.now().strftime("%H:%M:%S")))
                        f.write('Iterations tried: %d/%d     ' % (iteration, self.nr_iterations_trained))
                        f.write('Elapsed time: %dmin     ' % (elapsed_so_far))
                        f.write('Time remaining: %dmin.\n' % (time_remaining))
                        f.flush()

                    for iParticle, particle in enumerate(self.particle_list):
                        
                        train_score = particle.evaluate_particle('train')

                        is_swarm_best_train = (train_score < self.swarm_best_train)
                        
                        if is_swarm_best_train:
                            
                        
                            lastTimeSwarmBest = iteration
                            self.swarm_best_train = train_score
                            self.swarm_best_position = particle.position
                            
                            val_score = particle.evaluate_particle('val')
                            is_swarm_best_val = (val_score < self.swarm_best_val)
                            if is_swarm_best_val: # only update best weights after val highscore
                                self.best_weights = particle.get_weights()
                            
                            self.validationScoreHistory.append(val_score)
                            self.trainingScoreHistory.append(train_score)

                            print('Iteration %d, particle %d, new swarm best. Train: %.3e, Val: %.3e' % (iteration, 
                                                            iParticle, train_score, val_score))
                            f.write('Iteration %d, particle %d, new swarm best. Train: %.3e, Val: %.3e\n' % (iteration, 
                                                            iParticle, train_score, val_score))
                            f.flush()


                    self.update_swarm(speed_check, f)
                    
                    inertia_weight = self.update_inertia_weight(inertia_weight, inertia_weight_reduction, 
                                                                iteration, f)
                    
                    

        end = time.time()
        print('%s, Training complete.' % (datetime.datetime.now().strftime("%H:%M:%S")))
        f.write('%s, Training complete.' % (datetime.datetime.now().strftime("%H:%M:%S")))
        f.flush()
        
    def update_inertia_weight(self, inertia_weight, inertia_weight_reduction, iteration, f):
        
        isExploring = (inertia_weight > self.pso_param_dict['inertiaWeightMin'])
        if isExploring:
            inertia_weight = inertia_weight * inertia_weight_reduction
            isExploring = (inertia_weight > self.pso_param_dict['inertiaWeightMin'])
            if not isExploring:
                print('SWITCH TO EPLOIT! Iteration %d/%d.' % (iteration, self.nr_iterations_trained))
                f.write('SWITCH TO EPLOIT! Iteration %d/%d.\n' % (iteration, self.nr_iterations_trained))
                f.flush()
        return inertia_weight
        
    def predict_output(self, mode):
    
        y_pred = self.model.predict(self.training_data_dict['x_'+mode])

        return y_pred
    
    def initialise_swarm(self):
        
        self.particle_list = []
        
        for i in range(self.pso_param_dict['nr_particles']):
            
            particle = PSO_Particle(self)
            self.particle_list.append(particle)
            
        self.swarm_best_train = 1e20
        self.swarm_best_val = 1e20
        self.swarm_best_position = self.particle_list[0].best_position  # arbitrarily take the first position
        self.best_particle_nr = 0
        
    def update_swarm(self, speed_check, f):
        
        self.speeds_before = []
        self.speeds_after = []
        #self.term_one = []
        #self.term_two = []
        #self.too_fast_count = 0
        #self.mean_particle_best_difference = []
        #self.mean_swarm_best_difference = []
        
        #q = np.random.uniform(size = self.nr_variables)
        #r = np.random.uniform(size = self.nr_variables)
        
        for particle in self.particle_list:
            particle.update_particle()
            
        #print('term 1: ', np.mean(self.term_one))
        #print('term 2:', np.mean(self.term_two)) 
        #print('%d/%d particles were too fast.' % (self.too_fast_count, self.pso_param_dict['nr_particles']))
        #print('mean particle best diff: %.2f'% (np.mean(self.mean_particle_best_difference)))
        #print('mean swarm best diff: %.2f'% (np.mean(self.mean_swarm_best_difference)))
        #print('q: ', np.mean(q))
        #print('r: ', np.mean(r))
        avg_speed_before = np.mean(self.speeds_before)
        avg_speed_after = np.mean(self.speeds_after)
        self.avg_speed_before_history.append(avg_speed_before)
        self.avg_speed_after_history.append(avg_speed_after)
        
        if speed_check:
            print('Average speed of the particles before normalization is: ', avg_speed_before)
            print('Average speed of the particles after normalization is: ', avg_speed_after)
            f.write('Average speed of the particles before normalization is: %.2f' % (avg_speed_before))
            f.write('Average speed of the particles after normalization is: %.2f' % (avg_speed_after))
            f.flush()
            
            
    def set_weights(self, weightList):
        
        weightMatrixList = weightList[0]
        biasList = weightList[1]
        for i in range(len(weightMatrixList)):
            self.model.layers[i].set_weights([weightMatrixList[i], biasList[i]])
            
    def set_best_weights(self):
        weightMatrixList = self.best_weights[0]
        biasList = self.best_weights[1]
        for i in range(len(weightMatrixList)):
            self.model.layers[i].set_weights([weightMatrixList[i], biasList[i]])
            
            
    def evaluate_model(self, mode):
    
        y_pred = self.model.predict(self.training_data_dict['x_'+mode])
    
        if self.loss_function == 'mse':
            n_points, n_outputs = np.shape(y_pred)
            diff = y_pred - self.training_data_dict['y_'+mode]
            square = np.power(diff, 2)
            feature_scores = np.sum(square, 0) / n_points
            score = np.sum(feature_scores)
            
        elif self.loss_function == 'halo_mass_weighted_loss':
            
            weights = self.real_halo_masses[mode]
            n_points, n_outputs = np.shape(y_pred)
            diff = y_pred - self.training_data_dict['y_'+mode]
            square = np.power(diff, 2)
            transp_square = np.transpose(square)
            weighted_transp_square = np.multiply(transp_square, weights) / np.sum(weights)
            weighted_square = np.transpose(weighted_transp_square)
            feature_scores = np.sum(weighted_square, 0) / n_points
            score = np.sum(feature_scores)
            
        return score
        
class PSO_Particle(PSO_Swarm):
        
    def __init__(self, parent):
        
        self.parent = parent
            
        r1 = np.random.uniform(size=(self.parent.nr_variables))
        r2 = np.random.uniform(size=(self.parent.nr_variables))

        self.position = self.parent.pso_param_dict['xMin'] + r1 * (self.parent.pso_param_dict['xMax'] - 
                                self.parent.pso_param_dict['xMin'])
        self.velocity = self.parent.pso_param_dict['alpha']/self.parent.pso_param_dict['deltaT'] * \
                        ((self.parent.pso_param_dict['xMin'] - self.parent.pso_param_dict['xMax'])/2 + r2 * 
                         (self.parent.pso_param_dict['xMax'] - self.parent.pso_param_dict['xMin']))
        
        self.best_score = 1e20
        self.best_position = self.position
        
        
        
    def evaluate_particle(self, mode):
        
        weightList = self.get_weights()
        self.parent.set_weights(weightList)
        
        score = self.parent.evaluate_model(mode)
        
        if mode == 'train' and score < self.best_score:
            self.best_score = score
            self.best_position = self.position
            
        return score
        
    def get_weights(self): # sets the weights from the current pos in parameter space
        
        weightMatrixList = [] # will contain a list of all the weight matrices 
        biasList = []   # will contain a list of all the biases

        weightCounter = 0 # to help assign weights and biases to their correct matrix

        ### Extract weight matrices
        input_dim = len(self.parent.input_features)
        output_dim = len(self.parent.output_features)
        weightMatrix = np.zeros((input_dim, self.parent.nr_neurons_per_lay)) 
        for i in range(input_dim):  
            weightMatrix[i,:] = self.position[weightCounter:weightCounter+self.parent.nr_neurons_per_lay]
            weightCounter += self.parent.nr_neurons_per_lay
        weightMatrixList.append(weightMatrix)

        
        for iLayer in range(self.parent.nr_hidden_layers-1):
            weightMatrix = np.zeros((self.parent.nr_neurons_per_lay, self.parent.nr_neurons_per_lay))
            for iNeuron in range(self.parent.nr_neurons_per_lay):

                weightMatrix[iNeuron,:] = self.position[weightCounter:weightCounter+self.parent.nr_neurons_per_lay]
                weightCounter += self.parent.nr_neurons_per_lay

            weightMatrixList.append(weightMatrix)

        weightMatrix = np.zeros((self.parent.nr_neurons_per_lay, output_dim))
        for i in range(self.parent.nr_neurons_per_lay):  
            weightMatrix[i,:] = self.position[weightCounter:weightCounter+output_dim]
            weightCounter += output_dim

        weightMatrixList.append(weightMatrix)

        ### Extract bias vectors
        for iLayer in range(self.parent.nr_hidden_layers):

            biasVector = self.position[weightCounter:weightCounter+self.parent.nr_neurons_per_lay]
            weightCounter += self.parent.nr_neurons_per_lay

            biasList.append(biasVector)

        biasVector = np.zeros(output_dim)
        biasVector = self.position[weightCounter:weightCounter+output_dim] # for the output layer
        biasList.append(biasVector)

        weightCounter += output_dim
        
        weightList = [weightMatrixList, biasList]

        #print(weightCounter == len(self.position))  # a check if the number of variables is correct
        
        return weightList

    def update_particle(self):

        q = np.random.uniform()#size = self.parent.nr_variables)
        r = np.random.uniform()#size = self.parent.nr_variables)
        #print(q)
        #print(r)
        particle_best_difference = self.best_position - self.position
        swarm_best_difference = self.parent.swarm_best_position - self.position
        
        #self.parent.mean_particle_best_difference.append(np.mean(np.abs(particle_best_difference)))
        #self.parent.mean_swarm_best_difference.append(np.mean(np.abs(swarm_best_difference)))

        self.velocity = self.parent.inertia_weight * self.velocity + self.parent.pso_param_dict['c1'] * q * \
                        particle_best_difference / self.parent.pso_param_dict['deltaT'] + \
                        self.parent.pso_param_dict['c2'] * r * swarm_best_difference / \
                        self.parent.pso_param_dict['deltaT']
                    
        #self.parent.term_one.append(np.mean(np.abs(self.parent.pso_param_dict['c1'] * q * \
        #                particle_best_difference / self.parent.pso_param_dict['deltaT'])))
        #self.parent.term_two.append(np.mean(np.abs(self.parent.pso_param_dict['c2'] * r * swarm_best_difference / \
        #                self.parent.pso_param_dict['deltaT'])))

        # now limit velocity to vMax
        absolute_velocity_before_normalization = np.sqrt(np.sum(np.power(self.velocity, 2)))
        is_too_fast = absolute_velocity_before_normalization > self.parent.vMax
        if is_too_fast:
            #self.parent.too_fast_count += 1
            self.velocity = self.velocity * self.parent.vMax / absolute_velocity_before_normalization
            
        absolute_velocity_after_normalization = np.sqrt(np.sum(np.power(self.velocity, 2)))

        self.parent.speeds_before.append(absolute_velocity_before_normalization)
        self.parent.speeds_after.append(absolute_velocity_after_normalization)
            
        self.position = self.position + self.velocity * self.parent.pso_param_dict['deltaT']
        
        
