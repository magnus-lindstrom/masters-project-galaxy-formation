from keras import backend as K

def weighted_mse_1(y_true, y_pred):
	return K.mean(K.log(y_true+1.5) + K.square(y_pred - y_true), axis=-1)

def get_custom_objects():

	custom_objects_dict = {
		'mse': 'mse',
		'weighted_mse_1': weighted_mse_1
	}
	
	return custom_objects_dict	
