##########   A function for saving models to JSON
import json
import string
import random
import glob
from os.path import expanduser
home_dir = expanduser("~")
model_path = home_dir + '/models/'
from os import listdir
from os.path import isfile, join
from keras.models import load_model
from custom_keras_objects import get_custom_objects
import datetime
    
#def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
#    return ''.join(random.choice(chars) for _ in range(size))

def SaveModel(model, model_dict, description):
  
    saved_dict = {
        'model_dict': model_dict,
        'description': description
    }
#    random_string = id_generator()
    date_string = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    with open(model_path + date_string + '.json', 'w+') as f:
        json.dump(saved_dict, f)
    f.close()
    model.save(model_path + date_string + '.h5')
    print('File saved successfully.')

    
def SearchModel(search_dict, get_hits=False):
    
    file_matches = [] # store all of the matched filenames here
    for file in glob.glob(model_path + '*.json'):
        model_dict, description = LoadModel_Filename(file)
        file_is_match = 1
        for key in search_dict.keys():
            stored_value = model_dict.get(key)
            
            if (stored_value != search_dict[key]) and (stored_value != None):
                file_is_match = 0
                break
        if file_is_match:
            file_matches.append(file)
            
    if get_hits:
        print('There are %d models matching your search dictionary.' % (len(file_matches)))
        
    model_dict_dict = {}
    description_dict = {}
    for i_file, file in enumerate(file_matches):
        model_dict, description = LoadModel_Filename(file)
        model_dict_dict['%d' % (i_file)] = model_dict
        description_dict['%d' % (i_file)] = description
    return [model_dict_dict, description_dict]
    

        
def LoadModel_Filename(file_name, return_model=False):
    with open(file_name, 'r') as f:
        loaded_dict = json.load(f)
    f.close()
    
    model_dict = loaded_dict['model_dict']
    description = loaded_dict['description']
    
    if return_model:
        custom_objects_dict = get_custom_objects()
        model = load_model(file_name[:-4] + 'h5', 
            custom_objects={model_dict['loss_function']: custom_objects_dict[model_dict['loss_function']]})
        return model, model_dict, description
    else:
        return model_dict, description
    
def LoadModel(search_dict, get_number):
    
    file_matches = [] # store all of the matched filenames here
    for file in glob.glob(model_path + '*.json'):
        model_dict, description = LoadModel_Filename(file)
        file_is_match = 1
        for key in search_dict.keys():
            stored_value = model_dict.get(key)
            
            if (stored_value != search_dict[key]) and (stored_value != None):
                file_is_match = 0
                break
        if file_is_match:
            file_matches.append(file)
        
    file = file_matches[get_number]
    model, model_dict, description = LoadModel_Filename(file, return_model=True)
    print('Model loaded successfully')
    return [model, model_dict, description]


