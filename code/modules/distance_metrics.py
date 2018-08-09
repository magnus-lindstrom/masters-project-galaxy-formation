import numpy as np

def minkowski_distance(x, p=2):
    
    np.seterr(invalid='raise', over='raise', divide='raise')
    
    if type(x) is list:
        x = np.array(x)
        
    n_points, n_dims = np.shape(x)
    
    distance_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        
        for j in range(i+1, n_points):
            
            try:
                distance_matrix[i, j] = np.power(np.sum(np.power(np.absolute(x[i,:] - x[j,:]), p)), 1/p)
            except:
                print('invalid i={:d}, j={:d}'.format(i, j))
                tmp = np.absolute(x[i,:] - x[j,:])
                print(tmp)
                tmp = np.power(tmp, 2)
                print(tmp)
                tmp = np.sum(tmp)
                print('{:.2e}'.format(tmp))
                tmp = np.power(tmp, 1/p)
                print('{:.2e}'.format(tmp))
            
    distance_matrix = np.transpose(distance_matrix) + distance_matrix
    
    if n_points == 2:
        return distance_matrix[0][1]
    else:    
        return distance_matrix