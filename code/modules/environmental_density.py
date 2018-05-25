import numpy as np
import time
import datetime
import math

def get_density_periodic(coordinates, weights, nr_neighbours, box_sides, nr_points, verbatim=False, progress_file=False,
                         get_neighbours=False):

    # Returns the mass densities in a sphere surrounding points. 
    # The spheres extend far enough to enclose the nr of neighbours specified.
    
    if get_neighbours:
        neigh_indices = []
        neigh_masses = []
 
    with open('results.txt', 'w+') as f:

        neigh_densities = np.zeros(nr_points)

        start = time.time()
        for i in range(nr_points):

            if int(i/1000) == i/1000 and i > 0:
                end = time.time()
                elapsed_time = (end-start)/60
                time_remaining = elapsed_time / i * (nr_points - i)
                if verbatim:
                    print('%s      Time elapsed: %.2fmin. Time remaining: %.2fmin.' % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                             elapsed_time, time_remaining))
                if progress_file:
                    f.write('%s      Time elapsed: %.2fmin. Time remaining: %.2fmin.\n' % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                                            elapsed_time, time_remaining))
                    f.flush()
            norm_dist = np.abs(coordinates - coordinates[i,:])
            inv_dist = box_sides - norm_dist
            shortest_dists = np.minimum(norm_dist, inv_dist)
            squared_shortest_dists = shortest_dists**2
            summed_squared_shortest_dists = np.sum(squared_shortest_dists, 1)
            euclidean_dists = np.sqrt(summed_squared_shortest_dists)

            indices_of_k_closest_neighbours = np.argpartition(euclidean_dists, nr_neighbours)[:nr_neighbours]
            masses_of_k_closest_neighbours = weights[indices_of_k_closest_neighbours]

            sphere_radius = np.amax(euclidean_dists[indices_of_k_closest_neighbours])

            if sphere_radius > box_sides[0]/2:
                print('WARNING! Sphere with radius larger than %d mega parsecs created.' % (box_sides[0]/2))
                if progress_file:
                    f.write('WARNING! Sphere with radius larger than %d mega parsecs created.\n' % (box_sides[0]/2))
                    f.flush()
            sphere_volume = (4/3) * math.pi * sphere_radius**3
            density =  np.sum(masses_of_k_closest_neighbours) / sphere_volume
            neigh_densities[i] = density
            
            if get_neighbours:
                neigh_indices.append(indices_of_k_closest_neighbours)
                neigh_masses.append(masses_of_k_closest_neighbours)
            
        end = time.time()
        elapsed_time = (end-start)/60
        time_remaining = elapsed_time / i * (nr_points - i)
        if progress_file:
            f.write('%s      Script finished. Elapsed time: %.2fmin.' % (datetime.datetime.now().strftime("%H:%M:%S"), elapsed_time))
            
    if get_neighbours:
        return neigh_densities, neigh_indices, neigh_masses
    
    else:
        return neigh_densities
