from sklearn.neighbors import NearestNeighbors
import concurrent.futures
import pandas as pd
import numpy as np

def knn(min_class_points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(min_class_points)
    _, indices = nbrs.kneighbors(min_class_points)
    neighbourhood = []
    for i in (indices):
        neighbourhood.append(min_class_points[i])
    return(np.asarray(neighbourhood)) 

def neighbourhood_oversampling(args):
    # Extracting arguments
    neighbourhood,k,num_shadow_points,list_sigma_f,num_generated_points,num_aff_comb,seed = args
    # Setting seed
    np.random.seed(seed)
    # Calculating shadow points
    neighbourhood_shadow_sample = []
    for i in range(k):
        q = neighbourhood[i]
        for _ in range(num_shadow_points):
            shadow_points = q + np.random.normal(0,list_sigma_f)
            neighbourhood_shadow_sample.append(shadow_points)
    # Selecting randomly num_aff_comb shadow points
    idx = np.random.randint(num_shadow_points*k, size=(num_generated_points,num_aff_comb))
    # Create random weights for selected points
    affine_weights = []
    num_pts = 1
    for _ in range(num_pts):
        # Create random weights for selected points
        weights = np.random.randint(100, size=idx.shape)
        sums = np.repeat(np.reshape(np.sum(weights,axis=1),(num_generated_points,1)), num_aff_comb, axis=1)
        # Normalise the weights
        affine_weights.append(np.divide(weights,sums))
    selected_points = np.array(neighbourhood_shadow_sample)[idx,:]
    # Performing dot product beteen points and weights
    neighbourhood_loras_set = []
    for affine_weight in affine_weights:
        generated_LoRAS_sample_points = list(np.dot(affine_weight, selected_points).diagonal().T)
        neighbourhood_loras_set.extend(generated_LoRAS_sample_points)
    
    return neighbourhood_loras_set

def loras_oversampling(min_class_points, k, num_shadow_points, list_sigma_f, num_generated_points, num_aff_comb, seed):
    # Calculating neighbourhoods of each minority class parent data point p in min_class_points
    neighbourhoods = knn(min_class_points, k)
    # Preparing arguments
    args = []
    for neighbourhood in neighbourhoods:
        arg = (neighbourhood, k, num_shadow_points, list_sigma_f, num_generated_points, num_aff_comb, seed)
        args.append(arg)
    # Generating points
    loras_set = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for neighbourhood_loras_set in executor.map(neighbourhood_oversampling, args):
            # Adding generated LoRAS points from specific neighbourhood
            loras_set.extend(neighbourhood_loras_set)
    
    return np.asarray(loras_set)

def loras(maj_class_points, min_class_points, k=None, num_shadow_points=None, list_sigma_f=None, num_generated_points=None, num_aff_comb=None, seed=42):
    
    # Verifying constraints
    if len(min_class_points)==0:
        print("[PARAMETER ERROR] Empty minority class")
        raise SystemExit
    if len(maj_class_points)==0:
        print("[PARAMETER ERROR] Empty majority class")
        raise SystemExit
    if len(min_class_points) >= len(maj_class_points):
        print("[PARAMETER ERROR] Number of points in minority class is equal to or exceeds number of points in the majority class")
        raise SystemExit
    
    # Completing missing parameters w/ default values
    if k is None:
        k = 8 if len(min_class_points)<100 else 30
    if num_aff_comb is None:
        num_aff_comb = min_class_points.shape[1]
    if num_shadow_points is None:
        import math
        num_shadow_points = math.ceil(2*num_aff_comb / k)
    if list_sigma_f is None:
        list_sigma_f = [.005]*min_class_points.shape[1]
    if not isinstance(list_sigma_f, list):
        list_sigma_f = [list_sigma_f]*min_class_points.shape[1]
    if num_generated_points is None:
        import math
        num_generated_points = math.ceil((len(maj_class_points) + len(min_class_points)) / len(min_class_points))
        
    # Verifying constraints
    if k <= 1:
        print("[PARAMETER ERROR] Value of parameter k is too small")
        raise SystemExit
    if k > len(min_class_points):
        print("[PARAMETER ERROR] Value of parameter k is too large for minority class points")
        raise SystemExit
    if num_shadow_points < 1:
        print("[PARAMETER ERROR] Number of shadow points is too small")
        raise SystemExit
    if not all(elem >= 0.0 and elem <= 1.0 for elem in list_sigma_f):
        print("[PARAMETER ERROR] All elements in list of sigmas have to be in [0.0,1.0]")
    if num_aff_comb < 1:
        print("[PARAMETER ERROR] Number of affine combinations is too small")
        raise SystemExit
    if num_aff_comb > k * num_shadow_points:
        print("[PARAMETER ERROR] Number of affine combinations must be smaller or equal to k * number of shadow points")
        raise SystemExit
    
    return loras_oversampling(min_class_points, k, num_shadow_points, list_sigma_f, num_generated_points, num_aff_comb, seed)