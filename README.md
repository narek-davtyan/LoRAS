# LoRAS
Localized Randomized Affine Shadowsampling (LoRAS) oversampling technique

## Installation
The latest version is available on PyPi and installable with the command: `pip install loras`

## Usage
There is just one method `fit_resample(maj_class_points, min_class_points, k, num_shadow_points, list_sigma_f, num_generated_points, num_aff_comb, random_state=42)`

There are two mandatory inputs:  

- `maj_class_points` : Majority class parent data points which is a non-empty list containing numpy arrays acting as points
- `min_class_points` : Minority class parent data points which is a non-empty list containing numpy arrays acting as points   

There are also optional parameters:

- `k` : Number of nearest neighbours to be considered per parent data point (default value: `8 if len(min_class_points)<100 else 30`)
- `num_shadow_points` : Number of generated shadowsamples per parent data point (default value: `ceil(2*num_aff_comb / k)`)
- `list_sigma_f` : List of standard deviations for normal distributions for adding noise to each feature (default value: `[0.005, ... , 0.005]`)
- `num_generated_points` : Number of shadow points to be chosen for a random affine combination (default value: `ceil((len(maj_class_points) + len(min_class_points)) / len(min_class_points))`)
- `num_aff_comb` : Number of generated LoRAS points for each nearest neighbours group (default value: `min_class_points.shape[1]`)   

Output:

 - `min_class_points::oversampled_set` : Concatenation of original data points and oversampled ones
