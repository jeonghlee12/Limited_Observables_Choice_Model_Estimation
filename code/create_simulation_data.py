# ad-hoc way to import functions from files in parent directory
from functions import create_data

import numpy as np
import pickle

file_path = "../output/"

# set number of trials
T1 = 1000
T2 = 10000

# set parameters
N = 10000
J = 5

# random number generator seed
# set for reproducibility
seed = 111111
rng = np.random.default_rng(seed)

# set coefficients
a = 1
g = 0.5
b = 0.25

t1_base = [create_data(N, J, a, g, b, rng) for _ in range(T1)]
with open(file_path + 'data/t1_base.pkl', 'wb') as f:
    pickle.dump(t1_base, f)

t1_high_v_var = [create_data(N, J, a, g, b, rng, v_var = 4) for _ in range(T1)]
with open(file_path + 'data/t1_high_v_var.pkl', 'wb') as f:
    pickle.dump(t1_high_v_var, f)

t1_high_w_range = [create_data(N, J, a, g, b, rng, w_range = 8) for _ in range(T1)]
with open(file_path + 'data/t1_high_w_range.pkl', 'wb') as f:
    pickle.dump(t1_high_w_range, f)

t1_high_v_and_w = [create_data(N, J, a, g, b, rng, v_var = 4, w_range = 8) for _ in range(T1)]
with open(file_path + 'data/t1_high_v_and_w.pkl', 'wb') as f:
    pickle.dump(t1_high_v_and_w, f)

# If runtime reaches time limit of your system, split the code here for different runs.
t2_base = [create_data(N, J, a, g, b, rng) for _ in range(T2)]
with open(file_path + 'data/t2_base.pkl', 'wb') as f:
    pickle.dump(t2_base, f)

t2_high_v_var = [create_data(N, J, a, g, b, rng, v_var = 4) for _ in range(T2)]
with open(file_path + 'data/t2_high_v_var.pkl', 'wb') as f:
    pickle.dump(t2_high_v_var, f)

# If runtime reaches time limit of your system, split the code here for different runs.
t2_high_w_range = [create_data(N, J, a, g, b, rng, w_range = 8) for _ in range(T2)]
with open('file_path + "data/t2_high_w_range.pkl', 'wb') as f:
    pickle.dump(t2_high_w_range, f)

t2_high_v_and_w = [create_data(N, J, a, g, b, rng, v_var = 4, w_range = 8) for _ in range(T2)]
with open(file_path + 'data/t2_high_v_and_w.pkl', 'wb') as f:
    pickle.dump(t2_high_v_and_w, f)