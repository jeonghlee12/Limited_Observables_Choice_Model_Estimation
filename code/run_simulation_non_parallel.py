# For faster performance, try running the parallelized code in the 'run parellelized simulation' folder

# -------------------------------- Import packages --------------------------------
import pandas as pd
from collections import OrderedDict

# ignore warning output
import warnings
warnings.filterwarnings('ignore')

# read and write python objects
import pickle

from functions import run_simulations, save_csv

file_path = "../output/"

# ----------------- Create specifications for logistic regression -----------------

# this package of logit requires a specification to how we receive the coefficients
basic_specification = OrderedDict()
basic_names = OrderedDict()

# one intercept term for all 4 (J - 1 = 5 - 1 = 4) products
basic_specification["intercept"] = [[1, 2, 3, 4]]
basic_names["intercept"] = ['intercept']

# one gamma for all P
basic_specification["P"] = [[0, 1, 2, 3, 4]]
basic_names["P"] = ['P']

# one gamma for all X
basic_specification["X"] = [[0, 1, 2, 3, 4]]
basic_names["X"] = ['X']

# ---------------------------- Read in simulation data ----------------------------
print("Start!")
print("Trial 1")

# Base data
print('Base Case')
with open(file_path + 'data/t1_base.pkl', 'rb') as f:
    t1_base = pickle.load(f)

t1_base_coefs_1, t1_base_coefs_2, t1_base_coefs_3, t1_base_coefs_4 = run_simulations(t1_base, basic_specification, basic_names)

with open(file_path + 'coefs/t1_base_coefs.pkl', "wb") as f:
    pickle.dump(t1_base_coefs_1, f)
    pickle.dump(t1_base_coefs_2, f)
    pickle.dump(t1_base_coefs_3, f)
    pickle.dump(t1_base_coefs_4, f)

save_csv('t1_base_coefs', [t1_base_coefs_1, t1_base_coefs_2, t1_base_coefs_3, t1_base_coefs_4], file_path + 'hist_data/')

## High v variance
print('High v Case')
with open(file_path + 'data/t1_high_v_var.pkl', 'rb') as f:
    t1_high_v_var = pickle.load(f)

t1_high_v_var_coefs_1, t1_high_v_var_coefs_2, t1_high_v_var_coefs_3, t1_high_v_var_coefs_4 = run_simulations(t1_high_v_var, basic_specification, basic_names)

with open(file_path + 'coefs/t1_high_v_var_coefs.pkl', "wb") as f:
    pickle.dump(t1_high_v_var_coefs_1, f)
    pickle.dump(t1_high_v_var_coefs_2, f)
    pickle.dump(t1_high_v_var_coefs_3, f)
    pickle.dump(t1_high_v_var_coefs_4, f)

save_csv('t1_high_v_var_coefs', [t1_high_v_var_coefs_1, t1_high_v_var_coefs_2, t1_high_v_var_coefs_3, t1_high_v_var_coefs_4], file_path + 'hist_data/')

## High w range
print('High w Case')
with open(file_path + 'data/t1_high_w_range.pkl', 'rb') as f:
    t1_high_w_range = pickle.load(f)

t1_high_w_range_coefs_1, t1_high_w_range_coefs_2, t1_high_w_range_coefs_3, t1_high_w_range_coefs_4 = run_simulations(t1_high_w_range, basic_specification, basic_names)

with open(file_path + 'coefs/t1_high_w_range_coefs.pkl', "wb") as f:
    pickle.dump(t1_high_w_range_coefs_1, f)
    pickle.dump(t1_high_w_range_coefs_2, f)
    pickle.dump(t1_high_w_range_coefs_3, f)
    pickle.dump(t1_high_w_range_coefs_4, f)
    
save_csv('t1_high_w_range_coefs', [t1_high_w_range_coefs_1, t1_high_w_range_coefs_2, t1_high_w_range_coefs_3, t1_high_w_range_coefs_4], file_path + 'hist_data/')

## High v variance and w range
print('High v and w Case')
with open(file_path + 'data/t1_high_v_and_w.pkl', 'rb') as f:
    t1_high_v_and_w = pickle.load(f)

t1_high_v_and_w_coefs_1, t1_high_v_and_w_coefs_2, t1_high_v_and_w_coefs_3, t1_high_v_and_w_coefs_4 = run_simulations(t1_high_v_and_w, basic_specification, basic_names)

with open(file_path + 'coefs/t1_high_v_and_w_coefs.pkl', "wb") as f:
    pickle.dump(t1_high_v_and_w_coefs_1, f)
    pickle.dump(t1_high_v_and_w_coefs_2, f)
    pickle.dump(t1_high_v_and_w_coefs_3, f)
    pickle.dump(t1_high_v_and_w_coefs_4, f)

save_csv('t1_high_v_and_w_coefs', [t1_high_v_and_w_coefs_1, t1_high_v_and_w_coefs_2, t1_high_v_and_w_coefs_3, t1_high_v_and_w_coefs_4], file_path + 'hist_data/')
    
# If runtime reaches time limit of your system, split the code here for different runs.
print("Trial 2")

## Base data
print('Base Case')
with open(file_path + 'data/t2_base.pkl', 'rb') as f:
    t2_base = pickle.load(f)

t2_base_coefs_1, t2_base_coefs_2, t2_base_coefs_3, t2_base_coefs_4 = run_simulations(t2_base, basic_specification, basic_names)

with open(file_path + 'coefs/t2_base_coefs.pkl', "wb") as f:
    pickle.dump(t2_base_coefs_1, f)
    pickle.dump(t2_base_coefs_2, f)
    pickle.dump(t2_base_coefs_3, f)
    pickle.dump(t2_base_coefs_4, f)

save_csv('t2_base_coefs', [t2_base_coefs_1, t2_base_coefs_2, t2_base_coefs_3, t2_base_coefs_4], file_path + 'hist_data/')
    
## High v variance
print('High v Case')
with open(file_path + 'data/t2_high_v_var.pkl', 'rb') as f:
    t2_high_v_var = pickle.load(f)

t2_high_v_var_coefs_1, t2_high_v_var_coefs_2, t2_high_v_var_coefs_3, t2_high_v_var_coefs_4 = run_simulations(t2_high_v_var, basic_specification, basic_names)

with open(file_path + 'coefs/t2_high_v_var_coefs.pkl', "wb") as f:
    pickle.dump(t2_high_v_var_coefs_1, f)
    pickle.dump(t2_high_v_var_coefs_2, f)
    pickle.dump(t2_high_v_var_coefs_3, f)
    pickle.dump(t2_high_v_var_coefs_4, f)

save_csv('t2_high_v_var_coefs', [t2_high_v_var_coefs_1, t2_high_v_var_coefs_2, t2_high_v_var_coefs_3, t2_high_v_var_coefs_4], file_path + 'hist_data/')
    
# If runtime reaches time limit of your system, split the code here for different runs.

## High w range
print('High w Case')
with open(file_path + 'data/t2_high_w_range.pkl', 'rb') as f:
    t2_high_w_range = pickle.load(f)

t2_high_w_range_coefs_1, t2_high_w_range_coefs_2, t2_high_w_range_coefs_3, t2_high_w_range_coefs_4 = run_simulations(t2_high_w_range, basic_specification, basic_names)

with open(file_path + 'coefs/t2_high_w_range_coefs.pkl', "wb") as f:
    pickle.dump(t2_high_w_range_coefs_1, f)
    pickle.dump(t2_high_w_range_coefs_2, f)
    pickle.dump(t2_high_w_range_coefs_3, f)
    pickle.dump(t2_high_w_range_coefs_4, f)

save_csv('t2_high_w_range_coefs', [t2_high_w_range_coefs_1, t2_high_w_range_coefs_2, t2_high_w_range_coefs_3, t2_high_w_range_coefs_4], file_path + 'hist_data/')
    
print('High v and w Case')
## High v variance and w range
with open(file_path + 'data/t2_high_v_and_w.pkl', 'rb') as f:
    t2_high_v_and_w = pickle.load(f)

t2_high_v_and_w_coefs_1, t2_high_v_and_w_coefs_2, t2_high_v_and_w_coefs_3, t2_high_v_and_w_coefs_4 = run_simulations(t2_high_v_and_w, basic_specification, basic_names)

with open(file_path + 'coefs/t2_high_v_and_w_coefs.pkl', "wb") as f:
    pickle.dump(t2_high_v_and_w_coefs_1, f)
    pickle.dump(t2_high_v_and_w_coefs_2, f)
    pickle.dump(t2_high_v_and_w_coefs_3, f)
    pickle.dump(t2_high_v_and_w_coefs_4, f)

save_csv('t2_high_v_and_w_coefs', [t2_high_v_and_w_coefs_1, t2_high_v_and_w_coefs_2, t2_high_v_and_w_coefs_3, t2_high_v_and_w_coefs_4], file_path + 'hist_data/')