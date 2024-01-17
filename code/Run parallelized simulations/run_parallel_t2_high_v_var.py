# -------------------------------- Import packages --------------------------------
import pandas as pd
import numpy as np

# ignore warning output
import warnings
warnings.filterwarnings('ignore')

# read and write python objects
import pickle

# import parallelization functions
from parallelization_functions import task

# ad-hoc approach to import functions from parent directory
import sys
sys.path.append('..')
from functions import save_csv

# multiprocessing
import multiprocessing
from multiprocessing import Pool

file_path = "../../output/"

## 
with open(file_path + 'data/t2_high_v_var.pkl', 'rb') as f:
    t2_high_v_var = pickle.load(f)

# coefficients for version 1
a_coefs_1, g_coefs_1, b_coefs_1 = np.zeros(1000), np.zeros(1000), np.zeros(1000)

# coefficients for version 2
a_coefs_2, g_coefs_2, b_coefs_2 = np.zeros(1000), np.zeros(1000), np.zeros(1000)

# coefficients for version 3
a_coefs_3, g_coefs_3, b_coefs_3 = np.zeros(1000), np.zeros(1000), np.zeros(1000)

# coefficients for version 4
a_coefs_4, g_coefs_4, b_coefs_4 = np.zeros(1000), np.zeros(1000), np.zeros(1000)

index = 0

print("started process")
print("number of cores:", multiprocessing.cpu_count())
if __name__ == '__main__':
    with Pool() as pool:
        for result in pool.imap_unordered(task, t1_base):
            a_coefs_1[index] = result[0]
            g_coefs_1[index] = result[1]
            b_coefs_1[index] = result[2]
            a_coefs_2[index] = result[3]
            g_coefs_2[index] = result[4]
            b_coefs_2[index] = result[5]
            a_coefs_3[index] = result[6]
            g_coefs_3[index] = result[7]
            b_coefs_3[index] = result[8]
            a_coefs_4[index] = result[9]
            g_coefs_4[index] = result[10]
            b_coefs_4[index] = result[11]
            index = index + 1
            print("current index: ", index - 1)

coefs1 = pd.DataFrame({"alpha": a_coefs_1, "gamma": g_coefs_1, "beta": b_coefs_1})
coefs2 = pd.DataFrame({"alpha": a_coefs_2, "gamma": g_coefs_2, "beta": b_coefs_2})
coefs3 = pd.DataFrame({"alpha": a_coefs_3, "gamma": g_coefs_3, "beta": b_coefs_3})
coefs4 = pd.DataFrame({"alpha": a_coefs_4, "gamma": g_coefs_4, "beta": b_coefs_4})

with open(file_path + 'coefs/t2_high_v_var_coefs.pkl', "wb") as f:
    pickle.dump(coefs1, f)
    pickle.dump(coefs2, f)
    pickle.dump(coefs3, f)
    pickle.dump(coefs4, f)

save_csv('t2_high_v_var_coefs', [t2_high_v_var_coefs_1, t2_high_v_var_coefs_2, t2_high_v_var_coefs_3, t2_high_v_var_coefs_4], file_path + 'hist_data/')