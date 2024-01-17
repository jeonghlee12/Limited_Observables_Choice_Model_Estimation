import numpy as np
import pandas as pd
import math
import pylogit as pl

# For plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec


def create_data(N, J, a, g, b, rng, v_var = 1, w_range = 2):    
    # set regressors
    v = rng.normal(0, v_var, (N, J))
    X = np.repeat([range(J)], N, axis = 0) + v
    w = rng.uniform(-1 * w_range, w_range, size = (N, J))
    P = X + w
    
    # construct Y
    Y = np.zeros(shape = (N, J), dtype = int)
    for i in range(N):
        U = [a, a, a, a, a] - g * P[i] + b * X[i]  + rng.normal(0, math.pi / math.sqrt(6), J)
        idx = np.argmax(U)
        Y[i][idx] = 1

    # create data
    X = X.flatten()
    P = P.flatten()
    Y = Y.flatten()
    n = np.arange(1, N + 1, 1).repeat(J)  # add individual ID
    p = np.tile(np.arange(0, J, 1), N)    # add product ID

    df = pd.DataFrame({'i': n, "j": p, 'Y': Y, 'X': X, 'P': P})

    return df

def fill_all_price_with_average(df, N, J):
    # omit non chosen products
    omitted_data = df[df.Y == 1]
    
    # get average of P for chosen products
    averages = omitted_data.groupby('j').mean()
    P_j_hat = averages.P
    
    # create new column of predicted prices based on product
    pred_P = np.tile(np.array(P_j_hat), N)
    
    # replace all observed prices with predicted price
    data_hat = df.assign(P = pred_P)

    return data_hat

def fill_nonobserved_with_average(df, N, J):
    # omit non chosen products
    omitted_data = df[df.Y == 1]
    
    # get average of X and P for chosen products
    averages = omitted_data.groupby('j').mean()
    P_j_hat = averages.P
    
    # create new datatable with NaN for products not chosen by individual
    i = np.arange(1, N + 1, 1).repeat(J)
    j = np.tile(np.arange(0, J, 1), N)
    template = pd.DataFrame({'i': i, "j": j})
    data_hat = template.merge(omitted_data, how = 'left', on = ['i', 'j'])
    
    # fill the NaN values for not chosen product by individual with average P for product found above
    data_hat['Y'] = df.Y
    data_hat['X'] = df.X
    for prod in range(J):
        data_hat['P'] = np.where((data_hat['j'] == prod) & (data_hat['Y'] == 0), P_j_hat[prod], data_hat.P)

    return data_hat

def fill_all_price_with_difference(df, N, J):
    temp = fill_all_price_with_average(df, N, J)
    
    # create new column of of X_ij - j
    X_ij_j = df['X'] - df['j']
    
    pred_P = temp['P'] + X_ij_j
    
    # replace all observed prices with predicted price + X_ij - j
    data_hat = df.assign(P = pred_P)
        
    return data_hat

def normalize_data(df, N, J):
    # normalize to first product index for each individual
    refX = df.values[0::J, 3].repeat(J)
    refP = df.values[0::J, 4].repeat(J)


    new_X = df.values[:, 3] - refX
    new_P = df.values[:, 4] - refP
    
    new_data = df.assign(X = new_X, P = new_P)
    return new_data

def run_simulations(trial_data, basic_specification, basic_names):
    n = len(trial_data)
    
    # coefficients for version 1
    a_coefs_1 = []
    g_coefs_1 = []
    b_coefs_1 = []

    # coefficients for version 2
    a_coefs_2 = []
    g_coefs_2 = []
    b_coefs_2 = []

    # coefficients for version 3
    a_coefs_3 = []
    g_coefs_3 = []
    b_coefs_3 = []

    # coefficients for version 4
    a_coefs_4 = []
    g_coefs_4 = []
    b_coefs_4 = []
    
    # Set parameters
    N = 10000
    J = 5
    
    for t in range(n):
        data = trial_data[t]

        # version 1 regression
        data1 = normalize_data(data, N, J)
        model1 = pl.create_choice_model(data1,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model1.fit_mle(np.zeros(3), print_res = False)
        a1 = model1.params.intercept
        g1 = -1 * model1.params.P
        b1 = model1.params.X
        a_coefs_1.append(a1)
        g_coefs_1.append(g1)
        b_coefs_1.append(b1)

        # version 2 regression    
        data2 = normalize_data(fill_all_price_with_average(data, N, J), N, J)
        model2 = pl.create_choice_model(data2,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model2.fit_mle(np.zeros(3), print_res = False)
        a2 = model2.params.intercept
        g2 = -1 * model2.params.P
        b2 = model2.params.X
        a_coefs_2.append(a2)
        g_coefs_2.append(g2)
        b_coefs_2.append(b2)

        # version 3 regression
        data3 = normalize_data(fill_nonobserved_with_average(data, N, J), N, J)
        model3 = pl.create_choice_model(data3,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model3.fit_mle(np.zeros(3), print_res = False)
        a3 = model3.params.intercept
        g3 = -1 * model3.params.P
        b3 = model3.params.X
        a_coefs_3.append(a3)
        g_coefs_3.append(g3)
        b_coefs_3.append(b3)

        # version 4 regression    
        data4 = normalize_data(fill_all_price_with_difference(data, N, J), N, J)
        model4 = pl.create_choice_model(data4,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model4.fit_mle(np.zeros(3), print_res = False)
        a4 = model4.params.intercept
        g4 = -1 * model4.params.P
        b4 = model4.params.X
        a_coefs_4.append(a4)
        g_coefs_4.append(g4)
        b_coefs_4.append(b4)
        
    coefs1 = pd.DataFrame({"alpha": a_coefs_1, "gamma": g_coefs_1, "beta": b_coefs_1})
    coefs2 = pd.DataFrame({"alpha": a_coefs_2, "gamma": g_coefs_2, "beta": b_coefs_2})
    coefs3 = pd.DataFrame({"alpha": a_coefs_3, "gamma": g_coefs_3, "beta": b_coefs_3})
    coefs4 = pd.DataFrame({"alpha": a_coefs_4, "gamma": g_coefs_4, "beta": b_coefs_4})
    
    return coefs1, coefs2, coefs3, coefs4


def plot_coefs(data_desc, coefs):
    T = len(coefs[0])
    fig = plt.figure(constrained_layout = True, figsize = (20, 15))
    fig.suptitle("Distribution of Coefficients - " + data_desc, fontsize = 20)
    subfigs = fig.subfigures(nrows = 3, ncols = 1)
    
    for row, subfig in enumerate(subfigs):
        current_var = '$\\' + coefs[row].columns[row] + '$'
        subfig.suptitle(current_var, fontsize = 17)

        axs = subfig.subplots(nrows=1, ncols=3)
        max_y = 0
        min_x = 50
        max_x = 0
        for col, ax in enumerate(axs):
            # plot histogram
            h, _, _ = ax.hist(coefs[col].iloc[:, row], density = False, bins = 30)
            # calculate average
            avg = np.mean(coefs[col].iloc[:, row])
            ax.axvline(x = avg, color='black', lw = 1, ls = '--')

            # save max bar height to sync y-axis across plots        
            max_y = max(max_y, math.ceil(100 * max(h) / T) / 100.)
            
            # save min and max x-axis values to sync
            locs = ax.get_xticks() 
            min_x = min(min_x, min(locs))
            max_x = max(max_x, max(locs))

            # stylizing
            if col == 0:
                ax.set_ylabel('Density $\\' + coefs[row].columns[row] + '$', fontsize = 14)
                locs = ax.get_yticks() 
                ax.set_yticks(locs, np.round(locs / T,3))
            else:
                ax.set_yticks([], [])

            ax.tick_params(axis='x', labelsize = 14)
            ax.tick_params(axis='y', labelsize = 14)
            ax.set_title(f'Version {col + 1}', fontsize = 16)

        # sync x-axis and y-axis scales
        for col, ax in enumerate(axs):
            ax.set_ylim(0, max_y * T)
            ax.set_xlim(min_x, max_x)

def plot_coefs4(data_desc, coefs):
    T = len(coefs[0])
    fig, subfigs = plt.subplots(nrows = 4, ncols = 3, sharex='col', sharey='row', constrained_layout = True, figsize = (20, 15))
    fig.suptitle("Distribution of Coefficients - " + data_desc, fontsize = 20)
    
    max_y = 0
    for row, subfig in enumerate(subfigs):
        for col, ax in enumerate(subfig):
            # plot histogram
            h, b, _ = ax.hist(coefs[row].iloc[:, col], density = False, bins = 30)
            # calculate average
            avg = np.mean(coefs[row].iloc[:, col])
            ax.axvline(x = avg, color='black', lw = 1, ls = '--')

            # save max bar height to sync y-axis across plots        
            max_y = max(max_y, math.ceil(100 * max(h) / T) / 100.)
                        
            # stylizing
            if col == 0:
                ax.set_ylabel('Density', fontsize = 14)
            
#             if row == 0:
            ax.set_title('$\\' + coefs[row].columns[col] + '$', fontsize = 15)
                
#             if row == 3:
#                 ax.set_xlabel('$\\' + coefs[row].columns[col] + '$', fontsize = 15)
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.tick_params(axis='x', labelsize = 14)
            ax.tick_params(axis='y', labelsize = 14)

    # sync y-axis scales
    for row, subfig in enumerate(subfigs):
        for col, ax in enumerate(subfig):
            ax.set_ylim(0, max_y * T)
            locs = ax.get_yticks() 
            ax.set_yticks(locs, np.round(locs / T,3))
            
            
    grid = subfigs[0][0].get_gridspec()
    create_subtitle(fig, grid[0, ::], 'Base Case')
    create_subtitle(fig, grid[1, ::], 'High $\\sigma_v$')
    create_subtitle(fig, grid[2, ::], 'High $\\omega$ Range')
    create_subtitle(fig, grid[3, ::], 'High $\\sigma_v$ and $\\omega$ Range')
    
def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontsize=16)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')
    
def plot_coefs_col_based(data_desc, coefs):
    T = len(coefs[0])
    fig, subfigs = plt.subplots(nrows = 3, ncols = 3, sharex='col', sharey='row', constrained_layout = True, figsize = (20, 15))
    fig.suptitle("Distribution of Coefficients - " + data_desc, fontsize = 20)
    
    max_y = 0
    for row, subfig in enumerate(subfigs):
        for col, ax in enumerate(subfig):
            # plot histogram
            h, _, _ = ax.hist(coefs[row].iloc[:, col], density = False, bins = 30)
            # calculate average
            avg = np.mean(coefs[row].iloc[:, col])
            ax.axvline(x = avg, color='black', lw = 1, ls = '--')

            # save max bar height to sync y-axis across plots        
            max_y = max(max_y, math.ceil(100 * max(h) / T) / 100.)

            # stylizing
            if col == 0:
                ax.set_ylabel('Density', fontsize = 14)
            ax.set_title('$\\' + coefs[row].columns[col] + '$', fontsize = 15)
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.tick_params(axis='x', labelsize = 14)
            ax.tick_params(axis='y', labelsize = 14)
#             ax.set_title(f'Version {col + 1}', fontsize = 16)

    # sync y-axis scales
    for row, subfig in enumerate(subfigs):
        for col, ax in enumerate(subfig):
            ax.set_ylim(0, max_y * T)
            locs = ax.get_yticks() 
            ax.set_yticks(locs, np.round(locs / T,3))
    
    grid = subfigs[0][0].get_gridspec()
    create_subtitle(fig, grid[0, ::], 'Version 1')
    create_subtitle(fig, grid[1, ::], 'Version 2')
    create_subtitle(fig, grid[2, ::], 'Version 3')

def plot_coefs_all(data_desc, coefs):
    T = len(coefs[0])
    fig, subfigs = plt.subplots(nrows = 4, ncols = 3, sharex='col', sharey='row', constrained_layout = True, figsize = (20, 15))
    fig.suptitle("Distribution of Coefficients - " + data_desc, fontsize = 20)
    
    max_y = 0
    for row, subfig in enumerate(subfigs):
        for col, ax in enumerate(subfig):
            # plot histogram
            h, b, _ = ax.hist(coefs[row].iloc[:, col], density = False, bins = 30)
            # calculate average
            avg = np.mean(coefs[row].iloc[:, col])
            ax.axvline(x = avg, color='black', lw = 1, ls = '--')

            # save max bar height to sync y-axis across plots        
            max_y = max(max_y, math.ceil(100 * max(h) / T) / 100.)
                        
            # stylizing
            if col == 0:
                ax.set_ylabel('Density', fontsize = 14)
            
#             if row == 0:
            ax.set_title('$\\' + coefs[row].columns[col] + '$', fontsize = 15)
                
#             if row == 3:
#                 ax.set_xlabel('$\\' + coefs[row].columns[col] + '$', fontsize = 15)
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.tick_params(axis='x', labelsize = 14)
            ax.tick_params(axis='y', labelsize = 14)

    # sync y-axis scales
    for row, subfig in enumerate(subfigs):
        for col, ax in enumerate(subfig):
            ax.set_ylim(0, max_y * T)
            locs = ax.get_yticks() 
            ax.set_yticks(locs, np.round(locs / T,3))
            
            
    grid = subfigs[0][0].get_gridspec()
    create_subtitle(fig, grid[0, ::], 'Version 1')
    create_subtitle(fig, grid[1, ::], 'Version 2')
    create_subtitle(fig, grid[2, ::], 'Version 3')
    create_subtitle(fig, grid[3, ::], 'Version 4')



def run_simul123(trial_data, basic_specification, basic_names):
    n = len(trial_data)
    
    # coefficients for version 1
    a_coefs_1 = []
    g_coefs_1 = []
    b_coefs_1 = []

    # coefficients for version 2
    a_coefs_2 = []
    g_coefs_2 = []
    b_coefs_2 = []

    # coefficients for version 3
    a_coefs_3 = []
    g_coefs_3 = []
    b_coefs_3 = []
    
    # Set parameters
    N = 10000
    J = 5
    
    for t in range(n):
        print("\tcurrent index:", t)
        data = trial_data[t]

        # version 1 regression
        data1 = normalize_data(data, N, J)
        model1 = pl.create_choice_model(data1,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model1.fit_mle(np.zeros(3), print_res = False)
        a1 = model1.params.intercept
        g1 = -1 * model1.params.P
        b1 = model1.params.X
        a_coefs_1.append(a1)
        g_coefs_1.append(g1)
        b_coefs_1.append(b1)

        # version 2 regression    
        data2 = normalize_data(fill_all_price_with_average(data, N, J), N, J)
        model2 = pl.create_choice_model(data2,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model2.fit_mle(np.zeros(3), print_res = False)
        a2 = model2.params.intercept
        g2 = -1 * model2.params.P
        b2 = model2.params.X
        a_coefs_2.append(a2)
        g_coefs_2.append(g2)
        b_coefs_2.append(b2)
        
        # version 3 regression
        data3 = normalize_data(fill_nonobserved_with_average(data, N, J), N, J)
        model3 = pl.create_choice_model(data3,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model3.fit_mle(np.zeros(3), print_res = False)
        a3 = model3.params.intercept
        g3 = -1 * model3.params.P
        b3 = model3.params.X
        a_coefs_3.append(a3)
        g_coefs_3.append(g3)
        b_coefs_3.append(b3)

                
    coefs1 = pd.DataFrame({"alpha": a_coefs_1, "gamma": g_coefs_1, "beta": b_coefs_1})
    coefs2 = pd.DataFrame({"alpha": a_coefs_2, "gamma": g_coefs_2, "beta": b_coefs_2})
    coefs3 = pd.DataFrame({"alpha": a_coefs_3, "gamma": g_coefs_3, "beta": b_coefs_3})
    
    return coefs1, coefs2, coefs3

def run_simul4(trial_data, basic_specification, basic_names):
    n = len(trial_data)
    
    # coefficients for version 4
    a_coefs_4 = []
    g_coefs_4 = []
    b_coefs_4 = []
    
    # Set parameters
    N = 10000
    J = 5
    
    for t in range(n):
        data = trial_data[t]

        # version 4 regression    
        data4 = normalize_data(fill_all_price_with_difference(data, N, J), N, J)
        model4 = pl.create_choice_model(data4,
                                   alt_id_col = "j",
                                   obs_id_col = "i",
                                   choice_col = "Y",
                                   specification = basic_specification,
                                   model_type = "MNL",
                                   names = basic_names)
        model4.fit_mle(np.zeros(3), print_res = False)
        a4 = model4.params.intercept
        g4 = -1 * model4.params.P
        b4 = model4.params.X
        a_coefs_4.append(a4)
        g_coefs_4.append(g4)
        b_coefs_4.append(b4)
        
    coefs4 = pd.DataFrame({"alpha": a_coefs_4, "gamma": g_coefs_4, "beta": b_coefs_4})
    
    return coefs4