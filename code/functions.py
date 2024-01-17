import numpy as np
import pandas as pd
import math

# Must include collections as pylogit is not currently updated for Python 3.10+
import sys
if sys.version_info[:3] >= (3, 10):
    import collections.abc
    collections.Iterable = collections.abc.Iterable
import pylogit as pl

# For plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec


def create_data(N, J, a, g, b, rng, var_v = 1, omega = 2):
    '''
    Generates the choice data
    
    Parameters:
        N (int): number of individuals
        J (int): number of products
        a (float): intercept coefficient (alpha)
        g (float): price coefficient (gamma)
        b (float): individual coefficient (beta)
        rng (numpy.random.Generator): random number generator
        var_v (float): variance of v
        omega (float): range of uniform distribution
    Returns:
        df (pandas.DataFrame): choice data
    '''
    # set regressors
    v = rng.normal(0, var_v, (N, J))
    X = np.repeat([range(J)], N, axis = 0) + v
    w = rng.uniform(-1 * omega, omega, size = (N, J))
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
    '''
    Modifies choice data for Treatment 2.
    Replaces price for each product for ALL choices with the average price of each product when chosen.
    
    Parameters:
        df (pandas.DataFrame): original choice data
        N (int): number of individuals
        J (int): number of products
    Returns:
        data_hat (pandas.DataFrame): modified choice data for treatment 2
    '''
    # omit non chosen products
    omitted_data = df[df.Y == 1]
    
    # get average of P for chosen products
    averages = omitted_data.groupby('j').mean()
    P_j_hat = averages.P
    
    # create new column of predicted prices based on product
    pred_P = [P_j_hat[j] for j in df.j]
    
    # replace all observed prices with predicted price
    data_hat = df.assign(P = pred_P)

    return data_hat


def fill_nonobserved_with_average(df, N, J):
    '''
    Modifies choice data for Treatment 3.
    Replaces price for each product not chosen with the average price of each product when chosen.
    
    Parameters:
        df (pandas.DataFrame): original choice data
        N (int): number of individuals
        J (int): number of products
    Returns:
        data_hat (pandas.DataFrame): modified choice data for treatment 3
    '''
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
    '''
    Modifies choice data for Treatment 4.
    Replaces price for ALL choices with the sum of the predicted price of Treatment 2 and the difference between individual characteristics and product index.
    
    Parameters:
        df (pandas.DataFrame): original choice data
        N (int): number of individuals
        J (int): number of products
    Returns:
        data_hat (pandas.DataFrame): modified choice data for treatment 4
    '''
    temp = fill_all_price_with_average(df, N, J)
    
    # create new column of of X_ij - j
    X_ij_j = df['X'] - df['j']
    
    pred_P = temp['P'] + X_ij_j
    
    # replace all observed prices with predicted price + X_ij - j
    data_hat = df.assign(P = pred_P)
        
    return data_hat


def normalize_data(df, N, J):
    '''
    Normalizes data to run conditional logit
    
    Parameters:
        df (pandas.DataFrame): original choice data
        N (int): number of individuals
        J (int): number of products
    Returns:
        new_data (pandas.DataFrame): normalized choice data
    '''
    # normalize to first product index for each individual
    new_X = np.zeros(N * J)
    new_P = np.zeros(N * J)
    index = 0
    for i in range(N):
        refX = df[(df.i == i + 1) & (df.j == 0)].iloc[0].X
        refP = df[(df.i == i + 1) & (df.j == 0)].iloc[0].P
        
        for j in range(J):
            new_X[index] = df[(df.i == i + 1) & (df.j == j)].iloc[0].X - refX
            new_P[index] = df[(df.i == i + 1) & (df.j == j)].iloc[0].P - refP
            index = index + 1
    new_data = df.assign(X = new_X)
    new_data = new_data.assign(P = new_P)

    return new_data


def run_simulations(trial_data, basic_specification, basic_names):
    '''
    Runs coefficient estimation for all 4 treatments using conditional logit
    
    Parameters:
        trial_data (pandas.DataFrame): original choice data
        basic_specification (OrderedDict): specification for coefficients
        basic_names (OrderedDict): names for the coefficient specifications
    Returns:
        coefs1 (pandas.DataFrame): coefficient data for Treatment 1
        coefs2 (pandas.DataFrame): coefficient data for Treatment 2
        coefs3 (pandas.DataFrame): coefficient data for Treatment 3
        coefs4 (pandas.DataFrame): coefficient data for Treatment 4
    '''
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


def save_csv(name, df_list, path):
    '''
    Saves coefficient data into a csv file
    
    Parameters:
        name (str): name of file
        df_list (list): list of DataFrame containing coefficients for each Tratment
        path (str): file path to save
    Returns:
        None
    '''
    comp = df_list[0].join(df_list[1].join(df_list[2].join(df_list[3], lsuffix = '3', rsuffix = '4')), lsuffix = '1', rsuffix = '2')
    comp.to_csv(path + name + "_h.csv", index = False)


def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    '''
    Auxiliary function used in plotting coefficient distributions.
    Sign sets of subplots with title.
    
    Parameters:
        fig (matplotlib.plt.Figure): figure object
        grid (matplotlib.SubplotSpec): subplot specifications
        title (str): title
    Returns:
        None
    '''
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontsize=16)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')
    

def plot_coefs_all(data_desc, coefs):
    '''
    Creates distribution plot of coefficents for each Treament.
    
    Parameters:
        data_desc (str): title specifics
        coefs (list): List of DataFrames containing coefficients for each Treatment.
    Returns:
        None
    '''
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
            
            ax.set_title('$\\' + coefs[row].columns[col] + '$', fontsize = 15)
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