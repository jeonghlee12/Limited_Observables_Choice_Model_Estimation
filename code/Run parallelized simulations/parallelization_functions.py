# ad-hoc approach to import functions from parent directory
import sys
sys.path.append('..')
from functions import normalize_data, fill_all_price_with_average, fill_nonobserved_with_average, fill_all_price_with_difference

def task(df):
    N = 10000
    J = 5   

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

    data1 = normalize_data(df, N, J)
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

    data2 = normalize_data(fill_all_price_with_average(df, N, J), N, J)
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

    data3 = normalize_data(fill_nonobserved_with_average(df, N, J), N, J)
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


    data4 = normalize_data(fill_all_price_with_difference(df, N, J), N, J)
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

    return [a1, g1, b1, a2, g2, b2, a3, g3, b3, a4, g4, b4]