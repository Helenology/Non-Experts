from user_import import *


def sigmoid(x):
    """
    Compute 1 / (1 + \exp(-x)).
    :param x: A scalar or a vector to be computed.
    :return: 1 / (1 + \exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def present_df(item_list, name_list=None):
    """
    Print each item in item_list in dataframe form.
    :param item_list: A list of items.
    :param name_list: A list names.
    :return:
    """
    if len(item_list) == 0:  # if item_list is empty, then return nothing
        return

    if name_list is None or len(name_list) == 0:  # if name_list is empty
        name_list = ["item" + str(i) for i in range(len(item_list))]  # create name as itemi

    if len(name_list) != len(item_list):  # if the length of name_list and item_list does not match, raise error
        raise ValueError(f"name_list does not match the length of item_list")

    tmp_dict = {}
    for i, item in enumerate(item_list):
        name = name_list[i]
        tmp_dict[name] = item.reshape(-1)
    tmp_df = pd.DataFrame(tmp_dict)
    print(tmp_df)


def compute_RMSE(theta, theta_hat):
    """
    Compute the RMSE(root MSE) of the parameter and estimator.
    :param theta: The true parameter.
    :param theta_hat: The estimator.
    :return: RMSE
    """
    theta_hat = theta_hat.reshape(theta.shape)  # reshape the estimator
    mse = np.mean((theta - theta_hat)**2)       # mse
    rmse = np.sqrt(mse)                         # rmse
    return rmse


