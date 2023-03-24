from user_import import *


def generate_true_theta(rng, size, method="standard_normal"):
    """
    Generate the true theta (common factor) according to a random number generating method.
    :param rng: A random number generator.
    :param size: The dimension of theta.
    :param method: The random number generating method.
    :return: The true theta
    """
    if method == "standard_normal":
        theta = rng.standard_normal(size)  # generate standard normal random numbers
    elif method == "uniform":
        theta = rng.uniform(size=size)  # generate uniform random numbers
    else:
        theta = np.ones(size)  # generate all ones
    theta = theta / np.linalg.norm(theta)  # normalize the length
    return theta


def generate_labels(X, alpha, beta, thres=0.5):
    """
    Generate labels according to a logistic regression with intercept and coefficient as alpha and beta.
    :param X: The features.
    :param alpha: The intercept of a logistic regression.
    :param beta: The coefficients of a logistic regression.
    :param thres: The threshold to decide class 0 v.s. 1.
    :return: The generated labels
    """
    P = sigmoid(alpha + np.dot(X, beta))  # the probability
    y = (P > thres) * 1  # the labels
    return y


def generate_synthetic_data(rng, theta, N, p, thres=0.5):
    """
    Generate synthetic data X(features) and Y(labels).
    :param rng: A random number generator.
    :param theta: The true common factor.
    :param N: Sample size.
    :param p: Dimension.
    :param thres: The threshold to decide class 0 v.s. 1.
    :return: Features and labels
    """
    X = rng.uniform(size=(N, p))  # generate features

    # generate intercept of the true logistic regression
    X_beta = np.dot(X, theta)
    alpha = - X_beta[:, 0].mean()  ######################### alpha的产生方式可以改

    # generate labels of the true logistic regression
    Probs = sigmoid(alpha + X_beta)  # generate true probability
    Y = (Probs > thres) * 1  # generate true labels
    print(f"Class 1 Ratio: {Y.sum() / Y.shape[0]}")
    return X, Y, alpha


def generate_annotators_data(rng, theta, X, R, thres=0.5):
    """
    Generate true data for annotators.
    :param rng: A random number generator.
    :param theta: The true common factor.
    :param X: The features.
    :param R: The number of annotators.
    :param thres: The threshold to decide class 0 v.s. 1.
    :return: True data for annotators.
    """
    gammas = rng.uniform(low=0.1, high=1.1, size=(R, 1))
    annotator_betas = np.dot(theta,
                             np.transpose(gammas))  # individual parameters # 本来alpha也是在这里设定，但是太容易后面出现全部label都是1或0

    # individual logistic decision - deciding intercept alpha
    X_beta = np.dot(X, annotator_betas)
    xid1 = rng.integers(low=0, high=X_beta.shape[0], size=R)
    xid2 = rng.integers(low=0, high=X_beta.shape[0], size=R)
    yid = np.arange(R)
    annotator_alphas = - (X_beta[xid1, yid] + X_beta[xid2, yid]) * 0.49
    # annotator_alphas = - np.median(X_beta, axis=0) # 如果这样设置，后面的annotator_labels就全一样了
    annotator_alphas = annotator_alphas.reshape((-1, R))  # alpha's shape = (1, R)

    # individual logistic decision - individual label assignment
    annotator_Probs = sigmoid(annotator_alphas + X_beta)
    annotator_labels = (annotator_Probs > thres) * 1
    return gammas, annotator_betas, annotator_alphas, annotator_labels


def estimate_annotator_data(X, annotator_betas, annotator_labels, R):
    """
    Estimate the logistic parameters with respect to alpha and beta.
    :param annotator_betas: The true beta of annotators.
    :param X: The features.
    :param annotator_labels: Annotators' labels.
    :param R: The number of the annotators.
    :return:
    """
    annotator_beta_hats = np.zeros_like(
        annotator_betas)  # an empty array to store the estimated betas of the annotators
    annotator_alpha_hats = np.zeros((R, 1))  # an empty array to store the estimated alphas of the annotators
    for r in range(R):
        modelLR = LogisticRegression()  # construct a logistic regression model
        modelLR.fit(X, annotator_labels[:, r])  # fit the [r]th annotator labels
        annotator_beta_hats[:, r] = modelLR.coef_  # the coefficient beta_hat
        annotator_alpha_hats[r] = modelLR.intercept_  # the intercept alpha_hat
    return annotator_beta_hats, annotator_alpha_hats


def estimate_theta_initial(annotator_beta_hats):
    """
    Estimate an initial estimator for theta.
    :param annotator_beta_hats: Annotators' beta_hat.
    :return: initial estimator for theta and gamma
    """
    Omega = np.dot(annotator_beta_hats, np.transpose(annotator_beta_hats))
    w, v = eigh(Omega)  # eigenvecs and eigenvalues
    theta_hat = v[:, -1]  # the max eigenvector
    gamma_hat = w[-1]  # the max eigenvalue
    return theta_hat, gamma_hat


def estimate_gammar(annotator_beta_hats, theta_hat, R):
    """
    Estimate each gamma^(r) for each annotator.
    :param annotator_beta_hats: The estimator for beta.
    :param theta_hat: The initial estimator for theta.
    :param R: The number of annotator.
    :return: gamma^(r)
    """
    gamma_hats = np.dot(np.transpose(annotator_beta_hats), theta_hat)
    gamma_hats = gamma_hats.reshape((R, 1))
    return gamma_hats


if __name__ == "__main__":
    R = 10  # the number of annotators
    p = 20  # the dimension of the feature vector
    N = 500  # the sample size
    thres = 0.5  # the threshold for deciding class 0 v.s. 1
    B = 200  # repeat times

    rng = np.random.default_rng(0)  # random number generator
    theta = generate_true_theta(rng, size=(p, 1), method="standard_normal")  # common factors

    # generate synthetic features and true labels
    X, Y, alpha = generate_synthetic_data(rng, theta, N, p)

    # generate annotator data
    gammas, annotator_betas, annotator_alphas, annotator_labels = generate_annotators_data(rng, theta, X, R, thres)

    # estimate annotators' parameters - beta_hat and alpha_hat
    annotator_beta_hats, annotator_alpha_hats = estimate_annotator_data(X, annotator_betas, annotator_labels, R)

    class1_ratios = annotator_labels.sum(axis=0) / annotator_labels.shape[0]
    for r in range(R): print(f"Annotator [{r}] Label 1 Ratio: {class1_ratios[r]}")
