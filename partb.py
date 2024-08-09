import random

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

### DO NOT DELETE THIS PART ###
# import os
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:\\Users\\张明哲\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\PyQt5\\Qt5\\plugins'
### DO NOT DELETE THIS PART ###


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0

    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        z = theta[user] - beta[question]
        correct = data["is_correct"][i]
        log_lklihood += correct * z - np.log(1 + np.exp(z))

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    gradient_theta = np.zeros_like(theta)
    gradient_beta = np.zeros_like(beta)

    for i in range(len(data["user_id"])):
        user = data["user_id"][i]
        question = data["question_id"][i]
        z = theta[user] - beta[question]
        correct = data["is_correct"][i]

        gradient_theta[user] += correct - sigmoid(z)
        gradient_beta[question] -= correct - sigmoid(z)

    theta += lr * gradient_theta
    beta += lr * gradient_beta

    return theta, beta


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(len(data["user_id"]))
    beta = np.zeros(len(data["question_id"]))

    train_acc_lst = []
    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)

        train_lld_lst.append(-train_neg_lld)
        val_lld_lst.append(-val_neg_lld)

        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)

        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        print("Iteration: {}, Train NLLK: {}, Train Score: {}, Val NLLK: {}, Val Score: {}".format(i, train_neg_lld,
                                                                                                   train_score,
                                                                                                   val_neg_lld,
                                                                                                   val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst


################################################################################
# ---------------------- above are regular irt method -------------------------
# ---------------------- below are ensemble method (new) ----------------------
################################################################################


def split_data(data, ratio=0.8):
    """
    Split the data
    """
    sample_size = int(len(data['user_id']) * ratio)
    indices = list(range(len(data['user_id'])))
    sampled_indices = random.sample(indices, sample_size)
    random_slice = {key: [value[i] for i in sampled_indices] for key, value in data.items()}
    return random_slice


def irt_ensemble(train_data, val_data, lr, iterations):
    """
    Train the ensemble model
    """
    theta_lst = []
    beta_lst = []
    repeat = 4
    # resample 3 times with replacement from the training data
    for i in range(repeat):
        # divide the training data into subsets
        random_slice = split_data(train_data, ratio=0.8)

        print("training model {}".format(i))

        theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst = irt(
            random_slice, val_data, lr, iterations
        )
        theta_lst.append(theta)
        beta_lst.append(beta)

    return theta_lst, beta_lst


def evaluate_ensemble(theta_lst, beta_lst, data):
    """Evaluate the ensemble model given data and return the accuracy.

    :param theta_lst: List of theta vectors
    :param beta_lst: List of beta vectors
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :return: float
    """
    total_correct = 0
    for i in range(len(data["user_id"])):
        prdict_lst = []
        for j in range(len(theta_lst)):
            user = data["user_id"][i]
            question = data["question_id"][i]
            z = theta_lst[j][user] - beta_lst[j][question]
            p_a = sigmoid(z)
            prdict_lst.append(p_a)
        total_prob = np.sum(prdict_lst)
        predict = total_prob / len(theta_lst)
        if predict >= 0.5:
            predict = 1
        else:
            predict = 0
        if predict == data["is_correct"][i]:
            total_correct += 1

    return total_correct / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    ### DO NOT DELETE THIS PART ###
    # train_data = load_train_csv("C:\\CSC311_Final_Project\\data")
    # # You may optionally use the sparse matrix.
    # # sparse_matrix = load_train_sparse("C:\\CSC311_Final_Project\\data")
    # val_data = load_valid_csv("C:\\CSC311_Final_Project\\data")
    # test_data = load_public_test_csv("C:\\CSC311_Final_Project\\data")
    ### DO NOT DELETE THIS PART ###

    lr = 0.01
    iterations = 50

    theta_lst, beta_lst = irt_ensemble(train_data, val_data, lr, iterations)

    ensemble_train_acc = evaluate_ensemble(theta_lst, beta_lst, train_data)
    print("Ensemble Training Accuracy: {}".format(ensemble_train_acc))

    # evaluate the ensemble model
    ensemble_acc = evaluate_ensemble(theta_lst, beta_lst, val_data)
    print("Ensemble Validation Accuracy: {}".format(ensemble_acc))

    ensemble_test_acc = evaluate_ensemble(theta_lst, beta_lst, test_data)
    print("Ensemble Test Accuracy: {}".format(ensemble_test_acc))


if __name__ == "__main__":
    main()
