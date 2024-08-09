from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

## DO NOT DELETE THIS PART ###
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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0

    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        z = theta[user] - beta[question]
        correct = data["is_correct"][i]
        log_lklihood += correct * z - np.log(1 + np.exp(z))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
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
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


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
    # TODO: Initialize theta and beta.
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

        print("Iteration: {}, Train NLLK: {}, Train Score: {}, Val NLLK: {}, Val Score: {}".format(i, train_neg_lld, train_score, val_neg_lld, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst


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

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 100
    iteration_lst = list(range(iterations))

    theta, beta, train_acc_lst, val_acc_lst, train_lld_lst, val_lld_lst = irt(
        train_data, val_data, lr, iterations
    )

    plt.figure()
    plt.plot(iteration_lst, train_acc_lst, label="Train Accuracy")
    plt.plot(iteration_lst, val_acc_lst, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Iterations")
    plt.legend()
    plt.savefig("irt_accuracy.png")
    plt.show()

    plt.figure()
    plt.plot(iteration_lst, train_lld_lst, label="Train Log-Likelihood")
    plt.plot(iteration_lst, val_lld_lst, label="Validation Log-Likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood vs Iterations")
    plt.legend()
    plt.savefig("irt_llk.png")
    plt.show()

    final_train_acc = evaluate(train_data, theta, beta)
    print("Final train Accuracy: {}".format(final_train_acc))

    final_val_acc = evaluate(val_data, theta, beta)
    final_test_acc = evaluate(test_data, theta, beta)
    print("Final Validation Accuracy: {}".format(final_val_acc))
    print("Final Test Accuracy: {}".format(final_test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j_1 = beta.argmin()
    j_2 = beta.argmax()
    j_3 = np.argsort(beta)[len(beta) // 2]

    theta = np.linspace(-5.0, 5.0, 100)

    for j in [j_1, j_2, j_3]:
        lines = []
        for i in range(len(theta)):
            x = theta[i] - beta[j]
            y = sigmoid(x)
            lines.append(y)
        plt.plot(theta, lines, label="Question {}".format(j))

    plt.xlabel("Theta")
    plt.ylabel("P(Correct)")
    plt.title("P(Correct) vs Theta")
    plt.legend()
    plt.savefig("irt_question.png")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
