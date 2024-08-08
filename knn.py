import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

### DO NOT DELETE THIS PART ###
# import os
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:\\Users\\张明哲\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\PyQt5\\Qt5\\plugins'
### DO NOT DELETE THIS PART ###


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    ### DO NOT DELETE THIS PART ###
    # sparse_matrix = load_train_sparse("C:\\CSC311_Final_Project\\data").toarray()
    # val_data = load_valid_csv("C:\\CSC311_Final_Project\\data")
    # test_data = load_public_test_csv("C:\\CSC311_Final_Project\\data")
    ### DO NOT DELETE THIS PART ###

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]

    user_acc = []
    item_acc = []

    for k in k_values:
        user_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))
        item_acc.append(knn_impute_by_item(sparse_matrix, val_data, k))

    plt.figure()
    plt.plot(k_values, user_acc, label='User-based CF')
    plt.plot(k_values, item_acc, label='Item-based CF')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs k')
    plt.legend()
    plt.savefig('q1.png')
    plt.show()

    k_star_user = k_values[user_acc.index(max(user_acc))]
    k_star_item = k_values[item_acc.index(max(item_acc))]

    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, k_star_user)
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_star_item)

    print("Test Accuracy on user-based CF with k* = {}: {}".format(k_star_user, test_acc_user))
    print("Test Accuracy on item-based CF with k* = {}: {}".format(k_star_item, test_acc_item))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
