import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)

### DO NOT DELETE THIS PART ###
# import os
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:\\Users\\张明哲\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\PyQt5\\Qt5\\plugins'
### DO NOT DELETE THIS PART ###


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs

        out = F.sigmoid(self.g(out))
        out = F.sigmoid(self.h(out))

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_losses = []
    valid_accuracy = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            # loss.backward()

            combined_loss = loss + 0.5 * lamb * model.get_weight_norm()
            combined_loss.backward()

            # train_loss += loss.item()
            train_loss += combined_loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        train_losses.append(train_loss)
        valid_accuracy.append(valid_acc)

        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )

    return train_losses, valid_accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # choose k=50, lr=0.01, lamb=0.01, num_epoch=50
    # Best model with k=50, lr=0.01, lambda=0.01, Validation Accuracy: 0.6895286480383855
    k = 50
    model = AutoEncoder(train_matrix.shape[1], k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 50
    lamb = 0.01

    train_loss, valid_accuracy = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # Next, evaluate your network on validation/test data
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    test_acc = evaluate(model, zero_train_matrix, test_data)

    print("Validation Accuracy: ", valid_acc)
    print("Test Accuracy: ", test_acc)

    # plot the training and validation loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(valid_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ks = [10, 50, 100, 200, 500]
    # learning_rates = [0.01, 0.001, 0.0001]
    # lambdas = [0.01, 0.001, 0.0001]
    # num_epochs = 50
    #
    # best_validation_accuracy = 0
    # best_k = None
    # best_lr = None
    # best_lambda = None
    #
    # for k in ks:
    #     for lr in learning_rates:
    #         for lamb in lambdas:
    #             print(f'Training model with k={k}, lr={lr}, lambda={lamb}')
    #             model = AutoEncoder(train_matrix.shape[1], k)
    #             train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epochs)
    #             validation_accuracy = evaluate(model, zero_train_matrix, valid_data)
    #             print(f'Validation Accuracy: {validation_accuracy}')
    #
    #             if validation_accuracy > best_validation_accuracy:
    #                 best_validation_accuracy = validation_accuracy
    #                 best_k = k
    #                 best_lr = lr
    #                 best_lambda = lamb
    #
    # print( f'Best model with k={best_k}, lr={best_lr}, lambda={best_lambda}, Validation Accuracy: {
    # best_validation_accuracy}')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

