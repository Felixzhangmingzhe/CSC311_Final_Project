# TODO: complete this file.
import torch
import numpy as np
from sklearn.utils import resample
from torch.autograd import Variable

from neural_network import AutoEncoder, load_data, train
from utils import evaluate
from utils import load_public_test_csv, load_train_csv, load_train_sparse, load_valid_csv

import torch


def train_ensemble(models, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, n_models=3):
    ensemble_models = []
    for i in range(n_models):
        indices = np.random.choice(range(len(train_data)), len(train_data))
        boot_train_data = train_data[indices]
        boot_zero_train_data = zero_train_data[indices]

        model = AutoEncoder(num_question=train_data.shape[1], k=50)
        print(f"Training model {i + 1}")

        train_losses, valid_losses = train(model, lr, lamb, boot_train_data, boot_zero_train_data, valid_data,
                                           num_epoch)

        ensemble_models.append(model)

    return ensemble_models

def predict_ensemble(models, zero_train_data):
    total_predictions = torch.zeros_like(zero_train_data)
    for model in models:
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(zero_train_data).float())
            total_predictions += predictions

    average_predictions = total_predictions / len(models)
    return average_predictions


def evaluate_ensemble(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # # Tell PyTorch you are evaluating the model.
    # model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = predict_ensemble(model, inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    n_models = 3
    num_epoch = 50
    lr = 0.01
    lamb = 0.01

    models = [AutoEncoder(num_question=train_matrix.shape[1], k=50) for _ in range(n_models)]

    ensemble_models = train_ensemble(models, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, n_models)


    test_accuracy = evaluate_ensemble(ensemble_models, zero_train_matrix, test_data)
    valid_accuracy = evaluate_ensemble(ensemble_models, zero_train_matrix, valid_data)

    # prdictions = predict_ensemble(ensemble_models, test_data)
    # test_accuracy = evaluate(test_data, prdictions, threshold=0.5)

    print(f"Test Accuracy: {test_accuracy}")
    print(f"Validation Accuracy: {valid_accuracy}")


if __name__ == '__main__':
    main()
