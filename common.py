import torch
import numpy as np

def get_perf(y_prob_predict, y_test):
    y_test = y_test.flatten()
    loss_fn = torch.nn.CrossEntropyLoss()
    y_log_prob = np.log(y_prob_predict)
    test_loss = loss_fn(torch.tensor(y_log_prob), torch.tensor(y_test, dtype=torch.long))

    y_discrete_pred = np.argmax(y_prob_predict, axis=1).flatten()
    test_acc = np.mean(y_test == y_discrete_pred)
    return {
            "test_accuracy": test_acc,
            "test_loss": test_loss,
        }
