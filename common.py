import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def get_perf(y_prob_predict, y_test):
    y_test = y_test.flatten()
    loss_fn = torch.nn.CrossEntropyLoss()
    y_log_prob = np.log(y_prob_predict)
    #test_loss = (y_log_prob), torch.tensor(y_test, dtype=torch.long))
    print(y_log_prob.shape)
    is_binary = np.max(y_test) == 1
    assert is_binary

    print(y_test)
    print(y_prob_predict[:,-1])

    auc = roc_auc_score(y_test, y_prob_predict[:,-1]) if is_binary else None

    if y_log_prob.shape[1] == 1 and is_binary:
        # If there is a single output and it is a binary task, then interpret as probability of class 1
        log_prob_class = np.log(y_prob_predict * y_test + (1 - y_prob_predict) * (1 - y_test))

        y_discrete_pred = np.array((y_prob_predict > 0.5).flatten(), dtype=int)
        test_acc = np.mean(y_test == y_discrete_pred)
    else:
        log_prob_class = np.array(
            [y_log_prob[i, y_test[i]] for i in range(y_test.shape[0])]
        ).flatten()

        y_discrete_pred = np.argmax(y_prob_predict, axis=1).flatten()
        print(y_test)
        print(y_discrete_pred)
        test_acc = np.mean(y_test == y_discrete_pred)
    print(y_prob_predict[:10])
    print(y_test[:10])

    test_nll = -np.mean(log_prob_class)
    return {
            "test_accuracy": float(test_acc),
            "test_auc": float(auc),
            "test_loss": float(test_nll),
        }
