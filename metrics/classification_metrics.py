"""
Metrics to measure classification performance
"""

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, roc_auc_score)
from torch import nn
from torch.nn import functional as F

from utils.ensemble_utils import ensemble_forward_pass


def get_logits_labels(model, data_loader, device):
    """
    Utility function to get logits and labels.
    """
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            logit = model(data)
            logits.append(logit)
            labels.append(label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    return logits, labels


def test_classification_net_softmax(softmax_prob, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)

    # print(labels_list)
    # print(predictions_list)
    corrects = ((np.array(predictions_list) == np.array(labels_list))).astype(int)

    auroc = roc_auc_score(corrects, confidence_vals_list)
    auprc = average_precision_score(corrects, confidence_vals_list)

    return (auroc, auprc), (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


def test_classification_net_logits(logits, labels):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    return test_classification_net_softmax(softmax_prob, labels)


def test_classification_net(model, data_loader, device, auc=False):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    auroc_aupr, res = test_classification_net_logits(logits, labels)
    if auc:
        return auroc_aupr, res
    else:
        return res


def test_classification_net_ensemble(model_ensemble, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset
    for a deep ensemble.
    """
    for model in model_ensemble:
        model.eval()
    softmax_prob = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            softmax, _, _ = ensemble_forward_pass(model_ensemble, data)
            softmax_prob.append(softmax)
            labels.append(label)
    softmax_prob = torch.cat(softmax_prob, dim=0)
    labels = torch.cat(labels, dim=0)

    return test_classification_net_softmax(softmax_prob, labels)


# def brier_score(pred, label):

#     print(pred)
#     print(label)

#     num_nodes = pred.shape[0]
#     if num_nodes == 0:
#         return np.nan
#     indices = np.arange(num_nodes)
#     prob = pred.copy()
#     prob[indices, label] -= 1

#     return np.mean(np.linalg.norm(prob, axis=-1, ord=2))