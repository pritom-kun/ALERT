import numpy as np
import warnings
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
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
    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


def test_classification_net_auc_roc(softmax_prob, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    confidence_vals, predictions = torch.max(softmax_prob, dim=1)

    softmax_prob_np = softmax_prob.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    accuracy = accuracy_score(labels_np, predictions_np)
    f1_micro = f1_score(labels_np, predictions_np, average="micro")
    f1_macro = f1_score(labels_np, predictions_np, average="macro")

    num_classes = softmax_prob_np.shape[1]
    # Binarize true labels for one-vs-rest AUPR
    labels_binarized = label_binarize(labels_np, classes=np.arange(num_classes))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress ROC/AUPR warnings
        try:
            auroc = roc_auc_score(labels_np, softmax_prob_np, multi_class="ovr", labels=np.arange(num_classes))
        except ValueError:
            auroc = float("nan")

        try:
            aupr = average_precision_score(labels_binarized, softmax_prob_np, average="macro")
        except ValueError:
            aupr = float("nan")

    return (
        confusion_matrix(labels_np, predictions_np),
        accuracy,
        f1_micro,
        f1_macro,
        auroc,
        aupr
    )


def test_classification_net_logits(logits, labels, auc_roc=False):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    if auc_roc:
        return test_classification_net_auc_roc(softmax_prob, labels)
    else:
        return test_classification_net_softmax(softmax_prob, labels)


def test_classification_net(model, data_loader, device, auc_roc=False):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits(logits, labels, auc_roc)


def test_classification_net_ensemble(model_ensemble, data_loader, device, auc_roc=False):
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

    if auc_roc:
        return test_classification_net_auc_roc(softmax_prob, labels)
    else:
        return test_classification_net_softmax(softmax_prob, labels)
