import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import functional as F
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score

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


def test_classification_net_auc_roc(softmax_prob, labels, device):
    """
    Calculate classification metrics using TorchMetrics.
    
    Args:
        softmax_prob (torch.Tensor): Softmax probabilities from model
        labels (torch.Tensor): Ground truth labels
        num_classes (int): Number of classes
        
    Returns:
        tuple: (accuracy, auroc, average_precision, f1_micro, f1_macro)
    """

    num_classes = softmax_prob.shape[-1]
    # Initialize metrics
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=num_classes).to(device)
    aupr_metric = AveragePrecision(task="multiclass", num_classes=num_classes).to(device)
    f1_micro_metric = F1Score(task="multiclass", num_classes=num_classes, average="micro").to(device)
    f1_macro_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    # Calculate metrics
    accuracy = accuracy_metric(softmax_prob, labels)
    auroc = auroc_metric(softmax_prob, labels)
    aupr = aupr_metric(softmax_prob, labels)
    f1_micro = f1_micro_metric(softmax_prob, labels)
    f1_macro = f1_macro_metric(softmax_prob, labels)

    return accuracy.item(), f1_micro.item(), f1_macro.item(), auroc.item(), aupr.item()


def test_classification_net_logits(logits, labels, device='cpu', auc_roc=False):
    """
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    """
    softmax_prob = F.softmax(logits, dim=1)
    if auc_roc:
        return test_classification_net_auc_roc(softmax_prob, labels, device)
    else:
        return test_classification_net_softmax(softmax_prob, labels)


def test_classification_net(model, data_loader, device, auc_roc=False):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    logits, labels = get_logits_labels(model, data_loader, device)
    return test_classification_net_logits(logits, labels, device, auc_roc)


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
