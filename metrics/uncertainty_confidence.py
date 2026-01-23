"""
Metrics measuring either uncertainty or confidence of a model.
"""
import torch
import torch.nn.functional as F


def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def logsumexp(logits, temperature=1.0):
    return torch.logsumexp(logits / temperature, dim=1, keepdim=False)


def energy_score(logits, temperature=1.0):
    """
    Compute energy-based uncertainty scores.
    Lower energy = higher confidence, higher energy = higher uncertainty.
    
    Args:
        logits: Raw model outputs before softmax
        temperature: Temperature parameter to scale the energy
        
    Returns:
        Energy scores for each sample
    """
    # Energy is defined as negative logsumexp of logits
    return -temperature * logsumexp(logits, temperature=temperature)


def confidence(logits):
    p = F.softmax(logits, dim=1)
    confidence, _ = torch.max(p, dim=1)
    return confidence


def margin(logits):
    p = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.sort(p, dim=1, descending=True)
    # Calculate margin between highest and second highest probability
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return margin


def entropy_prob(probs):
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def mutual_information_prob(probs):
    mean_output = torch.mean(probs, dim=0)
    predictive_entropy = entropy_prob(mean_output)

    # Computing expectation of entropies
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    exp_entropies = torch.mean(-torch.sum(plogp, dim=2), dim=0)

    # Computing mutual information
    mi = predictive_entropy - exp_entropies
    return mi
