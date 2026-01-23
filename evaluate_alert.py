import torch
from torch.utils import data

from data.tram import create_tram_dataset
from metrics.classification_metrics import test_classification_net
# Import network architectures
from net.bert import scibert, roberta, modernbert

if __name__ == "__main__":

    seed = 1
    al_type = "margin"  # "confidence", "entropy", "energy", "margin"
    train_data_len = 600
    model_name = "scibert"  # "scibert", "roberta", "modernbert"

    tokenizer_name = ""
    model_fn = None

    if model_name == "scibert":
        model_fn = scibert
        tokenizer_name = "allenai/scibert_scivocab_uncased"
    elif model_name == "roberta":
        model_fn = roberta
        tokenizer_name = "roberta-base"
    elif model_name == "modernbert":
        model_fn = modernbert
        tokenizer_name = "answerdotai/ModernBERT-base"


    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    num_classes = 50

    train_dataset, test_dataset, tokenizer = create_tram_dataset(
        data_path="./data/cti/tram.json",
        tokenizer_name=tokenizer_name,
        seed=seed,
        transform=False
    )

    kwargs = {"num_workers": 0, "pin_memory": False} if cuda else {}
    test_loader = data.DataLoader(test_dataset,
        batch_size=16,
        shuffle=False,
        **kwargs
    )

    model = model_fn(tokenizer, num_classes).to(device=device)
    model.load_state_dict(torch.load(f"checkpoints/al_type_{al_type}_{train_data_len}_samples.pt"))

    print("Testing the model: Softmax/GMM======================================>")
    (accuracy, f1_micro, f1_macro, auroc, aupr) = test_classification_net(
        model, test_loader, device=device, auc_roc=True
    )

    print(f"Test set: Accuracy: {100.0 * accuracy:.2f}%")
    print(f"Test set: F1 (micro): {100.0 * f1_micro:.2f}%")
    print(f"Test set: F1 (macro): {100.0 * f1_macro:.2f}%")
    print(f"Test set: AUROC: {100.0 * auroc:.2f}%")
    print(f"Test set: AUPR: {100.0 * aupr:.2f}%")
    print(f"Training samples: {train_data_len}")
