import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.backends import cudnn

# Import data utilities
from torch.utils import data
from data.active_learning import active_learning
from data.ambiguous_mnist.ambiguous_mnist_dataset import AmbiguousMNIST
from data.tram import create_tram_dataset
from data.cti2mitre import create_cti2mitre_dataset

# Import network architectures
from net.bert import scibert, roberta, modernbert

# Import train and test utils
from utils.train_utils import train_single_epoch, train_single_epoch_aug, model_save_name

# Importing uncertainty metrics
from metrics.uncertainty_confidence import entropy, energy_score, margin, confidence
from metrics.classification_metrics import test_classification_net
from metrics.classification_metrics import test_classification_net_ensemble

# Importing args
from utils.args import al_args

# Importing GMM utilities
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.dropout_utils import train_dropout
from utils.ensemble_utils import ensemble_forward_pass


def class_probs(data_loader):
    class_n = len(data_loader.dataset)
    class_count = torch.zeros(args.num_classes)
    for data, label in data_loader:
        class_count += torch.Tensor([torch.sum(label == c) for c in range(args.num_classes)])

    class_prob = class_count / class_n
    return class_prob


def compute_density(logits, class_probs):
    return torch.sum((torch.exp(logits) * class_probs), dim=1)


def augmentation_collate(batch, use_augmentation=False):
    if use_augmentation:
        # Extract all components from the batch
        features, aug_ins, aug_sub, targets = zip(*batch)
        # Convert to tensors if needed
        return torch.stack(features), torch.stack(aug_ins), torch.stack(aug_sub), torch.stack(targets)
    else:
        # For non-augmented data, extract only features and targets
        features, targets = zip(*[(item[0], item[-1]) for item in batch])
        return torch.stack(features), torch.stack(targets)


def ambiguous_acquired(data_loader, threshold, model):
    """
    This method is required to identify the ambiguous samples which are acquired.
    """
    model.eval()
    logits = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            op = model(data)
            logits.append(op)

        logits = torch.cat(logits, dim=0)
    entropies = entropy(logits)

    return entropies.cpu().numpy().tolist(), (torch.sum(entropies > threshold).item() / len(data_loader.dataset))


if __name__ == "__main__":

    args = al_args().parse_args()
    print(args)

    # Checking if GPU is available
    cuda = torch.cuda.is_available()

    # Setting additional parameters
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    tokenizer_name = ""
    model_fn = None

    if args.model_name == "scibert":
        model_fn = scibert
        tokenizer_name = "allenai/scibert_scivocab_uncased"
    elif args.model_name == "roberta":
        model_fn = roberta
        tokenizer_name = "roberta-base"
    elif args.model_name == "modernbert":
        model_fn = modernbert
        tokenizer_name = "answerdotai/ModernBERT-base"

    train_dataset = None
    test_dataset = None
    tokenizer = None
    # Creating the datasets
    if args.dataset == "tram":
        train_dataset, test_dataset, tokenizer = create_tram_dataset(
            data_path="./data/cti/tram.json",
            tokenizer_name=tokenizer_name,
            seed=args.seed,
            transform=args.data_aug
        )

    elif args.dataset == "cti2mitre":
        train_dataset, test_dataset, tokenizer = create_cti2mitre_dataset(
            data_path="./data/cti/cti2mitre.csv",
            tokenizer_name=tokenizer_name,
            seed=args.seed,
            transform=args.data_aug
        )
    else:
        raise ValueError("Unknown dataset")

    # train_dataset, test_dataset, tokenizer = create_cti_hal_dataset(
    #     data_path="./data/CTI-HAL/data/",
    #     seed=args.seed,
    #     transform=args.data_aug
    # )

    if args.ambiguous:
        indices = np.random.choice(len(train_dataset), args.subsample)
        mnist_train_dataset = data.Subset(train_dataset, indices)
        train_dataset = data.ConcatDataset(
            [mnist_train_dataset, AmbiguousMNIST(root=args.dataset_root, train=True, device=device),]
        )

    # Load pretrained network for checking ambiguous samples
    if args.ambiguous:
        pretrained_net = model_fn(tokenizer, args.num_classes).to(device)
        pretrained_net = torch.nn.DataParallel(pretrained_net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        pretrained_net.load_state_dict(torch.load(args.saved_model_path + args.saved_model_name))


    # Creating a validation split
    idxs = list(range(len(train_dataset)))
    split = int(np.floor(0.1 * len(train_dataset)))
    np.random.seed(args.seed)
    np.random.shuffle(idxs)

    train_idx, val_idx = idxs[split:], idxs[:split]
    val_dataset = data.Subset(train_dataset, val_idx)
    # train_dataset = data.Subset(train_dataset, train_idx)

    initial_sample_indices = active_learning.get_balanced_sample_indices(
        train_dataset, num_classes=args.num_classes, n_per_digit=args.num_initial_samples / args.num_classes,
    )

    kwargs = {"num_workers": 0, "pin_memory": False} if cuda else {}
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda batch: augmentation_collate(batch, use_augmentation=False),
        **kwargs
    )
    test_loader = data.DataLoader(test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda batch: augmentation_collate(batch, use_augmentation=False),
        **kwargs
    )

    # Run experiment
    num_runs = 1
    test_accs = {}
    ambiguous_dict = {}
    ambiguous_entropies_dict = {}

    for i in range(num_runs):
        test_accs[i] = []
        ambiguous_dict[i] = []
        ambiguous_entropies_dict[i] = {}

    for run in range(num_runs):
        print("Experiment run: " + str(run) + " =====================================================================>")

        torch.manual_seed(args.seed + run)

        # Setup data for the experiment
        # Split off the initial samples first
        active_learning_data = active_learning.ActiveLearningData(train_dataset)

        # Acquiring the first training dataset from the total pool. This is random acquisition
        active_learning_data.acquire(initial_sample_indices)

        # Train loader for the current acquired training set
        sampler = active_learning.RandomFixedLengthSampler(
            dataset=active_learning_data.training_dataset, target_length=5056
        )
        train_loader = data.DataLoader(
            active_learning_data.training_dataset,
            batch_size=args.train_batch_size,
             collate_fn=lambda batch: augmentation_collate(batch, use_augmentation=args.data_aug),
            **kwargs,
        )
        small_train_loader = data.DataLoader(
            active_learning_data.training_dataset,
            shuffle=True,
            batch_size=args.train_batch_size,
            collate_fn=lambda batch: augmentation_collate(batch, use_augmentation=False),
            **kwargs,
        )
        # Pool loader for the current acquired training set
        pool_loader = data.DataLoader(
            active_learning_data.pool_dataset,
            batch_size=args.scoring_batch_size,
            shuffle=False,
            collate_fn=lambda batch: augmentation_collate(batch, use_augmentation=False),
            **kwargs,
        )

        # Run active learning iterations
        active_learning_iteration = 0
        while True:
            print("Active Learning Iteration: " + str(active_learning_iteration) + " ================================>")

            lr = 2e-5
            weight_decay = 5e-4
            eps = 1e-8
            if args.al_type == "ensemble":
                model_ensemble = [
                    model_fn(tokenizer, args.num_classes).to(device=device)
                    for _ in range(args.num_ensemble)
                ]
                optimizers = []
                for model in model_ensemble:
                    optimizers.append(torch.optim.Adam(model.parameters(), weight_decay=weight_decay))
                    model.train()
            else:
                model = model_fn(tokenizer, args.num_classes).to(device=device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps)
                model.train()

            # Train
            print("Length of train dataset: " + str(len(train_loader.dataset)))
            best_model = None
            best_val_accuracy = 0
            for epoch in range(args.epochs):
                if args.al_type == "ensemble":
                    for (model, optimizer) in zip(model_ensemble, optimizers):
                        train_single_epoch(epoch, model, train_loader, optimizer, device)
                elif args.data_aug:
                    train_single_epoch_aug(epoch, model, train_loader, optimizer, device)
                else:
                    train_single_epoch(epoch, model, train_loader, optimizer, device)

                val_accuracy, _, _, _, _ = (
                    test_classification_net_ensemble(model_ensemble, val_loader, device=device, auc_roc=True)
                    if args.al_type == "ensemble"
                    else test_classification_net(model, val_loader, device=device, auc_roc=True)
                )
                if val_accuracy >= best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = model_ensemble if args.al_type == "ensemble" else model

            if args.al_type == "ensemble":
                model_ensemble = best_model
            else:
                model = best_model

            if args.al_type == "gmm":
                # Fit the GMM on the trained model
                model.eval()
                embeddings, labels = get_embeddings(
                    model, small_train_loader, num_dim=768, dtype=torch.double, device=device, storage_device="cuda",
                )
                gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=args.num_classes)

            elif args.al_type == "dropout":
                # Train the dropout head
                model.eval()
                embeddings, labels = get_embeddings(
                    model, small_train_loader, num_dim=768, dtype=torch.float32, device=device, storage_device="cuda",
                )
                dropout_head = train_dropout(embeddings, labels, args.num_classes, args.train_batch_size, epochs=10, device=device)

            print("Training ended")

            # Testing the models
            if args.al_type == "ensemble":
                print("Testing the model: Ensemble======================================>")
                for model in model_ensemble:
                    model.eval()
                (accuracy, f1_micro, f1_macro, auroc, aupr) = test_classification_net_ensemble(
                    model_ensemble, test_loader, device=device, auc_roc=True
                )

            else:
                print("Testing the model: Softmax/GMM======================================>")
                (accuracy, f1_micro, f1_macro, auroc, aupr) = test_classification_net(
                    model, test_loader, device=device, auc_roc=True
                )

            test_accs[run].append({
                'accuracy': 100.0 * accuracy,
                'f1_micro': 100.0 * f1_micro,
                'f1_macro': 100.0 * f1_macro,
                'auroc': 100.0 * auroc,
                'aupr': 100.0 * aupr,
                'training_samples': len(active_learning_data.training_dataset)
            })

            print(f"Test set: Accuracy: ({100.0 * accuracy:.2f}%)")
            print(f"Test set: F1 (micro): ({100.0 * f1_micro:.2f}%)")
            print(f"Test set: F1 (macro): ({100.0 * f1_macro:.2f}%)")
            print(f"Test set: AUROC: ({100.0 * auroc:.2f}%)")
            print(f"Test set: AUPR: ({100.0 * aupr:.2f}%)")
            print(f"Training samples: {len(active_learning_data.training_dataset)}")

            # Save model at specific training sample counts
            save_checkpoints = [600, 1100, 1600, 2100]
            curr_train_len = len(active_learning_data.training_dataset)

            if curr_train_len in save_checkpoints:
                os.makedirs("checkpoints", exist_ok=True)
                model_save_path = f"checkpoints/al_type_{args.al_type}_{curr_train_len}_samples.pt"
                if args.al_type == "ensemble":
                    torch.save([m.state_dict() for m in model_ensemble], model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)
                print(f"Saved model at {curr_train_len} training samples to {model_save_path}")

            # Breaking clause
            if len(active_learning_data.training_dataset) >= args.max_training_samples:
                break

            # Acquisition phase
            N = len(active_learning_data.pool_dataset)

            print("Performing acquisition ========================================")
            if args.al_type == "ensemble":
                for model in model_ensemble:
                    model.eval()
                ensemble_uncs = []
                with torch.no_grad():
                    for data, _ in pool_loader:
                        data = data.to(device)
                        mean_output, predictive_entropy, mi = ensemble_forward_pass(model_ensemble, data)

                        ensemble_uncs.append(mi if args.mi else predictive_entropy)
                    ensemble_uncs = torch.cat(ensemble_uncs, dim=0)

                    (candidate_scores, candidate_indices,) = active_learning.get_top_k_scorers(
                        ensemble_uncs, args.acquisition_batch_size
                    )
            elif args.al_type == "gmm":
                model.eval()
                class_prob = class_probs(train_loader)
                logits, labels = gmm_evaluate(
                    model,
                    gaussians_model,
                    pool_loader,
                    device=device,
                    num_classes=args.num_classes,
                    storage_device="cpu",
                )
                (candidate_scores, candidate_indices,) = active_learning.get_top_k_scorers(
                    compute_density(logits, class_prob), args.acquisition_batch_size, uncertainty="gmm",
                )
            elif args.al_type == "dropout":
                model.eval()
                dropout_head.eval()
                dropout_head.dropout.train()

                embeddings, _ = get_embeddings(
                    model, pool_loader, num_dim=768, dtype=torch.float32, device=device, storage_device="cuda",
                )

                all_probs = []
                for _ in range(args.mc_dropout_passes):
                    with torch.no_grad():
                        logits = dropout_head(embeddings)
                        probs = F.softmax(logits, dim=1)
                        all_probs.append(probs)

                p = torch.stack(all_probs).mean(0)
                entropy_values = -torch.sum(p * torch.log(p + 1e-10), dim=1)

                (candidate_scores, candidate_indices,) = active_learning.get_top_k_scorers(
                    entropy_values, args.acquisition_batch_size, uncertainty="entropy",
                )

            elif args.al_type == "coreset":
                # Get embeddings for the labeled data
                model.eval()
                labeled_embeddings, _ = get_embeddings(
                    model, small_train_loader, num_dim=768, dtype=torch.double, device=device, storage_device="cpu",
                )

                # Get embeddings for the unlabeled data
                unlabeled_embeddings, _ = get_embeddings(
                    model, pool_loader, num_dim=768, dtype=torch.double, device=device, storage_device="cpu",
                )

                # Use the coreset method to select samples
                candidate_scores, candidate_indices = active_learning.greedy_coreset_selection(
                    unlabeled_embeddings, labeled_embeddings, args.acquisition_batch_size
                )
            elif args.al_type == "random":
                candidate_indices = np.random.choice(N, args.acquisition_batch_size, replace=False)
                candidate_scores = torch.zeros(args.acquisition_batch_size).to(device)
            else:
                model.eval()

                if args.al_type == "entropy":
                    score_function = entropy
                    uncertainty = "entropy"
                elif args.al_type == "energy":
                    score_function = energy_score
                    uncertainty = "energy"
                elif args.al_type == "confidence":
                    score_function = confidence
                    uncertainty = "confidence"
                elif args.al_type == "margin":
                    score_function = margin
                    uncertainty = "margin"
                else:
                    raise ValueError("Unknown acquisition function")

                logits = []
                with torch.no_grad():
                    for data, _ in pool_loader:
                        data = data.to(device)
                        logits.append(model(data))
                    logits = torch.cat(logits, dim=0)
                (candidate_scores, candidate_indices,) = active_learning.find_acquisition_batch(
                    logits, args.acquisition_batch_size, score_function, uncertainty
                )

            # Performing acquisition
            active_learning_data.acquire(candidate_indices)
            if args.ambiguous:
                entropies, amb_percent = ambiguous_acquired(small_train_loader, args.threshold, pretrained_net)
                ambiguous_dict[run].append(amb_percent)
                ambiguous_entropies_dict[run][active_learning_iteration] = entropies
            active_learning_iteration += 1

    # Save the dictionaries
    save_name = model_save_name(args.model_name, args.sn, args.mod, args.coeff, args.seed)
    save_ensemble_mi = "_mi" if (args.al_type == "ensemble" and args.mi) else ""

    os.makedirs("results", exist_ok=True)
    if args.ambiguous:
        accuracy_file_name = f"results/test_accs_{save_name}_{args.al_type}{save_ensemble_mi}_dirty_mnist_{args.subsample}.json"
        ambiguous_file_name = f"results/ambiguous_{save_name}_{args.al_type}{save_ensemble_mi}_dirty_mnist_{args.subsample}.json"
        ambiguous_entropies_file_name = f"results/ambiguous_entropies_{save_name}_{args.al_type}{save_ensemble_mi}_dirty_mnist_{args.subsample}.json"
    else:
        accuracy_file_name = f"results/metrics_{save_name}_{args.al_type}{save_ensemble_mi}_{args.dataset}.json"

    with open(accuracy_file_name, "w") as acc_file:
        json.dump(test_accs, acc_file)

    if args.ambiguous:
        with open(ambiguous_file_name, "w") as ambiguous_file:
            json.dump(ambiguous_dict, ambiguous_file)
        with open(ambiguous_entropies_file_name, "w") as ambiguous_entropies_file:
            json.dump(ambiguous_entropies_dict, ambiguous_entropies_file)
