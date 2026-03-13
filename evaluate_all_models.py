import torch
from torch.utils import data
import pandas as pd
import sys

from data.tram import create_tram_dataset
from metrics.classification_metrics import test_classification_net
# Import network architectures
from net.bert import scibert, roberta, modernbert

if __name__ == "__main__":
    
    seed = 1
    model_name = "scibert"  # "scibert", "roberta", "modernbert"
    
    # Define all AL types and train data lengths to evaluate
    al_types = ["confidence", "coreset", "dropout", "energy", "entropy", "gmm", "margin", "random"]
    train_data_lengths = [600, 1100, 1600, 2100]
    
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

    # Create dataset (only once)
    print("Loading dataset...")
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

    # Store results
    results = []

    # Evaluate all combinations
    total_evaluations = len(al_types) * len(train_data_lengths)
    current_eval = 0
    
    for al_type in al_types:
        for train_data_len in train_data_lengths:
            current_eval += 1
            checkpoint_path = f"checkpoints/al_type_{al_type}_{train_data_len}_samples.pt"
            
            print(f"\n[{current_eval}/{total_evaluations}] Evaluating: AL Type={al_type}, Train Samples={train_data_len}")
            print(f"Loading checkpoint: {checkpoint_path}")
            
            try:
                # Load model
                model = model_fn(tokenizer, num_classes).to(device=device)
                model.load_state_dict(torch.load(checkpoint_path))
                
                # Evaluate
                print("Testing the model...")
                (accuracy, f1_micro, f1_macro, auroc, aupr) = test_classification_net(
                    model, test_loader, device=device, auc_roc=True
                )
                
                # Print results
                print(f"  Accuracy: {100.0 * accuracy:.2f}%")
                print(f"  F1 (macro): {100.0 * f1_macro:.2f}%")
                print(f"  AUROC: {100.0 * auroc:.2f}%")
                print(f"  AUPR: {100.0 * aupr:.2f}%")
                
                # Store results
                results.append({
                    'al_type': al_type,
                    'train_samples': train_data_len,
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'auroc': auroc,
                    'aupr': aupr
                })
                
            except Exception as e:
                print(f"  ERROR: Failed to evaluate {checkpoint_path}")
                print(f"  {str(e)}")
                # Add NaN results for failed evaluations
                results.append({
                    'al_type': al_type,
                    'train_samples': train_data_len,
                    'accuracy': float('nan'),
                    'f1_macro': float('nan'),
                    'auroc': float('nan'),
                    'aupr': float('nan')
                })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    output_file = "./results/evaluation_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to: {output_file}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(df.to_string(index=False))
