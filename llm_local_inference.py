import argparse
import os
import json
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Model mapping for supported open-source models
MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "chat_template": True,  # Uses transformers chat template
    },
    "qwen-3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "chat_template": True,
    },
    "deepseek": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "chat_template": True,
    },
    "deepseek-chat": {
        "model_id": "deepseek-ai/deepseek-llm-7b-chat",
        "chat_template": True,
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "chat_template": True,
    },
    "llama-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "chat_template": True,
    },
}


class LocalLLMClassifier:
    def __init__(self, model_type: str = 'qwen', quantize: str = None):
        """
        Initialize local LLM classifier.
        
        Args:
            model_type: One of 'qwen', 'qwen-3b', 'deepseek', 'deepseek-chat', 'llama', 'llama-3b'
            quantize: None, '4bit', or '8bit' for quantization
        """
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type '{model_type}'. Use one of: {list(MODEL_CONFIGS.keys())}")
        
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.model_id = self.config["model_id"]
        
        print(f"Loading model: {self.model_id}")
        
        # Configure quantization if requested
        quantization_config = None
        if quantize == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantize == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        self.model.eval()
        print(f"Model loaded successfully on {self.model.device}")
        
    def create_classification_prompt(self, text: str, zeroshot: bool = True) -> str:
        """Create prompt for MITRE ATT&CK technique classification."""
        
        if zeroshot:
            prompt = f"""You are a cybersecurity analyst specializing in threat intelligence. Your task is to classify excerpts from Cyber Threat Intelligence (CTI) reports according to MITRE ATT&CK techniques.

Given a CTI report excerpt, identify the most relevant MITRE ATT&CK technique ID that corresponds to the described threat behavior or activity. 

Rules:
- Output ONLY the technique ID (e.g., T1566.001, T1055, T1003.001)
- If multiple techniques apply, output the most specific/relevant one
- If no clear technique matches, output "None"
- Do not provide explanations, justifications, or additional text

CTI Report Excerpt:
{text}

Technique ID:"""
        else:
            prompt = f"""You are a cybersecurity analyst specializing in threat intelligence. Your task is to classify excerpts from Cyber Threat Intelligence (CTI) reports according to MITRE ATT&CK techniques.

Given a CTI report excerpt, identify the most relevant MITRE ATT&CK technique ID that corresponds to the described threat behavior or activity.

Rules:
- Output ONLY the technique ID (e.g., T1566.001, T1055, T1003.001)
- If multiple techniques apply, output the most specific/relevant one
- If no clear technique matches, output "None"
- Do not provide explanations, justifications, or additional text

Here are some examples:

CTI Report Excerpt: "network traffic communicates over a raw socket."
Technique ID: T1095

CTI Report Excerpt: "has the ability to set file attributes to hidden."
Technique ID: T1564.001

CTI Report Excerpt: "searches for files that are 60mb and less and contain the following extensions: .doc .docx .xls .xlsx .ppt .pptx .exe .zip and .rar."
Technique ID: T1083

CTI Report Excerpt: "Attackers can use legitimate domains that are registered under the same CDN provider."
Technique ID: T1090

CTI Report Excerpt: "has registered two registry keys for shim databases."
Technique ID: T1112

CTI Report Excerpt: "may also deliver a weaponized Office document that executes the ReconShark reconnaissance malware."
Technique ID: T1566.001

CTI Report Excerpt: "has masqueraded as a Flash Player installer through the executable file install_flash_player.exe."
Technique ID: T1036.005

CTI Report Excerpt: "During the threat actors downloaded files and tools onto a victim machine."
Technique ID: T1105

CTI Report Excerpt: "payloads are obfuscated prior to compilation to inhibit analysis and/or reverse engineering."
Technique ID: T1027

CTI Report Excerpt: "has been executed through user installation of an executable disguised as a flash installer."
Technique ID: T1204.002

CTI Report Excerpt: "{text}"
Technique ID:"""

        return prompt
    
    def classify_single(self, text: str, classes: List[str], zeroshot: bool = True) -> str:
        """Classify a single text sample using local LLM."""
        prompt = self.create_classification_prompt(text, zeroshot)
        
        try:
            # Format as chat message if model supports it
            if self.config["chat_template"]:
                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                input_text = prompt
            
            # Tokenize
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean prediction - extract technique ID
            prediction = prediction.replace('"', '').replace("'", "").strip()
            
            # Try to extract technique ID pattern (T followed by numbers)
            import re
            match = re.search(r'T\d+(?:\.\d+)?', prediction)
            if match:
                prediction = match.group(0)
            
            # If prediction not in classes, try to match closest
            if prediction not in classes:
                prediction_lower = prediction.lower()
                for cls in classes:
                    if cls.lower() in prediction_lower or prediction_lower in cls.lower():
                        return cls
                # If no match, return first class as default
                return classes[0]
            
            return prediction
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return classes[0]  # Return first class as fallback
    
    def classify_batch(self, texts: List[str], classes: List[str], zeroshot: bool = True) -> List[str]:
        """Classify multiple texts."""
        predictions = []
        
        for text in tqdm(texts, desc="Classifying"):
            pred = self.classify_single(text, classes, zeroshot)
            predictions.append(pred)
        
        return predictions


def compute_classification_metrics(y_true: np.array, y_pred: np.array,
                                 class_names: List[str] = None) -> Dict:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels (integers)
        y_pred: Predicted labels (integers)
        class_names: List of class names
    
    Returns:
        Dictionary with all metrics
    """
    
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }
    
    return metrics


def main(args):
    with open("./data/cti/tram.json") as f:
        data_json = json.loads(f.read())

    df = pd.DataFrame(
        [
            {'text': row['text'], 'label': row['mappings'][0]['attack_id']}
            for row in data_json['sentences']
            if len(row['mappings']) > 0
        ]
    )

    # Load splits
    with open("./saves/splits_seed_1.json", 'r') as f:
        splits = json.load(f)

    df = df.iloc[splits['test_idx']][['text', 'label']]

    print(f"Test samples: {len(df)}")

    classes = df['label'].unique().tolist()
    print(f"Number of classes: {len(classes)}")

    # Initialize classifier
    classifier = LocalLLMClassifier(args.model, quantize=args.quantize)

    fshot = "zero-shot" if args.zeroshot else "few-shot"
    print(f"Starting {fshot} Classification with {args.model}...")

    # Get predictions
    predictions = classifier.classify_batch(df['text'].tolist(), classes, args.zeroshot)

    # Convert string labels to integers for metric computation
    label_to_int = {label: i for i, label in enumerate(classes)}

    y_true = np.array([label_to_int[label] for label in df['label']])
    y_pred = np.array([label_to_int[pred] for pred in predictions])

    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred, classes)

    print(f"\n" + "="*30)
    print("PERFORMANCE METRICS")
    print("="*30)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"F1 Macro:     {metrics['f1_macro']:.4f}")
    print(f"F1 Micro:     {metrics['f1_micro']:.4f}")

    accuracy_file_name = f"results/{args.model}.json"

    os.makedirs("results", exist_ok=True)
    with open(accuracy_file_name, "w") as acc_file:
        json.dump(metrics, acc_file)

    return metrics, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local LLM inference for CTI classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen", 
        choices=list(MODEL_CONFIGS.keys()),
        help="Local LLM to use"
    )
    parser.add_argument(
        "--zeroshot", 
        action="store_true", 
        default=False, 
        help="Use zero-shot (True) or few-shot (False) prompting"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["4bit", "8bit"],
        help="Quantization level to reduce memory usage"
    )

    args = parser.parse_args()

    metrics, predictions = main(args)
