import argparse
import json
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from openai import OpenAI
from perplexity import Perplexity
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

random.seed(1)
np.random.seed(1)


class LLMClassifier:
    def __init__(self, model_type: str = 'gpt', api_key: str = None):

        self.model_type = model_type

        if self.model_type == "gpt":
            self.client = OpenAI(api_key=api_key)
        elif self.model_type == "gemini":
            self.client = genai.Client(api_key=api_key)
        elif self.model_type == "perplexity":
            self.client = Perplexity(api_key=api_key)
        else:
            raise ValueError("Unsupported model type. Use one of 'gpt', 'gemini' or claude.")

        self.class_mapping = {}
        
    def create_classification_prompt(self, text: str, zeroshot: bool = True) -> str:

        if zeroshot:
            prompt = f"""
            You are a cybersecurity analyst specializing in threat intelligence. Your task is to classify excerpts from Cyber Threat Intelligence (CTI) reports according to MITRE ATT&CK techniques.

            Given a CTI report excerpt, identify the most relevant MITRE ATT&CK technique ID that corresponds to the described threat behavior or activity. 

            Rules:
            - Output ONLY the technique ID (e.g., T1566.001, T1055, T1003.001)
            - If multiple techniques apply, output the most specific/relevant one
            - If no clear technique matches, output "None"
            - Do not provide explanations, justifications, or additional text

            CTI Report Excerpt:
            {text}

            Technique ID:
            """
        else:
            prompt = f"""
            You are a cybersecurity analyst specializing in threat intelligence. Your task is to classify excerpts from Cyber Threat Intelligence (CTI) reports according to MITRE ATT&CK techniques.

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

            CTI Report Excerpt: "searches for files that are 60mb and less and contain the following extensions: .doc .docx .xls .xlsx .ppt .pptx .exe .zip and .rar.  also runs the echo %APPDATA% command to list the contents of the directory.(Citation: Securelist Sofacy Feb 2018)(Citation: ESET Zebrocy Nov 2018)(Citation: ESET Zebrocy May 2019)  can obtain the current execution path as well as perform drive enumeration.(Citation: Accenture SNAKEMACKEREL Nov 2018) "
            Technique ID: T1083

            CTI Report Excerpt: "Attackers can use legitimate domains that are registered under the same CDN provider."
            Technique ID: T1090

            CTI Report Excerpt: "has registered two registry keys for shim databases."
            Technique ID: T1112

            CTI Report Excerpt: "may also deliver a weaponized Office document that executes the ReconShark reconnaissance malware."
            Technique ID: T1566.001

            CTI Report Excerpt: "has masqueraded as a Flash Player installer through the executable file install_flash_player.exe.(Citation: ESET Bad Rabbit)"
            Technique ID: T1036.005

            CTI Report Excerpt: "During  the threat actors downloaded files and tools onto a victim machine."
            Technique ID: T1105

            CTI Report Excerpt: "payloads are obfuscated prior to compilation to inhibit analysis and/or reverse engineering.(Citation: SecureList SynAck Doppelgnging May 2018)"
            Technique ID: T1027

            CTI Report Excerpt: "has been executed through user installation of an executable disguised as a flash installer.(Citation: ESET Bad Rabbit)"
            Technique ID: T1204.002           

            CTI Report Excerpt: "{text}"
            Technique ID:
            """

        return prompt
    
    def classify_single(self, text: str, classes: List[str], zeroshot: bool = True) -> str:
        """Classify a single text sample using GPT-5"""
        prompt = self.create_classification_prompt(text, zeroshot)

        try:
            if self.model_type == 'gpt':
                response = self.client.responses.create(
                    model="gpt-5",  # Use mini for faster classification
                    input=prompt,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "low"}  # Concise responses
                )

                prediction = response.output_text.strip()

            elif self.model_type == 'gemini':
                grounding_tool = types.Tool(
                    google_search=types.GoogleSearch()
                )
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        # temperature=0.1,  # Low temperature for deterministic output
                        # # maxOutputTokens=50,
                        # top_p=0.9,
                        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                        tools=[grounding_tool]
                    ),
                )

                prediction = response.text.strip()

            elif self.model_type == 'perplexity':
                # Perplexity API implementation
                response = self.client.chat.completions.create(
                    model="sonar",  # Fast reasoning model for classification
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0.1,  # Low temperature for deterministic output
                    # max_tokens=50,    # Short response for classification
                    # top_p=0.9
                )
                prediction = response.choices[0].message.content.strip()

            # Clean and validate prediction
            prediction = prediction.replace('"', '').replace("'", "").strip()

            # If prediction not in classes, try to match closest
            if prediction not in classes:
                prediction_lower = prediction.lower()
                for cls in classes:
                    if cls.lower() in prediction_lower or prediction_lower in cls.lower():
                        return cls
                # If no match, return first class as default
                return classes[0]

            # print(prediction)
            return prediction

        except Exception as e:
            print(f"Error in classification: {e}")
            return classes[0]  # Return first class as fallback

    def classify_batch(self, texts: List[str], classes: List[str], zeroshot: bool = True) -> List[str]:
        """Classify multiple texts"""
        predictions = []

        for i, text in tqdm(enumerate(texts)):
            # if i % 10 == 0:
            #     print(f"Processing {i+1}/{len(texts)}")

            pred = self.classify_single(text, classes, zeroshot)
            predictions.append(pred)

            # Small delay to avoid rate limits
            # if i % 50 == 0: 
            time.sleep(0.1)

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

# Example usage with sample data
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

    print(len(df))
    # df = df.head(10)

    classes = df['label'].unique().tolist()
    print(len(classes))

    # Initialize classifier
    classifier = LLMClassifier(args.model)

    fshot = "zero-shot" if args.zeroshot else "few-shot"
    print(f"Starting {fshot} Classification...")

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

    with open(accuracy_file_name, "w") as acc_file:
        json.dump(metrics, acc_file)

    return metrics, predictions

if __name__ == "__main__":
    # Run the example

    parser = argparse.ArgumentParser(
            description="Args for training parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--model", type=str, default="gpt", dest="model", help="llm to use",
    )

    parser.add_argument(
        "--zeroshot", action="store_true", default=False, dest="zeroshot", help="zero shot or few shot",
    )

    args = parser.parse_args()

    metrics, predictions = main(args)
