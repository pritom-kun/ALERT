from transformers import AutoModelForSequenceClassification
from torch import nn


class BERT(nn.Module):
    def __init__(self, tokenizer, model_name, num_classes=5):
        super(BERT, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=True,
    )
        self.tokenizer = tokenizer
        self.feature = None

    def forward(self, x):
        out = self.model(x, attention_mask=x.ne(self.tokenizer.pad_token_id).to(int))

        self.feature = out.hidden_states[-1][:, 0, :].clone().detach()
        return out.logits


def scibert(tokenizer, num_classes=5):
    return BERT(tokenizer, "allenai/scibert_scivocab_uncased", num_classes)

def roberta(tokenizer, num_classes=5):
    return BERT(tokenizer, "roberta-base", num_classes)

def modernbert(tokenizer, num_classes=5):
    return BERT(tokenizer, "answerdotai/ModernBERT-base", num_classes)

