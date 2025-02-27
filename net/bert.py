import transformers
from torch import nn


class SciBERT(nn.Module):
    def __init__(self, tokenizer, num_classes=5):
        super(SciBERT, self).__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased",
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    )

        self.tokenizer = tokenizer

        self.feature = None

    def forward(self, x):
        out = self.model(x, attention_mask=x.ne(self.tokenizer.pad_token_id).to(int))

        self.feature = out.logits.clone().detach()
        return out.logits


def scibert(tokenizer, num_classes=5):

    model = SciBERT(tokenizer, num_classes)

    return model
