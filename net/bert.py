import transformers
from torch import nn


class SciBERT(nn.Module):
    def __init__(self, tokenizer, num_classes=5):
        super(SciBERT, self).__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased",
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

    return SciBERT(tokenizer, num_classes)

