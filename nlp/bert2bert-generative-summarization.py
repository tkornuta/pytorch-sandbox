# SPDX-License-Identifier: Apache-2.0
#
# Finetunes a transformer-based seq2seq model on generative task: ccdv/pubmed-summarization.
# Losely based on:
# https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_Dailymail.ipynb#scrollTo=JD2jv3GkyjR-

import os
import torch
import datasets
from datasets import load_from_disk

from transformers import EncoderDecoderModel
from transformers import BertTokenizer, BertTokenizerFast

from pytorch_lightning import Trainer, LightningModule


# Sequences lengths.
encoder_max_length=256 # 512
decoder_max_length=128

# Model.
class Bert2Bert(LightningModule):
    def __init__(self):
        super().__init__()
        # Model - load pretrained BERT-based encoder-decoder.
        self.bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

        # Set special tokens.
        self.bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
        self.bert2bert.config.eos_token_id = tokenizer.eos_token_id
        self.bert2bert.config.pad_token_id = tokenizer.pad_token_id

        # Sensible parameters for beam search.
        self.bert2bert.config.vocab_size = self.bert2bert.config.decoder.vocab_size
        self.bert2bert.config.max_length = 142
        self.bert2bert.config.min_length = 56
        self.bert2bert.config.no_repeat_ngram_size = 3
        self.bert2bert.config.early_stopping = True
        self.bert2bert.config.length_penalty = 2.0
        self.bert2bert.config.num_beams = 4

    #def forward(self, x):
    #    return self.bert2bert(input_ids=x["input_ids"])

    def training_step(self, batch, batch_idx):

        # Get predictions.
        preds = self.bert2bert(input_ids=batch["input_ids"], decoder_input_ids=batch['decoder_input_ids'])

        # Calculate masked cross entropy - non-contributing targer sequence elements in `labels` are already set to ignore_index (-100).
        loss = torch.nn.functional.cross_entropy(preds["logits"].view(-1, preds["logits"].size(-1)), batch["labels"].view(-1))
        
        return loss

    def validation_step(self, batch, batch_idx):

        # Get predictions.
        preds = self.bert2bert(input_ids=batch["input_ids"], decoder_input_ids=batch['decoder_input_ids'])

        # Calculate masked cross entropy - non-contributing targer sequence elements in `labels` are already set to ignore_index (-100).
        loss = torch.nn.functional.cross_entropy(preds["logits"].view(-1, preds["logits"].size(-1)), batch["labels"].view(-1))
        
        return loss

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=0.001)


# Preprocessing - tokenization + padding.
def process_data_to_model_inputs(batch):

    # tokenize the inputs and labels
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    outputs = tokenizer(batch["abstract"], padding="max_length", truncation=True, max_length=decoder_max_length)
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

def get_dataset_split(split, local_path):
    if not os.path.exists(local_path):
        # Load data.
        dataset = datasets.load_dataset("ccdv/pubmed-summarization", split=split)
        # Truncate!
        #dataset = dataset.select(range(256))

        # Process it.
        dataset = dataset.map(
            process_data_to_model_inputs, 
            batched=True, 
            batch_size=128, 
            #remove_columns=["article", "abstract", "id"],
        )
        # Save.
        dataset.save_to_disk(local_path)
    else:
        dataset = load_from_disk(local_path)
    
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    return dataset


# Main program.

# Tokenizer.
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

# Get splits.
train_dataset = get_dataset_split(split="train", local_path="./pubmed-summarization/train_split")
val_dataset = get_dataset_split(split="validation", local_path="./pubmed-summarization/valid_split")


# Create data loaders for our datasets.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# Model.
model = Bert2Bert()

# Create PTL trainer.
trainer = Trainer(max_epochs=5, accelerator="gpu", devices=1, strategy="ddp")

trainer.fit(model, train_loader, val_loader)


# load rouge for validation
rouge = datasets.load_metric("rouge")

#fake_preds = ["hello there", "general kenobi"]
#fake_labels = ["hi there", "generally kenobi"]
#print(rouge.compute(predictions=fake_preds, references=fake_labels))

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

##### EVALUATION
# Use 16 training examples.
test_data = datasets.load_dataset("ccdv/pubmed-summarization", split="test")
test_data = test_data.select(range(16))

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
    input_ids = inputs.input_ids .to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.bert2bert.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


# Rune evaluation on gpu.
model.eval()
model.to("cuda")
results = test_data.map(generate_summary, batched=True, batch_size=16, remove_columns=["article"])
pred_str = results["pred"]
label_str = results["abstract"]

print("Exemplary target     :\n", label_str[0][:200])
print("Exemplary prediction :\n", pred_str[0][:200])

# Calculate Rouge.
rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
print("Rouge score on test set: ", rouge_output)