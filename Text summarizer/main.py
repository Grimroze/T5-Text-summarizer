# Abstractive Summarizer with T5-small on CNN/DailyMail (mini subset)

# If needed in a fresh environment, first install:
# !pip install -q transformers datasets evaluate accelerate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import torch

# 1) Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 2) Load CNN/DailyMail dataset (version 3.0.0)
raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

# Tiny subsets for quick training
small_train = raw_datasets["train"].shuffle(seed=42).select(range(1000))
small_val = raw_datasets["validation"].shuffle(seed=42).select(range(200))

print(small_train[0])

# 3) Model and tokenizer (T5-small)
model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# 4) Preprocessing: tokenize article and summary
max_input_length = 512   # article length
max_target_length = 128  # summary length

def preprocess_function(examples):
    # Tokenize the article (input text)
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
    )

    # Tokenize the summary (target text)
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=max_target_length,
        truncation=True,
    )

    # Attach labels to inputs
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_train = small_train.map(preprocess_function, batched=True)
tokenized_val = small_val.map(preprocess_function, batched=True)

# Convert to PyTorch tensors for Trainer
tokenized_train = tokenized_train.with_format("torch")
tokenized_val = tokenized_val.with_format("torch")

# 5) Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 6) ROUGE metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # Replace ignore index (-100) with pad_token_id before decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # Convert to percentages
    result = {k: round(v * 100, 2) for k, v in result.items()}
    return result

# 7) Training arguments
batch_size = 4  # smaller because seq2seq uses more memory

training_args = TrainingArguments(
    output_dir="t5-summarizer",
    evaluation_strategy="epoch",        # evaluate every epoch
    save_strategy="epoch",              # save checkpoints every epoch
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none",
    predict_with_generate=True,
    generation_max_length=128,          # max summary length
    generation_num_beams=4,             # beam search
)

# 8) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9) Train
trainer.train()

# 10) Evaluate
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# 11) Inference helper
def summarize(text, max_new_tokens=80, num_beams=4):
    # Add T5 task prefix
    input_text = "summarize: " + text

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    # Generate summary using beam search
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )

    # Decode to string
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 12) Quick test
example_article = small_val[0]["article"]
print("ORIGINAL ARTICLE:\n", example_article[:800], "...\n")
print("MODEL SUMMARY:\n", summarize(example_article))
