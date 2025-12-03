The training pipeline uses:

Seq2Seq modeling

Beam search decoding

ROUGE evaluation metrics

HuggingFace Trainer API

PyTorch tensors and GPU acceleration

The implementation is structured, beginner-friendly, and focuses on teaching core NLP generation concepts.

🚀 Features

Fine-tunes a pre-trained T5 model for summarization

Uses beam search for higher-quality generated summaries

Evaluates summaries using ROUGE-1, ROUGE-2, and ROUGE-L

Includes easy-to-use summarize() inference function

Works on GPU or CPU

Trains on a small dataset subset for fast experimentation

📚 Dataset

Dataset: CNN/DailyMail (news articles + human-written highlights)

Loaded directly from HuggingFace Datasets:

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")


Training used a small subset for speed:

1000 training examples

200 validation examples

Each example has:

article → news content

highlights → summary

🧠 Model

Base model:

t5-small


Chosen because it is:

Lightweight

Fast on small GPUs

Good quality for summarization tasks

The model architecture is encoder–decoder, unlike BERT.

🏋️ Training

Training handled by HuggingFace Trainer.

Key hyperparameters:

TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
)

Highlights:

Beam search improves summary coherence

predict_with_generate=True forces the trainer to generate full summaries during evaluation

Best model checkpoint saved automatically

📊 Evaluation Metrics

The project uses ROUGE metrics to evaluate summary quality:

ROUGE-1: single-word overlap

ROUGE-2: two-word overlap

ROUGE-L: longest-subsequence overlap

Higher score = better match with human summary.

Example result (on small dataset):

ROUGE-1: ~25–35
ROUGE-2: ~10–15
ROUGE-L: ~20–30


Note: Values depend on dataset size and hyperparameters.

🧪 Inference

The model can generate summaries using a helper function:

def summarize(text, max_new_tokens=80, num_beams=4):
    input_text = "summarize: " + text
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
    
    return tokenizer.decode(ids[0], skip_special_tokens=True)


Example:

summary = summarize(article_text)
print(summary)


Output:

The government introduced a new policy aimed at improving education.

🧰 Technologies Used

Python

PyTorch

Transformers

HuggingFace Datasets

Evaluate (ROUGE)

Accelerate

Google Colab GPU

📦 Project Structure
├── README.md
├── summarizer.ipynb 