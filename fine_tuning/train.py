from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# Load base model
model_name = "google/flan-t5-base"  # lighter than large, good for Colab
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load your CSV (FAQs dataset)
dataset = load_dataset("csv", data_files={"train": "faqs.csv", "test": "faqs.csv"})

# Preprocess
def preprocess(batch):
    inputs = [q for q in batch["question"]]
    targets = [a for a in batch["response"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# Training
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,   # keep ≤25
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

trainer.train()
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
