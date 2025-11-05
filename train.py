from transformers import (
  BertTokenizerFast,
  AutoModelForTokenClassification,
  TrainingArguments, 
  Trainer
)
import data

train_dataset_path = "./train.txt"
vali_dataset_path = "./validation.txt"
model_path = "./bert-base-chinese-ner"
save_path = "./bert-base-e10"
train_epochs = 10
batch_size = 32

print("Loading dataset...")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
label2id, id2label, train_dataset = data.get_dataset(train_dataset_path, tokenizer)
_, _, vali_dataset = data.get_dataset(vali_dataset_path, tokenizer)
num_labels = len(label2id)
print("id2label:", id2label)

print("Loading model...")
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    num_labels=num_labels,   
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  
)

training_args = TrainingArguments(
    output_dir=save_path,
    logging_steps=50,
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    num_train_epochs=train_epochs,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

print("Begin Fine-Tuning...")
trainer.train()

print("Fine-Tuning Finish")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)