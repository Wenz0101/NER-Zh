from transformers import (
  BertTokenizerFast,
  AutoModelForTokenClassification,
  TrainingArguments, 
  Trainer
)
import data
import argparse

def main(args):
    print("Loading dataset...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    label2id, id2label, train_dataset = data.get_dataset(args.train_dataset, tokenizer)
    num_labels = len(label2id)
    print("id2label:", id2label)

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=args.save_path,
        logging_steps=50,
        save_strategy="no",
        # save_steps=500,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("Begin Fine-Tuning...")
    trainer.train()

    print("Fine-Tuning Finish.")
    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for NER task")

    parser.add_argument("--train_dataset", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained model path or name")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")

    args = parser.parse_args()
    main(args)