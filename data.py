from datasets import Dataset

def load_dataset(filepath):
    tokens, labels = [], []
    all_tokens, all_labels = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  
                if tokens:
                    all_tokens.append(tokens)
                    all_labels.append(labels)
                    tokens, labels = [], []
            else:
                word, tag = line.split()
                tokens.append(word)
                labels.append(tag)
    if tokens:
        all_tokens.append(tokens)
        all_labels.append(labels)
    return Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_labels})

def get_map(dataset):
    unique_labels = sorted({tag for tags in dataset["ner_tags"] for tag in tags})
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def get_dataset(path, tokenizer):
    train_dataset = load_dataset(path)
    label2id, id2label = get_map(train_dataset)
    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )
        all_labels = []

        for i, labels in enumerate(example["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[labels[word_idx]])
                else:
                    label_ids.append(label2id[labels[word_idx]])
                previous_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs
    tokenized_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    return label2id, id2label, tokenized_dataset