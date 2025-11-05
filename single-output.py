from transformers import (
  BertTokenizerFast,
  AutoModelForTokenClassification,
)
import torch

def extract_entities(tokens, labels):
    entities = []
    entity_tokens = []
    entity_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B_"):
            if entity_tokens:
                entities.append((entity_type, "".join(entity_tokens)))
            entity_type = label[2:]
            entity_tokens = [token]
        elif label.startswith("I_") and entity_tokens:
            entity_tokens.append(token)
        else:
            if entity_tokens:
                entities.append((entity_type, "".join(entity_tokens)))
                entity_tokens = []
                entity_type = None
    if entity_tokens:
        entities.append((entity_type, "".join(entity_tokens)))

    return entities


# load model and tokenizer
model_path = "./bert-tiny-finetuned-e50"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path) 

text = "小李和小王在周末去了广州塔，然后又去珠江新城参加音乐节。"

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [model.config.id2label[p.item()] for p in predictions[0]]

# for token, label in zip(tokens, labels):
#     print(f"{token}\t{label}")

entities = extract_entities(tokens, labels)
print(text)
print(entities)