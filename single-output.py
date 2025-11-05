import argparse
from transformers import BertTokenizerFast, AutoModelForTokenClassification
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

def chunk_text(text, max_len):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def main(args):
    
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()

    # 读取整个文本
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 对长文本进行分段
    text_chunks = chunk_text(text, tokenizer.model_max_length)

    all_entities = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [model.config.id2label[p.item()] for p in predictions[0]]
        entities = extract_entities(tokens, labels)
        all_entities.extend(entities)
    for entity_type, entity_name in all_entities:
        print(f"{entity_type}\t{entity_name}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            for entity_type, entity_name in all_entities:
                f_out.write(f"{entity_type}\t{entity_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT NER entity extraction from paragraph")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save results")
    args = parser.parse_args()
    main(args)
