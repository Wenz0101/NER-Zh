from transformers import BertTokenizerFast, AutoModelForTokenClassification
import torch

model_path = "./bert-base-e10/checkpoint-100"
test_file = "test.txt"
output_file = "bert-base-output.txt"

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

sentences = []
sentence = []

with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:  # 遇到空行表示一句结束
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            parts = line.split()
            if len(parts) == 2:
                token, label = parts
                sentence.append((token, label))
    if sentence:  # 文件最后一句
        sentences.append(sentence)

# 推理并写出结果
with open(output_file, "w", encoding="utf-8") as f:
    for sentence in sentences:
        tokens = [w for w, _ in sentence]
        gold_labels = [l for _, l in sentence]
        text = "".join(tokens)

        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

        # 对齐预测标签
        pred_labels = []
        for i, token_id in enumerate(inputs.word_ids()):
            if token_id is not None:
                label_id = predictions[i]
                pred_labels.append(model.config.id2label[label_id])

        # 写入文件：每行三列（token, gold, pred）
        for t, g, p in zip(tokens, gold_labels, pred_labels):
            f.write(f"{t}\t{g}\t{p}\n")
        f.write("\n")  # 句子间空行

print(f"推理完成，结果已保存到 {output_file}")
