import argparse
from transformers import BertTokenizerFast, AutoModelForTokenClassification
import torch

def main(args):
    # 加载模型和分词器
    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()

    # 读取测试文件
    sentences = []
    sentence = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 空行表示一句结束
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    token, label = parts
                    sentence.append((token, label))
        if sentence:
            sentences.append(sentence)

    # 推理并写入结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            tokens = [w for w, _ in sentence]
            gold_labels = [l for _, l in sentence]

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

            # 写入三列：token, gold, pred
            for t, g, p in zip(tokens, gold_labels, pred_labels):
                f.write(f"{t}\t{g}\t{p}\n")
            f.write("\n")  # 句子间空行

    print(f"推理完成，结果已保存到 {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER inference script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save prediction results")

    args = parser.parse_args()
    main(args)
