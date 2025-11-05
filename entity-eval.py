import pandas as pd
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import math
import argparse

def normalize_tag(tag):
    """清理缺失数据 & 格式化标签"""
    if isinstance(tag, str):
        tag = tag.strip()
        if tag == "" or tag.lower() == "nan":
            return "O"
        # 把 B_Time -> B-Time
        tag = tag.replace("_", "-")
        return tag
    else:
        return "O"

def read_and_process_file(data_path):
    df = pd.read_csv(data_path, sep="\t", names=["input", "label", "output"], skip_blank_lines=False)
    
    df["label"] = df["label"].apply(normalize_tag)
    df["output"] = df["output"].apply(normalize_tag)

    sentences, true_labels, pred_labels = [], [], []
    tmp_sent, tmp_true, tmp_pred = [], [], []

    for _, row in df.iterrows():
        token = row["input"]
        if isinstance(token, float) and math.isnan(token):
            if tmp_sent:  # 一句结束
                sentences.append(tmp_sent)
                true_labels.append(tmp_true)
                pred_labels.append(tmp_pred)
                tmp_sent, tmp_true, tmp_pred = [], [], []
        else:
            tmp_sent.append(str(row["input"]))
            tmp_true.append(row["label"])
            tmp_pred.append(row["output"])

    # 处理最后一句
    if tmp_sent:
        sentences.append(tmp_sent)
        true_labels.append(tmp_true)
        pred_labels.append(tmp_pred)

    return sentences, true_labels, pred_labels

def main(args):
    sentences, true_labels, pred_labels = read_and_process_file(args.file)

    print(f"共读取 {len(sentences)} 句。\n")

    # 计算指标
    acc = accuracy_score(true_labels, pred_labels)
    p = precision_score(true_labels, pred_labels, zero_division=0)
    r = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    print(classification_report(true_labels, pred_labels, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    main(args)
