import pandas as pd
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import math


data_path = "bert-base-output.txt"
# data_path = "crf-output.txt"

df = pd.read_csv(data_path, sep="\t", names=["input", "label", "output"], skip_blank_lines=False)

# 清理缺失数据 & 格式化标签
def normalize_tag(tag):
    if isinstance(tag, str):
        tag = tag.strip()
        if tag == "" or tag == "nan":
            return "O"
        # 把 B_Time -> B-Time
        tag = tag.replace("_", "-")
        return tag
    else:
        return "O"

df["label"] = df["label"].apply(normalize_tag)
df["output"] = df["output"].apply(normalize_tag)

# 按句子分组
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

# 打印总句数检查
print(f"共读取 {len(sentences)} 句。")

# 实体级评测
p = precision_score(true_labels, pred_labels)
r = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"\nPrecision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1-score:  {f1:.4f}\n")

print("详细报告：")
print(classification_report(true_labels, pred_labels))
