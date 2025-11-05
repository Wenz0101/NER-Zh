import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def read_ner_file(path):
    gold_labels = []
    pred_labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 空行跳过
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            gold_labels.append(parts[1])
            pred_labels.append(parts[2])
    return gold_labels, pred_labels


def compute_token_level_scores(path):
    gold, pred = read_ner_file(path)

    precision = precision_score(gold, pred, average='macro', zero_division=0)
    recall = recall_score(gold, pred, average='macro', zero_division=0)
    f1 = f1_score(gold, pred, average='macro', zero_division=0)
    accuracy = accuracy_score(gold, pred)

    print("Token-level Evaluation:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute token-level NER evaluation metrics")
    parser.add_argument("--file", type=str, required=True, help="Path to the NER output file")
    args = parser.parse_args()

    compute_token_level_scores(args.file)
