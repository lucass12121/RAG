import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

GT_FILE = "../origndata/prediction.jsonl"
PRED_FILE = "../inference/out_0.5b_v4q.jsonl"

LABELS = ["法律类", "违规类", "闲聊类"]
# 你可以把这里改成 3b / 05b / 32b 等
MODEL_TAG = "0.5bs"

OUT_TXT = f"../results/result_{MODEL_TAG}.txt"
OUT_JSON = f"../results/result_{MODEL_TAG}.json"



def load_jsonl(path, key_name):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item[key_name]
    return data


def main():
    lines = []  # 收集所有输出行，最后写入 txt

    def log(s=""):
        print(s)
        lines.append(s)

    # ========== 1. 读取数据 ==========
    gt = load_jsonl(GT_FILE, "label")
    pred = load_jsonl(PRED_FILE, "pred")

    y_true, y_pred = [], []
    missing = 0
    for k in gt:
        if k in pred:
            y_true.append(gt[k])
            y_pred.append(pred[k])
        else:
            missing += 1

    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    wrong = total - correct

    log("===== 基本统计 =====")
    log(f"模型标识      : {MODEL_TAG}")
    log(f"GT 文件       : {GT_FILE}")
    log(f"Pred 文件     : {PRED_FILE}")
    log(f"总样本数      : {len(gt)}")
    log(f"成功对齐样本数: {total}")
    if missing:
        log(f"⚠️ 缺失预测样本数: {missing}")
    log(f"预测正确数    : {correct}")
    log(f"预测错误数    : {wrong}")

    # ========== 2. Accuracy ==========
    acc = accuracy_score(y_true, y_pred)
    log("\n===== Accuracy =====")
    log(f"Accuracy = 正确预测数 / 总样本数 = {correct} / {total} = {acc:.4f}")

    # ========== 3. Confusion Matrix ==========
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    log("\n===== 混淆矩阵（Confusion Matrix）=====")
    header = "实际\\预测".ljust(12) + "".join([lbl.center(10) for lbl in LABELS])
    log(header)
    for i, lbl in enumerate(LABELS):
        row = lbl.ljust(12) + "".join([str(cm[i][j]).center(10) for j in range(len(LABELS))])
        log(row)

    # ========== 4. TP / FP / FN / TN（逐类） ==========
    log("\n===== 各类别 TP / FP / FN / TN =====")
    per_class_counts = {}

    for i, lbl in enumerate(LABELS):
        TP = int(cm[i][i])
        FP = int(cm[:, i].sum() - TP)
        FN = int(cm[i, :].sum() - TP)
        TN = int(total - TP - FP - FN)

        per_class_counts[lbl] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

        log(f"\n[{lbl}]")
        log(f"TP (真正例) = {TP}")
        log(f"FP (误判)   = {FP}")
        log(f"FN (漏判)   = {FN}")
        log(f"TN (真反例) = {TN}")
        log("计算公式：")
        log("Precision = TP / (TP + FP)")
        log("Recall    = TP / (TP + FN)")
        log("F1        = 2 * Precision * Recall / (Precision + Recall)")

    # ========== 5. Precision / Recall / F1 ==========
    log("\n===== Precision / Recall / F1（按类别） =====")
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    per_class_metrics = {}
    for i, lbl in enumerate(LABELS):
        per_class_metrics[lbl] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        log(
            f"{lbl:6s} | "
            f"P={p[i]:.4f} | "
            f"R={r[i]:.4f} | "
            f"F1={f1[i]:.4f} | "
            f"Support={support[i]}"
        )

    # ========== 6. Macro / Weighted ==========
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    log("\n===== Overall =====")
    log(f"Macro-F1    = {f1_macro:.4f}")
    log(f"Weighted-F1 = {f1_weighted:.4f}")

    # ========== 7. sklearn 官方报告 ==========
    report = classification_report(y_true, y_pred, labels=LABELS, digits=4)
    log("\n===== Classification Report（sklearn） =====")
    log(report)

    # ========== 8. 写入文件 ==========
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    result_obj = {
        "model_tag": MODEL_TAG,
        "gt_file": GT_FILE,
        "pred_file": PRED_FILE,
        "total_gt": len(gt),
        "aligned": total,
        "missing": missing,
        "correct": correct,
        "wrong": wrong,
        "accuracy": float(acc),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm.tolist()
        },
        "per_class_counts": per_class_counts,
        "per_class_metrics": per_class_metrics,
        "macro": {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f1_macro)},
        "weighted": {"precision": float(p_weighted), "recall": float(r_weighted), "f1": float(f1_weighted)},
        "classification_report": report
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)

    log(f"\n✅ 结果已保存：{OUT_TXT}")
    log(f"✅ 指标已保存：{OUT_JSON}")


if __name__ == "__main__":
    main()
