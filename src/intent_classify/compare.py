import json
import csv

# ====== 需要对比的结果文件 ======
RESULT_FILES = [
    "../results/result_05b.json",
    "../results/result_1.5b.json",
    "../results/result_3b.json",
    "../results/result_7b.json",
]

OUT_CSV = "../results/compare_results.csv"



def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    model = obj.get("model_tag", path.replace("result_", "").replace(".json", ""))

    accuracy = obj["accuracy"]
    macro_f1 = obj["macro"]["f1"]
    weighted_f1 = obj["weighted"]["f1"]

    # 法律类 recall（关键指标）
    legal_recall = obj["per_class_metrics"]["法律类"]["recall"]

    return {
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "legal_recall": legal_recall,
    }


def main():
    rows = []

    for path in RESULT_FILES:
        try:
            rows.append(load_result(path))
        except FileNotFoundError:
            print(f"⚠️ 文件不存在，跳过: {path}")

    # ====== 控制台打印 ======
    print("\n===== 模型性能对比（Key Metrics）=====")
    print(
        f"{'Model':<8} | "
        f"{'Acc':>8} | "
        f"{'Macro-F1':>8} | "
        f"{'Weighted-F1':>11} | "
        f"{'Legal Recall':>13}"
    )
    print("-" * 60)

    for r in rows:
        print(
            f"{r['model']:<8} | "
            f"{r['accuracy']:.4f} | "
            f"{r['macro_f1']:.4f} | "
            f"{r['weighted_f1']:.4f} | "
            f"{r['legal_recall']:.4f}"
        )

    # ====== 写 CSV（方便做表/画图）======
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "Accuracy",
            "Macro_F1",
            "Weighted_F1",
            "Legal_Recall"
        ])
        for r in rows:
            writer.writerow([
                r["model"],
                r["accuracy"],
                r["macro_f1"],
                r["weighted_f1"],
                r["legal_recall"]
            ])

    print(f"\n✅ 对比结果已保存到: {OUT_CSV}")


if __name__ == "__main__":
    main()
