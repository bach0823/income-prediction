"""Huấn luyện và so sánh CART/Random Forest trên hai phiên bản dữ liệu A/B.

Chạy: python train_models.py
Dùng nguyên train/test và nhãn đã lưu; không chia lại dữ liệu hay tính lại ngưỡng.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data/processed/texas_2024"
OUTPUT_DIR = PROJECT_DIR / "data/results/cart_randomforest"

VARIANTS = ("A_detailed", "B_grouped")
LABELS = [0, 1, 2]
LABEL_NAMES = ["Thấp", "Trung bình", "Cao"]
RANDOM_STATE = 42


def load_data():
    """Đọc 8 file X/y đã được chuẩn bị từ bước trước."""
    data = {"X": {}, "y": {}}

    for split in ("train", "test"):
        y = pd.read_csv(DATA_DIR / f"y_{split}.csv")["INCOME_CLASS"]

        if y.isna().any() or not y.isin(LABELS).all():
            raise ValueError(f"Nhãn {split} bị thiếu hoặc không thuộc 0/1/2.")

        data["y"][split] = y.reset_index(drop=True)

        for variant in VARIANTS:
            x = pd.read_csv(DATA_DIR / f"X_{variant}_{split}.csv")

            if len(x) != len(y):
                raise ValueError(f"Số dòng X và y không khớp: {variant}/{split}.")

            if x.isna().any().any() or not all(
                pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes
            ):
                raise ValueError(f"X {variant}/{split} có missing hoặc cột không phải số.")

            if {"PERNP", "INCOME_CLASS"} & set(x.columns):
                raise ValueError("X chứa thu nhập hoặc nhãn mục tiêu.")

            data["X"].setdefault(variant, {})[split] = x.reset_index(drop=True)

    # Hai phiên bản A/B phải giữ nguyên thứ tự các dòng và giá trị ở cột chung.
    for split in ("train", "test"):
        a = data["X"]["A_detailed"][split]
        b = data["X"]["B_grouped"][split]
        common = a.columns.intersection(b.columns)

        if not a[common].equals(b[common]):
            raise ValueError(f"Dữ liệu A/B không khớp thứ tự dòng ở tập {split}.")

    return data


def make_search(model_name, cv):
    """Tìm siêu tham số bằng StratifiedKFold trên tập train."""
    folds = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=RANDOM_STATE
    )

    if model_name == "CART":
        model = DecisionTreeClassifier(
            criterion="gini", random_state=RANDOM_STATE
        )
        params = {
            "max_depth": [12, None],
            "min_samples_leaf": [20, 100],
        }
    else:
        model = RandomForestClassifier(
            criterion="gini", random_state=RANDOM_STATE, n_jobs=1
        )
        params = {
            "n_estimators": [100, 200],
            "max_features": ["sqrt"],
            "min_samples_leaf": [5, 20],
        }

    return GridSearchCV(
        model,
        params,
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def run_experiments(variant="both", cv=3):
    if cv < 2:
        raise ValueError("cv phải >= 2.")

    selected_variants = VARIANTS if variant == "both" else (variant,)
    if any(item not in VARIANTS for item in selected_variants):
        raise ValueError("Phiên bản dữ liệu không hợp lệ.")

    data = load_data()
    y_train = data["y"]["train"]
    y_test = data["y"]["test"]

    if y_train.value_counts().reindex(LABELS, fill_value=0).min() < cv:
        raise ValueError("Mỗi lớp trong train phải có ít nhất cv bản ghi.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline: luôn dự đoán lớp xuất hiện nhiều nhất trong train.
    majority_class = int(y_train.mode().iloc[0])
    baseline = [majority_class] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline)
    baseline_f1 = f1_score(
        y_test, baseline, average="macro", labels=LABELS, zero_division=0
    )

    summaries = []

    for version in selected_variants:
        x_train = data["X"][version]["train"]
        x_test = data["X"][version]["test"]

        for model_name in ("CART", "RandomForest"):
            print(f"\n[{version} / {model_name}] Đang huấn luyện...", flush=True)

            search = make_search(model_name, cv)
            start = perf_counter()
            search.fit(x_train, y_train)
            train_seconds = perf_counter() - start

            # GridSearchCV với refit=True đã huấn luyện mô hình tốt nhất trên toàn bộ train.
            best_model = search.best_estimator_
            prediction = best_model.predict(x_test)

            report = classification_report(
                y_test,
                prediction,
                labels=LABELS,
                target_names=LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            )

            prefix = OUTPUT_DIR / f"{version}_{model_name}"

            # 1. Kết quả precision, recall, F1 theo từng lớp.
            pd.DataFrame(report).T.to_csv(
                f"{prefix}_classification_report.csv",
                encoding="utf-8-sig",
            )

            # 2. Ma trận nhầm lẫn: hàng là nhãn thật, cột là nhãn dự đoán.
            pd.DataFrame(
                confusion_matrix(y_test, prediction, labels=LABELS),
                index=LABEL_NAMES,
                columns=LABEL_NAMES,
            ).rename_axis("Nhãn thật").to_csv(
                f"{prefix}_confusion_matrix.csv",
                encoding="utf-8-sig",
            )

            # 3. Mức độ quan trọng của các đặc trưng theo mô hình.
            pd.DataFrame(
                {
                    "feature": x_train.columns,
                    "importance": best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False).to_csv(
                f"{prefix}_feature_importances.csv",
                index=False,
                encoding="utf-8-sig",
            )

            summaries.append(
                {
                    "variant": version,
                    "model": model_name,
                    "cv_macro_f1": search.best_score_,
                    "test_accuracy": accuracy_score(y_test, prediction),
                    "test_macro_f1": report["macro avg"]["f1-score"],
                    "test_macro_precision": report["macro avg"]["precision"],
                    "test_macro_recall": report["macro avg"]["recall"],
                    "baseline_accuracy": baseline_accuracy,
                    "baseline_macro_f1": baseline_f1,
                    "training_seconds": train_seconds,
                    "best_params": json.dumps(
                        search.best_params_, ensure_ascii=False
                    ),
                }
            )

            print(
                f"CV Macro-F1={search.best_score_:.4f} | "
                f"Test Macro-F1={report['macro avg']['f1-score']:.4f}",
                flush=True,
            )

    comparison = pd.DataFrame(summaries).sort_values(
        "cv_macro_f1", ascending=False
    )
    comparison.to_csv(
        OUTPUT_DIR / "comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Chỉ dùng CV Macro-F1 để xếp hạng; không chọn mô hình theo kết quả test.
    best = comparison.iloc[0]
    print("\nSo sánh các mô hình:")
    print(comparison.to_string(index=False))
    print(
        f"\nMô hình được ưu tiên theo CV: {best['variant']} / {best['model']}"
    )
    print(f"Đã lưu kết quả tại: {OUTPUT_DIR.resolve()}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["both", *VARIANTS],
        default="both",
        help="Huấn luyện cả A/B hoặc chỉ một phiên bản.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Số fold của StratifiedKFold.",
    )
    args = parser.parse_args()
    run_experiments(args.variant, args.cv)


if __name__ == "__main__":
    main()
