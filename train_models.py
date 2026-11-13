"""Huấn luyện và so sánh CART/Random Forest trên hai phiên bản dữ liệu A/B.

Chạy: python train_models.py
Dùng nguyên train/test và nhãn đã được chuẩn bị từ bước trước.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
CV_FOLDS = 3


def load_data():
    """Đọc 8 file X/y đã được chuẩn bị từ bước trước."""
    data = {"X": {}, "y": {}}

    for split in ("train", "test"):
        data["y"][split] = (
            pd.read_csv(DATA_DIR / f"y_{split}.csv")["INCOME_CLASS"]
            .reset_index(drop=True)
        )

        for variant in VARIANTS:
            data["X"].setdefault(variant, {})[split] = (
                pd.read_csv(DATA_DIR / f"X_{variant}_{split}.csv")
                .reset_index(drop=True)
            )

    return data


def make_search(model_name):
    """Tìm siêu tham số bằng StratifiedKFold trên tập train."""
    folds = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    if model_name == "CART":
        model = DecisionTreeClassifier(
            criterion="gini",
            random_state=RANDOM_STATE,
        )
        params = {
            "max_depth": [12, None],
            "min_samples_leaf": [20, 100],
        }
    else:
        model = RandomForestClassifier(
            criterion="gini",
            random_state=RANDOM_STATE,
            n_jobs=1,
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
    )


def run_experiments():
    data = load_data()
    y_train = data["y"]["train"]
    y_test = data["y"]["test"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for version in VARIANTS:
        x_train = data["X"][version]["train"]
        x_test = data["X"][version]["test"]

        for model_name in ("CART", "RandomForest"):
            print(f"\n[{version} / {model_name}] Đang huấn luyện...", flush=True)

            search = make_search(model_name)
            search.fit(x_train, y_train)

            # Huấn luyện mô hình tốt nhất trên toàn bộ tập train.
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

            # 1. Precision, recall, F1 theo từng lớp.
            pd.DataFrame(report).T.to_csv(
                f"{prefix}_classification_report.csv",
                encoding="utf-8-sig",
            )

            # 2. Ma trận nhầm lẫn.
            pd.DataFrame(
                confusion_matrix(y_test, prediction, labels=LABELS),
                index=LABEL_NAMES,
                columns=LABEL_NAMES,
            ).rename_axis("Nhãn thật").to_csv(
                f"{prefix}_confusion_matrix.csv",
                encoding="utf-8-sig",
            )

            # 3. Mức độ quan trọng của các đặc trưng.
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
                    "best_params": json.dumps(
                        search.best_params_,
                        ensure_ascii=False,
                    ),
                }
            )

            print(
                f"CV Macro-F1={search.best_score_:.4f} | "
                f"Test Macro-F1={report['macro avg']['f1-score']:.4f}",
                flush=True,
            )

    comparison = pd.DataFrame(summaries).sort_values(
        "cv_macro_f1",
        ascending=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Chỉ dùng CV Macro-F1 để xếp hạng, không chọn mô hình theo kết quả test.
    best = comparison.iloc[0]

    print("\nSo sánh các mô hình:")
    print(comparison.to_string(index=False))
    print(f"\nMô hình được ưu tiên theo CV: {best['variant']} / {best['model']}")
    print(f"Đã lưu kết quả tại: {OUTPUT_DIR.resolve()}")

    return comparison


if __name__ == "__main__":
    run_experiments()
