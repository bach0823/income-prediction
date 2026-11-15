"""
Huấn luyện và so sánh CART/Random Forest trên hai phiên bản dữ liệu A/B.
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


# 1. Thiết lập đường dẫn và thông số
DATA_DIR = Path("data/processed/texas_2024")
OUTPUT_DIR = Path("data/results/cart_randomforest")

VARIANTS = ["A_detailed", "B_grouped"]
LABELS = [0, 1, 2]
LABEL_NAMES = ["Thấp", "Trung bình", "Cao"]

RANDOM_STATE = 42
CV_FOLDS = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Đọc dữ liệu
y_train = pd.read_csv(DATA_DIR / "y_train.csv")["INCOME_CLASS"]
y_test = pd.read_csv(DATA_DIR / "y_test.csv")["INCOME_CLASS"]

summaries = []


# 3. Huấn luyện và đánh giá các mô hình
for version in VARIANTS:

    print(f"\n========== Phiên bản {version} ==========")

    # Đọc dữ liệu tương ứng với phiên bản A hoặc B
    x_train = pd.read_csv(DATA_DIR / f"X_{version}_train.csv")
    x_test = pd.read_csv(DATA_DIR / f"X_{version}_test.csv")

    for model_name in ["CART", "RandomForest"]:

        print(f"\n[{version} / {model_name}] Đang huấn luyện...", flush=True)

        # 3.1. Khởi tạo mô hình và lưới siêu tham số
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
                n_jobs=-1,
            )

            params = {
                "n_estimators": [100, 200],
                "max_features": ["sqrt"],
                "min_samples_leaf": [5, 20],
            }

        # 3.2. Thiết lập Stratified K-Fold
        folds = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        # 3.3. Tìm siêu tham số tốt nhất
        search = GridSearchCV(
            model,
            params,
            scoring="f1_macro",
            cv=folds,
            n_jobs=1,
        )

        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        prediction = best_model.predict(x_test)

        # 3.4. Đánh giá mô hình
        report = classification_report(
            y_test,
            prediction,
            labels=LABELS,
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )

        prefix = OUTPUT_DIR / f"{version}_{model_name}"

        # Lưu Classification Report
        pd.DataFrame(report).T.to_csv(
            f"{prefix}_classification_report.csv",
            encoding="utf-8-sig",
        )

        # Lưu Confusion Matrix
        pd.DataFrame(
            confusion_matrix(y_test, prediction, labels=LABELS),
            index=LABEL_NAMES,
            columns=LABEL_NAMES,
        ).rename_axis("Nhãn thật").to_csv(
            f"{prefix}_confusion_matrix.csv",
            encoding="utf-8-sig",
        )

        # Lưu Feature Importance
        pd.DataFrame({
            "feature": x_train.columns,
            "importance": best_model.feature_importances_,
        }).sort_values(
            "importance",
            ascending=False,
        ).to_csv(
            f"{prefix}_feature_importances.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 3.5. Lưu kết quả tổng hợp
        summaries.append({
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
        })

        print(
            f"CV Macro-F1 = {search.best_score_:.4f} | "
            f"Test Macro-F1 = {report['macro avg']['f1-score']:.4f}",
            flush=True,
        )


# 4. So sánh và xếp hạng mô hình
comparison = pd.DataFrame(summaries).sort_values(
    "cv_macro_f1",
    ascending=False,
)

comparison.to_csv(
    OUTPUT_DIR / "comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSo sánh các mô hình:")
print(comparison.to_string(index=False))

best = comparison.iloc[0]

print(
    f"\nMô hình được ưu tiên theo CV: "
    f"{best['variant']} / {best['model']}"
)

print(f"Đã lưu kết quả tại: {OUTPUT_DIR}")
