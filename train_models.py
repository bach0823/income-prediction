"""Huấn luyện CART/Random Forest trên A/B. Chạy: python train_models.py

Chạy thử luồng với ít dữ liệu: python train_models.py --quick
Không tính lại nhãn, không chia lại train/test và không thay đổi dữ liệu đã chuẩn bị.
"""

import argparse
import json
import sys
from io import TextIOWrapper
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parent
VARIANTS = ("A_detailed", "B_grouped")
LABELS = [0, 1, 2]
LABEL_NAMES = ["Thấp", "Trung bình", "Cao"]


def load_training_data(data_dir: Path) -> dict:
    """Kiểm tra schema, nhãn, vị trí dòng trước khi fit để tránh dùng nhầm CSV."""
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    result = {"metadata": metadata, "X": {}, "y": {}, "rows": {}}
    for split in ("train", "test"):
        target = pd.read_csv(data_dir / f"y_{split}.csv")
        if target.columns.tolist() != ["INCOME_CLASS"]:
            raise ValueError(f"Sai schema nhãn {split}")
        y = target.INCOME_CLASS
        if y.isna().any() or not y.isin(LABELS).all():
            raise ValueError(f"Nhãn {split} không thuộc 0/1/2")
        ids = pd.read_csv(data_dir / f"rows_{split}.csv")
        if ids.columns.tolist() != ["prepared_row_id"]:
            raise ValueError(f"Sai schema vị trí dòng {split}")
        if len(y) != metadata["splits"][split]["rows"] or len(ids) != len(y):
            raise ValueError(f"Số dòng {split} không khớp metadata/ID")
        if ids.prepared_row_id.isna().any() or not ids.prepared_row_id.is_unique:
            raise ValueError(f"ID {split} bị thiếu/trùng")
        actual_counts = y.value_counts().reindex(LABELS, fill_value=0)
        expected_counts = metadata["splits"][split]["class_counts"]
        if any(
            int(actual_counts[code]) != expected_counts[str(code)] for code in LABELS
        ):
            raise ValueError(f"Số lượng nhãn {split} không khớp metadata")
        result["y"][split] = y
        result["rows"][split] = ids.prepared_row_id
    if set(result["rows"]["train"]) & set(result["rows"]["test"]):
        raise ValueError("Train/test có vị trí dòng giao nhau")
    if sum(len(y) for y in result["y"].values()) != metadata["rows"]:
        raise ValueError("Tổng dòng không khớp metadata")
    for variant in VARIANTS:
        result["X"][variant] = {}
        for split in ("train", "test"):
            frame = pd.read_csv(data_dir / f"X_{variant}_{split}.csv")
            expected = metadata["variants"][variant]["features"]
            if frame.columns.tolist() != expected or len(frame) != len(
                result["y"][split]
            ):
                raise ValueError(f"Sai schema/số dòng {variant}/{split}")
            if {
                "PERNP",
                "INCOME_CLASS",
                "prepared_row_id",
                "SERIALNO",
                "SPORDER",
            } & set(frame.columns):
                raise ValueError("Đầu vào chứa nhãn/thu nhập hoặc ID")
            if "CIT" not in frame or "NATIVITY" in frame:
                raise ValueError(
                    "Cần dữ liệu đã thay NATIVITY bằng CIT; chạy lại chuẩn bị dữ liệu"
                )
            if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
                raise ValueError("Đầu vào mô hình phải là mã số, không phải chuỗi")
            if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
                raise ValueError("Đầu vào có missing/vô cực")
            result["X"][variant][split] = frame
    for split in ("train", "test"):
        a, b = (result["X"][v][split] for v in VARIANTS)
        common = a.columns.intersection(b.columns)
        if not a[common].equals(b[common]):
            raise ValueError(f"Các thuộc tính dùng chung A/B lệch dòng trong {split}")
    return result


def make_search(
    model_name: str, cv: int, seed: int, jobs: int, quick: bool
) -> GridSearchCV:
    """CV chỉ trên train; RF chạy song song bên trong, không lồng nhiều tầng worker."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    if model_name == "CART":
        estimator = DecisionTreeClassifier(criterion="gini", random_state=seed)
        grid = {
            "max_depth": [6, None] if quick else [12, None],
            "min_samples_leaf": [10] if quick else [20, 100],
        }
    elif model_name == "RandomForest":
        estimator = RandomForestClassifier(
            criterion="gini", random_state=seed, n_jobs=jobs
        )
        grid = {
            "n_estimators": [20] if quick else [100, 200],
            "max_features": ["sqrt"],
            "min_samples_leaf": [5, 20],
        }
    else:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")
    return GridSearchCV(
        estimator,
        grid,
        scoring="f1_macro",
        cv=folds,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )


def run_experiments(
    data_dir: Path,
    output_dir: Path,
    model: str = "both",
    variant: str = "both",
    cv: int = 3,
    jobs: int = -1,
    seed: int = 42,
    quick: bool = False,
) -> pd.DataFrame:
    if cv < 2 or jobs == 0:
        raise ValueError("cv phải >= 2 và jobs không được bằng 0")
    if model not in ("both", "cart", "randomforest") or variant not in (
        "both",
        *VARIANTS,
    ):
        raise ValueError("Tên mô hình/phiên bản không hợp lệ")
    data = load_training_data(data_dir)
    selected_variants = VARIANTS if variant == "both" else (variant,)
    models = (
        ["CART", "RandomForest"]
        if model == "both"
        else [{"cart": "CART", "randomforest": "RandomForest"}[model]]
    )
    positions = {}
    for split, y in data["y"].items():
        indices = np.arange(len(y))
        limit = 3000 if split == "train" else 1000
        if quick and len(indices) > limit:
            indices, _ = train_test_split(
                indices, train_size=limit, stratify=y, random_state=seed
            )
        positions[split] = indices
    train_ids, test_ids = positions["train"], positions["test"]
    y_train = data["y"]["train"].iloc[train_ids].reset_index(drop=True)
    y_test = data["y"]["test"].iloc[test_ids].reset_index(drop=True)
    if y_train.value_counts().reindex(LABELS, fill_value=0).min() < cv:
        raise ValueError("Mỗi lớp train phải có ít nhất cv bản ghi")
    output_dir.mkdir(parents=True, exist_ok=True)
    majority = int(y_train.mode().iloc[0])
    baseline = np.full(len(y_test), majority)
    summaries = []
    details = []
    for version in selected_variants:
        x_train = data["X"][version]["train"].iloc[train_ids].reset_index(drop=True)
        x_test = data["X"][version]["test"].iloc[test_ids].reset_index(drop=True)
        for name in models:
            print(
                f"\n[{version}/{name}] CV={cv}; train={len(y_train):,}, test={len(y_test):,}",
                flush=True,
            )
            search = make_search(name, cv, seed, jobs, quick)
            started = perf_counter()
            search.fit(x_train, y_train)
            fit_seconds = perf_counter() - started
            # refit=True đã fit mô hình tốt nhất trên toàn bộ train, không fit thêm bằng test.
            best = search.best_estimator_
            train_prediction = best.predict(x_train)
            started = perf_counter()
            prediction = best.predict(x_test)
            predict_seconds = perf_counter() - started
            report = classification_report(
                y_test,
                prediction,
                labels=LABELS,
                target_names=LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            )
            summary = {
                "variant": version,
                "model": name,
                "quick": quick,
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "cv_macro_f1": float(search.best_score_),
                "cv_macro_f1_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "train_accuracy": accuracy_score(y_train, train_prediction),
                "train_macro_f1": f1_score(
                    y_train,
                    train_prediction,
                    average="macro",
                    labels=LABELS,
                    zero_division=0,
                ),
                "test_accuracy": accuracy_score(y_test, prediction),
                "test_macro_f1": report["macro avg"]["f1-score"],
                "test_macro_precision": report["macro avg"]["precision"],
                "test_macro_recall": report["macro avg"]["recall"],
                "search_and_refit_seconds": fit_seconds,
                "test_predict_seconds": predict_seconds,
                "best_params": json.dumps(
                    search.best_params_, ensure_ascii=False, sort_keys=True
                ),
            }
            summary["macro_f1_train_test_gap"] = (
                summary["train_macro_f1"] - summary["test_macro_f1"]
            )
            prefix = output_dir / f"{version}_{name}"
            pd.DataFrame(search.cv_results_).to_csv(f"{prefix}_cv.csv", index=False)
            pd.DataFrame(report).T.to_csv(
                f"{prefix}_classification_report.csv", encoding="utf-8-sig"
            )
            pd.DataFrame(
                confusion_matrix(y_test, prediction, labels=LABELS),
                index=LABEL_NAMES,
                columns=LABEL_NAMES,
            ).rename_axis("actual").to_csv(
                f"{prefix}_confusion_matrix.csv", encoding="utf-8-sig"
            )
            pd.DataFrame(
                {"feature": x_train.columns, "importance": best.feature_importances_}
            ).sort_values("importance", ascending=False).to_csv(
                f"{prefix}_feature_importances.csv", index=False
            )
            pd.DataFrame(
                {
                    "prepared_row_id": data["rows"]["test"].iloc[test_ids].to_numpy(),
                    "actual": y_test.to_numpy(),
                    "predicted": prediction,
                }
            ).to_csv(f"{prefix}_predictions.csv", index=False)
            joblib.dump(best, f"{prefix}.joblib")
            details.append(
                {
                    **summary,
                    "best_params": search.best_params_,
                    "classification_report": report,
                }
            )
            summaries.append(summary)
            print(
                f"  CV Macro-F1={summary['cv_macro_f1']:.4f}; "
                f"test Accuracy={summary['test_accuracy']:.4f}; test Macro-F1={summary['test_macro_f1']:.4f}",
                flush=True,
            )
    comparisons = pd.DataFrame(summaries)
    comparisons.to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    # Không chọn lại mô hình/siêu tham số theo test. Ưu tiên theo điểm CV train.
    recommended = comparisons.sort_values("cv_macro_f1", ascending=False).iloc[0]
    experiment = {
        "quick": quick,
        "seed": seed,
        "cv_folds": cv,
        "selection_metric": "f1_macro",
        "data_dir": str(data_dir.resolve()),
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__},
        "data_metadata": data["metadata"],
        "results": details,
        "majority_baseline": {
            "predicted_class": majority,
            "test_accuracy": accuracy_score(y_test, baseline),
            "test_macro_f1": f1_score(
                y_test, baseline, average="macro", labels=LABELS, zero_division=0
            ),
        },
        "recommended_by_cv": {
            "variant": recommended["variant"],
            "model": recommended["model"],
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    notes = [
        "KẾT QUẢ CART / RANDOM FOREST — "
        + ("CHẠY THỬ, KHÔNG DÙNG NỘP BÁO CÁO" if quick else "THỰC NGHIỆM ĐẦY ĐỦ"),
        "A: mã chi tiết; B: gom nhóm. Đọc nhãn/ngưỡng và train/test đã lưu, không tính lại.",
        f"Train sử dụng: {len(y_train):,}; test sử dụng: {len(y_test):,}; seed={seed}; CV={cv} folds.",
        "Chọn siêu tham số bằng StratifiedKFold chỉ trên train, tối ưu Macro-F1; refit trên train.",
        "Đánh giá test sau khi chốt tham số; không dùng test để điều chỉnh mô hình.",
        "Thời gian tìm kiếm bao gồm các lượt CV và refit, không chỉ một lần fit.",
        "Ma trận nhầm lẫn: hàng là nhãn thật, cột là nhãn dự đoán (Thấp, Trung bình, Cao).",
        "Importance là mức giảm impurity, có thể thiên về biến nhiều giá trị; không chứng minh quan hệ nhân quả.",
        "Mã CIT/nghề/ngành được xử lý bằng ngưỡng số; không phải hỗ trợ nominal trực tiếp.",
        (
            f"Baseline luôn đoán lớp phổ biến nhất train ({majority}): "
            f"Accuracy={experiment['majority_baseline']['test_accuracy']:.4f}, "
            f"Macro-F1={experiment['majority_baseline']['test_macro_f1']:.4f}."
        ),
        comparisons[
            [
                "variant",
                "model",
                "cv_macro_f1",
                "train_macro_f1",
                "test_accuracy",
                "test_macro_f1",
            ]
        ].to_string(index=False),
        f"Ưu tiên theo CV (không chọn theo test): {recommended['variant']} / {recommended['model']}.",
        "Đọc mô hình bằng joblib.load chỉ với file tin cậy; dùng đúng thứ tự cột trong metadata.",
    ]
    if quick:
        notes.append(
            "Quick lấy mẫu phân tầng tối đa 3000 train/1000 test, RF 20 cây; chỉ kiểm tra luồng chạy."
        )
    (output_dir / "ket_qua_mo_hinh.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )
    print(f"\nĐã lưu: {output_dir.resolve()}", flush=True)
    return comparisons


def main():
    if isinstance(sys.stdout, TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR / "data/processed/texas_2024"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model", choices=["both", "cart", "randomforest"], default="both"
    )
    parser.add_argument("--variant", choices=["both", *VARIANTS], default="both")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Số luồng RF; -1 dùng các lõi khả dụng, 1 giảm tải máy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Chạy thử ít dữ liệu, KHÔNG dùng kết quả cho báo cáo",
    )
    args = parser.parse_args()
    output = args.output_dir or PROJECT_DIR / "data/results" / (
        "cart_randomforest_quick" if args.quick else "cart_randomforest"
    )
    try:
        run_experiments(
            args.data_dir,
            output,
            args.model,
            args.variant,
            args.cv,
            args.jobs,
            args.seed,
            args.quick,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(
            1,
            f"Lỗi: {error}\nNếu thiếu dữ liệu, chạy python data_prepareration.py 2024 trước.\n",
        )


if __name__ == "__main__":
    main()
