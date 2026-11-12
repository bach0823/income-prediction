import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from data_transformation import export_transformed_data, prepare_transformed_data
from train_models import load_training_data, make_search, run_experiments

DICTIONARY_PATH = (
    Path(__file__).resolve().parent
    / "data/pums/data_dict_2024/PUMS_Data_Dictionary_2024.csv"
)
VARIANTS = ("A_detailed", "B_grouped")
MODELS = {"CART": DecisionTreeClassifier, "RandomForest": RandomForestClassifier}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = np.arange(84)
        frame = pd.DataFrame(
            {
                "AGEP": 25 + rows % 40,
                "SEX": 1 + rows % 2,
                "MAR": 1 + rows % 5,
                "ENG": 1 + rows % 4,
                "SCHL": np.where(rows % 4 == 3, 20, 21),
                "FOD1P": np.resize([2102, 2407, 4009, 0], len(rows)),
                "COW": 1,
                "WKWN": 12 + rows % 41,
                "WKHP": 20 + rows % 30,
                "OCCP": 10,
                "INDP": 170,
                "RAC1P": 1 + rows % 9,
                "CIT": 1 + rows % 5,
                "PERNP": 1000 + 1379 * rows,
            }
        )
        cls.prepared = prepare_transformed_data(frame, DICTIONARY_PATH)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_dir = self.root / "transformed"
        self.output_dir = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()):
            export_transformed_data(self.prepared, self.data_dir)

    def change_csv(self, name, transform):
        path = self.data_dir / name
        frame = pd.read_csv(path)
        transform(frame).to_csv(path, index=False)

    def change_metadata(self, transform):
        path = self.data_dir / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        transform(metadata)
        path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    def set_feature_schema(self, variant, columns):
        # Keep the declared schema consistent to reach content-specific guards.
        self.change_metadata(
            lambda metadata: metadata["variants"][variant].update(
                features=columns, feature_count=len(columns)
            )
        )
        for split in ("train", "test"):
            self.change_csv(
                f"X_{variant}_{split}.csv",
                lambda frame: frame.reindex(columns=columns, fill_value=1),
            )

    def test_load_preserves_features_labels_metadata_and_disjoint_ids(self):
        data = load_training_data(self.data_dir)
        self.assertEqual(set(data), {"metadata", "X", "y", "rows"})
        self.assertEqual(data["metadata"]["rows"], 84)
        self.assertEqual(set(data["X"]), set(VARIANTS))
        for split in ("train", "test"):
            positions = self.prepared[f"{split}_rows"]
            np.testing.assert_array_equal(data["rows"][split], positions)
            pd.testing.assert_series_equal(
                data["y"][split],
                self.prepared["labels"].iloc[positions].reset_index(drop=True),
            )
            self.assertEqual(set(data["y"][split]), {0, 1, 2})
            for variant in VARIANTS:
                with self.subTest(split=split, variant=variant):
                    frame = data["X"][variant][split]
                    pd.testing.assert_frame_equal(
                        frame,
                        self.prepared["variants"][variant]
                        .iloc[positions]
                        .reset_index(drop=True),
                    )
                    self.assertIn("CIT", frame)
                    self.assertNotIn("NATIVITY", frame)
                    self.assertEqual(frame.shape, (len(positions), 14))
        train, test = (set(data["rows"][split]) for split in ("train", "test"))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(range(84)))

    def test_load_rejects_feature_schema_and_row_count_changes(self):
        path = self.data_dir / "X_A_detailed_train.csv"
        original = pd.read_csv(path)
        for changed in (original.iloc[:, ::-1], original.iloc[:-1]):
            with self.subTest(shape=changed.shape):
                changed.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "schema"):
                    load_training_data(self.data_dir)
        original.to_csv(path, index=False)

    def test_load_requires_cit_and_rejects_nativity(self):
        original = self.prepared["variants"]["A_detailed"].columns.tolist()
        for columns in (
            [column for column in original if column != "CIT"],
            ["NATIVITY" if column == "CIT" else column for column in original],
            original + ["NATIVITY"],
        ):
            with self.subTest(columns=columns):
                self.set_feature_schema("A_detailed", columns)
                with self.assertRaisesRegex(ValueError, "CIT"):
                    load_training_data(self.data_dir)

    def test_load_rejects_forbidden_columns_even_when_declared_in_metadata(self):
        original = self.prepared["variants"]["B_grouped"].columns.tolist()
        for column in (
            "PERNP",
            "INCOME_CLASS",
            "prepared_row_id",
            "SERIALNO",
            "SPORDER",
        ):
            with self.subTest(column=column):
                self.set_feature_schema("B_grouped", original + [column])
                with self.assertRaisesRegex(ValueError, "ID"):
                    load_training_data(self.data_dir)

    def test_load_rejects_invalid_label_schema_values_and_counts(self):
        path = self.data_dir / "y_train.csv"
        original = pd.read_csv(path)
        changes = [original.rename(columns={"INCOME_CLASS": "target"})]
        for value in (3, 0.5, np.nan):
            changed = original.astype(float)
            changed.iloc[0, 0] = value
            changes.append(changed)
        changed = original.copy()
        changed.iloc[0, 0] = (int(changed.iloc[0, 0]) + 1) % 3
        changes.extend([changed, original.iloc[:-1]])
        for index, changed in enumerate(changes):
            with self.subTest(case=index):
                changed.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_training_data(self.data_dir)
        original.to_csv(path, index=False)

    def test_load_rejects_invalid_ids_and_train_test_overlap(self):
        path = self.data_dir / "rows_test.csv"
        original = pd.read_csv(path)
        changes = [original.rename(columns={"prepared_row_id": "id"})]
        for value in (original.iloc[1, 0], np.nan, self.prepared["train_rows"][0]):
            changed = original.astype(float)
            changed.iloc[0, 0] = value
            changes.append(changed)
        changes.append(original.iloc[:-1])
        for index, changed in enumerate(changes):
            with self.subTest(case=index):
                changed.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_training_data(self.data_dir)
        original.to_csv(path, index=False)

    def test_load_rejects_metadata_total_mismatch(self):
        self.change_metadata(lambda metadata: metadata.update(rows=85))
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_training_data(self.data_dir)

    def test_load_rejects_non_numeric_nan_and_infinite_features(self):
        path = self.data_dir / "X_B_grouped_test.csv"
        original = pd.read_csv(path)
        for value in ("not-a-code", np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                changed = original.astype(object)
                changed.iloc[0, 0] = value
                changed.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_training_data(self.data_dir)
        original.to_csv(path, index=False)

    def test_load_rejects_misaligned_shared_features(self):
        self.change_csv(
            "X_B_grouped_test.csv", lambda frame: frame.assign(CIT=6 - frame.CIT)
        )
        with self.assertRaisesRegex(ValueError, "A/B"):
            load_training_data(self.data_dir)

    def test_make_search_fits_deterministically_and_scores_training_folds(self):
        data = load_training_data(self.data_dir)
        x, y = data["X"]["A_detailed"]["train"], data["y"]["train"]
        for name, estimator_type in MODELS.items():
            with self.subTest(model=name):
                search = make_search(name, cv=2, seed=17, jobs=1, quick=True)
                self.assertIsInstance(search, GridSearchCV)
                self.assertIsInstance(search.estimator, estimator_type)
                self.assertIsInstance(search.cv, StratifiedKFold)
                self.assertEqual(search.scoring, "f1_macro")
                self.assertEqual(search.n_jobs, 1)
                self.assertTrue(search.refit)
                self.assertTrue(search.return_train_score)
                self.assertEqual(search.error_score, "raise")
                self.assertEqual(search.estimator.random_state, 17)
                self.assertEqual(search.estimator.criterion, "gini")
                self.assertEqual(search.cv.n_splits, 2)
                self.assertTrue(search.cv.shuffle)
                self.assertEqual(search.cv.random_state, 17)
                if name == "RandomForest":
                    self.assertEqual(search.estimator.n_jobs, 1)
                search.fit(x, y)
                repeated = make_search(name, 2, 17, 1, True).fit(x, y)
                self.assertIsInstance(search.best_estimator_, estimator_type)
                self.assertEqual(search.best_params_, repeated.best_params_)
                self.assertEqual(search.best_score_, repeated.best_score_)
                np.testing.assert_array_equal(search.predict(x), repeated.predict(x))
                np.testing.assert_array_equal(
                    search.best_estimator_.classes_, [0, 1, 2]
                )
                self.assertEqual(search.best_estimator_.n_features_in_, x.shape[1])
                self.assertTrue(set(search.predict(x)).issubset({0, 1, 2}))
                validation_positions = []
                for fold, (train, valid) in enumerate(search.cv.split(x, y)):
                    self.assertFalse(set(train) & set(valid))
                    self.assertEqual(set(train) | set(valid), set(range(len(y))))
                    self.assertFalse(
                        set(data["rows"]["train"].iloc[np.r_[train, valid]])
                        & set(data["rows"]["test"])
                    )
                    validation_positions.extend(valid)
                    fitted = clone(search.best_estimator_).fit(
                        x.iloc[train], y.iloc[train]
                    )
                    for split, positions in (("train", train), ("test", valid)):
                        expected = f1_score(
                            y.iloc[positions],
                            fitted.predict(x.iloc[positions]),
                            average="macro",
                            zero_division=0,
                        )
                        self.assertAlmostEqual(
                            search.cv_results_[f"split{fold}_{split}_score"][
                                search.best_index_
                            ],
                            expected,
                        )
                self.assertEqual(sorted(validation_positions), list(range(len(y))))

    def test_quick_experiments_write_four_complete_results_using_only_train(self):
        data = load_training_data(self.data_dir)
        searches = []

        def tracked_search(*args, **kwargs):
            search = make_search(*args, **kwargs)
            search.fit = mock.Mock(wraps=search.fit)
            searches.append(search)
            return search

        with mock.patch("train_models.make_search", side_effect=tracked_search):
            with contextlib.redirect_stdout(io.StringIO()):
                comparisons = run_experiments(
                    self.data_dir,
                    self.output_dir,
                    model="both",
                    variant="both",
                    cv=2,
                    jobs=1,
                    seed=17,
                    quick=True,
                )
        self.assertEqual(len(comparisons), 4)
        self.assertEqual(
            set(zip(comparisons.variant, comparisons.model)),
            {(variant, name) for variant in VARIANTS for name in MODELS},
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(self.output_dir / "comparison.csv"),
            comparisons,
            check_dtype=False,
        )
        experiment = json.loads(
            (self.output_dir / "results.json").read_text(encoding="utf-8")
        )
        self.assertTrue(experiment["quick"])
        self.assertEqual(experiment["seed"], 17)
        self.assertEqual(experiment["cv_folds"], 2)
        self.assertEqual(experiment["selection_metric"], "f1_macro")
        self.assertEqual(experiment["data_dir"], str(self.data_dir.resolve()))
        self.assertEqual(experiment["data_metadata"], data["metadata"])
        self.assertEqual(set(experiment["versions"]), {"sklearn", "pandas"})
        self.assertEqual(len(experiment["results"]), 4)
        self.assertEqual(len(searches), 4)
        self.assertEqual(
            experiment["majority_baseline"]["predicted_class"],
            int(data["y"]["train"].mode().iloc[0]),
        )
        recommended = experiment["recommended_by_cv"]
        chosen = comparisons[
            (comparisons.variant == recommended["variant"])
            & (comparisons.model == recommended["model"])
        ]
        self.assertEqual(chosen.cv_macro_f1.iloc[0], comparisons.cv_macro_f1.max())
        self.assertTrue(
            (self.output_dir / "ket_qua_mo_hinh.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        for index, summary in comparisons.iterrows():
            variant, name = summary.variant, summary.model
            with self.subTest(variant=variant, model=name):
                x_train, x_test = (
                    data["X"][variant][split] for split in ("train", "test")
                )
                searches[index].fit.assert_called_once()
                fit_x, fit_y = searches[index].fit.call_args.args
                pd.testing.assert_frame_equal(fit_x, x_train)
                pd.testing.assert_series_equal(fit_y, data["y"]["train"])
                self.assertEqual(summary.train_rows, len(fit_y))
                self.assertEqual(summary.test_rows, len(data["y"]["test"]))
                self.assertTrue(summary.quick)
                prefix = self.output_dir / f"{variant}_{name}"
                for suffix in (
                    ".joblib",
                    "_cv.csv",
                    "_classification_report.csv",
                    "_confusion_matrix.csv",
                    "_feature_importances.csv",
                    "_predictions.csv",
                ):
                    self.assertTrue(Path(f"{prefix}{suffix}").is_file(), suffix)
                estimator = joblib.load(f"{prefix}.joblib")
                self.assertIsInstance(estimator, MODELS[name])
                prediction = pd.read_csv(f"{prefix}_predictions.csv")
                self.assertEqual(
                    prediction.columns.tolist(),
                    ["prepared_row_id", "actual", "predicted"],
                )
                self.assertEqual(len(prediction), len(data["y"]["test"]))
                np.testing.assert_array_equal(
                    prediction.prepared_row_id, data["rows"]["test"]
                )
                np.testing.assert_array_equal(prediction.actual, data["y"]["test"])
                np.testing.assert_array_equal(
                    prediction.predicted, estimator.predict(x_test)
                )
                matrix = pd.read_csv(f"{prefix}_confusion_matrix.csv", index_col=0)
                self.assertEqual(matrix.shape, (3, 3))
                self.assertEqual(matrix.to_numpy().sum(), len(prediction))
                np.testing.assert_array_equal(
                    matrix,
                    confusion_matrix(
                        prediction.actual, prediction.predicted, labels=[0, 1, 2]
                    ),
                )
                report = pd.read_csv(f"{prefix}_classification_report.csv", index_col=0)
                self.assertEqual(report.loc["macro avg", "support"], len(prediction))
                self.assertAlmostEqual(
                    report.loc["macro avg", "f1-score"], summary.test_macro_f1
                )
                cv_results = pd.read_csv(f"{prefix}_cv.csv")
                self.assertEqual(len(cv_results), 2)
                self.assertTrue(
                    {
                        "mean_train_score",
                        "split0_test_score",
                        "split1_test_score",
                    }.issubset(cv_results.columns)
                )
                self.assertAlmostEqual(
                    cv_results.mean_test_score.max(), summary.cv_macro_f1
                )
                importance = pd.read_csv(f"{prefix}_feature_importances.csv")
                self.assertEqual(set(importance.feature), set(x_train.columns))
                self.assertEqual(len(importance), x_train.shape[1])
                detail = experiment["results"][index]
                self.assertEqual((detail["variant"], detail["model"]), (variant, name))
                self.assertEqual(detail["train_rows"], len(fit_y))
                self.assertEqual(detail["test_rows"], len(prediction))
                self.assertEqual(detail["best_params"], json.loads(summary.best_params))

    def test_invalid_cv_jobs_and_options_are_rejected_before_outputs(self):
        for options in (
            {"cv": 0},
            {"cv": 1},
            {"cv": -1},
            {"jobs": 0},
            {"model": "invalid"},
            {"model": "CART"},
            {"variant": "invalid"},
            {"cv": 100},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    run_experiments(self.data_dir, self.output_dir, **options)
                self.assertFalse(self.output_dir.exists())
        with self.assertRaisesRegex(ValueError, "invalid"):
            make_search("invalid", cv=2, seed=17, jobs=1, quick=True)


if __name__ == "__main__":
    unittest.main()
