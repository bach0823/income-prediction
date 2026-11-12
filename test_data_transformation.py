import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_transformation import (
    export_transformed_data,
    label_income,
    load_group_mappings,
    prepare_transformed_data,
)

DICTIONARY_PATH = (
    Path(__file__).resolve().parent
    / "data/pums/data_dict_2024/PUMS_Data_Dictionary_2024.csv"
)


class DataTransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mappings = load_group_mappings(DICTIONARY_PATH)

    def setUp(self):
        rows = np.arange(60)
        self.frame = pd.DataFrame(
            {
                "AGEP": 25 + rows % 40,
                "SEX": 1 + rows % 2,
                "MAR": 1 + rows % 5,
                "ENG": 1 + rows % 4,
                "SCHL": 21,
                "FOD1P": np.resize([2102, 2407, 4009, 0], len(rows)),
                "COW": 1,
                "WKWN": 12 + rows % 41,
                "WKHP": 20 + rows % 30,
                "OCCP": 10,
                "INDP": 170,
                "RAC1P": 1 + rows % 9,
                "CIT": 1 + rows % 5,
                "PERNP": 1000 + 1379 * rows,
                "ID": 9000 + rows,
                "SERIALNO": 8000 + rows,
                "SPORDER": 1,
                "INCOME_CLASS": 99,
            },
            index=100 + rows * 3,
        )

    def prepare(self, frame=None, **kwargs):
        return prepare_transformed_data(
            self.frame if frame is None else frame, DICTIONARY_PATH, **kwargs
        )

    def test_known_mappings_and_unique_raw_codes(self):
        self.assertEqual(set(self.mappings), {"OCCP", "INDP", "FOD1P"})
        columns = ["raw_code", "raw_label", "group_key", "group_label", "group_code"]
        for column, table in self.mappings.items():
            with self.subTest(column=column):
                self.assertEqual(table.columns.tolist(), columns)
                self.assertFalse(table.empty)
                self.assertTrue(table.raw_code.is_unique)
                self.assertFalse(table.isna().any().any())
                self.assertTrue(
                    table.groupby("group_key").group_code.nunique().eq(1).all()
                )
                self.assertTrue(
                    table.groupby("group_code").group_key.nunique().eq(1).all()
                )
        for column, raw_code, group in [
            ("OCCP", 10, "MGR"),
            ("INDP", 170, "AGR"),
            ("FOD1P", 2102, 21),
            ("FOD1P", 2407, 24),
            ("FOD1P", 4009, 40),
            ("FOD1P", 0, 0),
        ]:
            with self.subTest(column=column, raw_code=raw_code):
                entry = self.mappings[column].set_index("raw_code").loc[raw_code]
                self.assertEqual(entry.group_key, group)
                if column == "FOD1P":
                    self.assertEqual(entry.group_code, group)

    def test_overlapping_dictionary_codes_are_rejected(self):
        with DICTIONARY_PATH.open(encoding="utf-8-sig", newline="") as source:
            records = list(csv.reader(source))
        duplicate = next(
            row
            for row in records
            if len(row) >= 7
            and row[0] == "VAL"
            and row[1] == "OCCP"
            and row[4] == "0010"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlapping_dictionary.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as output:
                csv.writer(output).writerows(records + [duplicate])
            with self.assertRaises(ValueError):
                load_group_mappings(path)

    def test_unknown_codes_are_rejected(self):
        for column, unknown in [("OCCP", 99999), ("INDP", 99999), ("FOD1P", 99999)]:
            with self.subTest(column=column):
                self.assertNotIn(unknown, self.mappings[column].raw_code.values)
                frame = self.frame.copy(deep=True)
                frame.loc[frame.index[0], column] = unknown
                with self.assertRaises(ValueError):
                    self.prepare(frame)

    def test_fractional_missing_and_nonfinite_values_are_rejected(self):
        for column in ["OCCP", "INDP", "FOD1P", "WKWN", "WKHP", "PERNP"]:
            for value in [1.5, np.nan, np.inf, -np.inf]:
                with self.subTest(column=column, value=value):
                    frame = self.frame.copy(deep=True)
                    frame[column] = frame[column].astype(float)
                    frame.loc[frame.index[0], column] = value
                    with self.assertRaises(ValueError):
                        self.prepare(frame)

    def test_features_annual_hours_and_input_immutability(self):
        before = self.frame.copy(deep=True)
        prepared = self.prepare()
        pd.testing.assert_frame_equal(self.frame, before)
        self.assertEqual(set(prepared["variants"]), {"A_detailed", "B_grouped"})
        expected_hours = (self.frame.WKHP * self.frame.WKWN).reset_index(drop=True)
        for name, features in prepared["variants"].items():
            with self.subTest(variant=name):
                self.assertEqual(len(features), len(self.frame))
                self.assertEqual(features.index.tolist(), list(range(len(self.frame))))
                self.assertTrue(
                    {"PERNP", "ID", "SERIALNO", "SPORDER", "INCOME_CLASS"}.isdisjoint(
                        features.columns
                    )
                )
                pd.testing.assert_series_equal(
                    features.ANNUAL_HOURS, expected_hours.rename("ANNUAL_HOURS")
                )
        detailed = prepared["variants"]["A_detailed"]
        grouped = prepared["variants"]["B_grouped"]
        for column, mapping in self.mappings.items():
            self.assertIn(column, detailed)
            self.assertNotIn(column, grouped)
            expected = (
                self.frame[column]
                .reset_index(drop=True)
                .map(mapping.set_index("raw_code").group_code)
            )
            pd.testing.assert_series_equal(
                grouped[f"{column}_GROUP"], expected.rename(f"{column}_GROUP")
            )
        common = [column for column in detailed if column not in self.mappings]
        pd.testing.assert_frame_equal(detailed[common], grouped[common])
        self.assertEqual(prepared["labels"].name, "INCOME_CLASS")
        self.assertEqual(set(prepared["labels"]), {0, 1, 2})

    def test_shared_split_is_disjoint_complete_and_reproducible(self):
        for options in [{}, {"test_size": 0.25, "random_state": 17}]:
            with self.subTest(options=options):
                first = self.prepare(**options)
                second = self.prepare(**options)
                train, test = first["train_rows"], first["test_rows"]
                self.assertIsInstance(train, np.ndarray)
                self.assertIsInstance(test, np.ndarray)
                self.assertTrue(np.issubdtype(train.dtype, np.integer))
                self.assertTrue(np.issubdtype(test.dtype, np.integer))
                self.assertEqual(len(np.intersect1d(train, test)), 0)
                np.testing.assert_array_equal(
                    np.sort(np.concatenate([train, test])), np.arange(len(self.frame))
                )
                self.assertEqual(
                    len(test), int(np.ceil(len(self.frame) * first["test_size"]))
                )
                self.assertEqual(first["test_size"], options.get("test_size", 0.3))
                self.assertEqual(first["random_state"], options.get("random_state", 42))
                for split in ["train_rows", "test_rows"]:
                    np.testing.assert_array_equal(first[split], second[split])
                self.assertEqual(first["thresholds"], second["thresholds"])
                pd.testing.assert_series_equal(first["labels"], second["labels"])
                for variant in first["variants"]:
                    pd.testing.assert_frame_equal(
                        first["variants"][variant], second["variants"][variant]
                    )
                    for positions in [train, test]:
                        self.assertEqual(
                            first["variants"][variant].iloc[positions].index.tolist(),
                            first["labels"].iloc[positions].index.tolist(),
                        )

    def test_thresholds_and_train_labels_do_not_use_test_income(self):
        initial = self.prepare()
        train, test = initial["train_rows"], initial["test_rows"]
        expected = tuple(self.frame.PERNP.iloc[train].quantile([1 / 3, 2 / 3]))
        self.assertIsInstance(initial["thresholds"], tuple)
        self.assertEqual(initial["thresholds"], expected)
        changed = self.frame.copy(deep=True)
        changed.iloc[test, changed.columns.get_loc("PERNP")] = 10_000_000 + np.arange(
            len(test)
        )
        modified = self.prepare(changed)
        np.testing.assert_array_equal(modified["train_rows"], train)
        np.testing.assert_array_equal(modified["test_rows"], test)
        self.assertEqual(modified["thresholds"], initial["thresholds"])
        pd.testing.assert_series_equal(
            modified["labels"].iloc[train], initial["labels"].iloc[train]
        )
        self.assertTrue(modified["labels"].iloc[test].eq(2).all())
        self.assertFalse(initial["labels"].iloc[test].eq(2).all())
        for variant in initial["variants"]:
            pd.testing.assert_frame_equal(
                modified["variants"][variant], initial["variants"][variant]
            )

    def test_label_boundaries_and_ties(self):
        earnings = pd.Series([10, 20, 20, 21, 40, 40, 41], index=[9, 2, 8, 4, 6, 5, 1])
        before = earnings.copy(deep=True)
        actual = label_income(earnings, (20, 40))
        expected = pd.Series(
            [0, 0, 0, 1, 1, 1, 2],
            index=earnings.index,
            name="INCOME_CLASS",
            dtype="int64",
        )
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_series_equal(earnings, before)
        self.assertTrue(actual.groupby(earnings).nunique().eq(1).all())

    def test_invalid_thresholds_and_income_are_rejected(self):
        for thresholds in [
            (20, 20),
            (40, 20),
            (np.nan, 40),
            (20, np.nan),
            (-np.inf, 40),
            (20, np.inf),
        ]:
            with self.subTest(thresholds=thresholds):
                with self.assertRaises(ValueError):
                    label_income(pd.Series([10, 30, 50]), thresholds)
        for value in [0, -1, np.nan, np.inf, -np.inf]:
            with self.subTest(income=value):
                with self.assertRaises(ValueError):
                    label_income(pd.Series([10, value, 50]), (20, 40))

    def test_wkw_is_not_used_as_actual_weeks(self):
        frame = self.frame.copy(deep=True)
        frame["WKW"] = 1 + np.arange(len(frame)) % 6
        prepared = self.prepare(frame)
        for variant, features in prepared["variants"].items():
            with self.subTest(variant=variant):
                self.assertNotIn("WKW", features)
                np.testing.assert_array_equal(
                    features.ANNUAL_HOURS, frame.WKHP * frame.WKWN
                )
                self.assertFalse(
                    np.array_equal(features.ANNUAL_HOURS, frame.WKHP * frame.WKW)
                )
        with self.assertRaisesRegex(ValueError, "WKWN"):
            self.prepare(frame.drop(columns="WKWN"))

    def test_export_roundtrip_preserves_all_rows_labels_and_metadata(self):
        prepared = self.prepare(test_size=0.25, random_state=17)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transformed"
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = export_transformed_data(prepared, output)
            saved = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, json.loads(json.dumps(metadata)))
            self.assertEqual(saved["survey_year"], "2024")
            self.assertEqual(saved["state"], "TX")
            self.assertEqual(saved["rows"], len(self.frame))
            self.assertEqual(saved["random_state"], 17)
            self.assertEqual(saved["requested_test_size"], 0.25)
            self.assertEqual(
                saved["thresholds_train_only"],
                dict(zip(["q1", "q2"], prepared["thresholds"])),
            )
            self.assertEqual(set(saved["label_names"]), {"0", "1", "2"})
            self.assertIn("no stratification", saved["split_method"])
            self.assertIn("WKHP * WKWN", saved["annual_hours"])
            self.assertIn("PERNP <= q1", saved["label_rule"])
            self.assertEqual(set(saved["variants"]), set(prepared["variants"]))
            self.assertEqual(set(saved["splits"]), {"train", "test"})
            for split in ["train", "test"]:
                positions = prepared[f"{split}_rows"]
                rows = pd.read_csv(output / f"rows_{split}.csv")
                self.assertEqual(rows.columns.tolist(), ["prepared_row_id"])
                np.testing.assert_array_equal(rows.prepared_row_id, positions)
                expected_y = prepared["labels"].iloc[positions].reset_index(drop=True)
                pd.testing.assert_frame_equal(
                    pd.read_csv(output / f"y_{split}.csv"), expected_y.to_frame()
                )
                self.assertEqual(saved["splits"][split]["rows"], len(positions))
                self.assertEqual(
                    saved["splits"][split]["class_counts"],
                    {str(code): int(expected_y.eq(code).sum()) for code in range(3)},
                )
                for variant, features in prepared["variants"].items():
                    with self.subTest(split=split, variant=variant):
                        pd.testing.assert_frame_equal(
                            pd.read_csv(output / f"X_{variant}_{split}.csv"),
                            features.iloc[positions].reset_index(drop=True),
                        )
                        self.assertEqual(
                            saved["variants"][variant],
                            {
                                "features": features.columns.tolist(),
                                "feature_count": features.shape[1],
                            },
                        )
            for column, table in prepared["mappings"].items():
                pd.testing.assert_frame_equal(
                    pd.read_csv(output / f"mapping_{column}.csv"), table
                )
                self.assertEqual(
                    saved["observed_groups"][column],
                    prepared["variants"]["B_grouped"][f"{column}_GROUP"].nunique(),
                )
            domains = pd.read_csv(output / "feature_domains.csv")
            expected_domains = pd.DataFrame(
                [
                    {
                        "variant": variant,
                        "feature": column,
                        "min": int(features[column].min()),
                        "max": int(features[column].max()),
                        "unique_values": int(features[column].nunique()),
                    }
                    for variant, features in prepared["variants"].items()
                    for column in features
                ]
            )
            pd.testing.assert_frame_equal(domains, expected_domains)
            self.assertTrue(
                (output / "ket_qua_bien_doi.txt").read_text(encoding="utf-8").strip()
            )


if __name__ == "__main__":
    unittest.main()
