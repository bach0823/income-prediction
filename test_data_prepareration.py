import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_prepareration import (
    apply_data_homogenization,
    check_noise_by_variable_groups,
)


class NoiseAuditTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "AGEP": [40],
                "SEX": [1],
                "MAR": [1],
                "ENG": [np.nan],
                "SCHL": [16],
                "FOD1P": [np.nan],
                "COW": [1],
                "WKWN": [52],
                "WKHP": [40],
                "OCCP": [10],
                "INDP": [170],
                "RAC1P": [1],
                "CIT": [1],
                "PERNP": [50000],
            }
        )

    def audit(self, frame, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = check_noise_by_variable_groups(frame, **kwargs)
        return result, output.getvalue()

    def test_missing_is_separate_and_input_is_unchanged(self):
        before = self.frame.copy(deep=True)
        result, output = self.audit(self.frame)
        self.assertIs(result, self.frame)
        pd.testing.assert_frame_equal(self.frame, before)
        self.assertIn("0 ô sai miền/kiểu trên 0 dòng", output)
        self.assertIn("Tổng missing (xử lý riêng ở bước sau): 2 ô", output)

    def test_invalid_values_are_not_silently_truncated_or_accepted(self):
        for col, value in [
            ("SEX", 1.5),
            ("SEX", "bad"),
            ("AGEP", 100),
            ("WKHP", 120),
            ("WKWN", 53),
            ("OCCP", 11),
            ("INDP", 171),
            ("FOD1P", 9999),
            ("PERNP", np.inf),
        ]:
            with self.subTest(col=col, value=value):
                frame = self.frame.copy()
                frame[col] = value
                with self.assertRaisesRegex(ValueError, "sai miền/kiểu"):
                    self.audit(frame)

    def test_citizenship_domain_uses_all_five_codes(self):
        for code in range(1, 6):
            frame = self.frame.copy()
            frame["CIT"] = code
            self.audit(frame)
        for code in [0, 6, 1.5]:
            frame = self.frame.copy()
            frame["CIT"] = code
            with self.assertRaisesRegex(ValueError, "sai miền/kiểu"):
                self.audit(frame)

    def test_extreme_values_are_preserved(self):
        frame = self.frame.copy()
        frame["AGEP"] = 92
        frame["WKHP"] = 99
        for earnings in [1, 988000]:
            with self.subTest(earnings=earnings):
                frame["PERNP"] = earnings
                result, _ = self.audit(frame)
                pd.testing.assert_frame_equal(result, frame)

    def test_all_missing_category_does_not_crash(self):
        self.audit(self.frame)  # ENG/FOD1P đều khuyết, không gọi min/max trên tập rỗng.

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rỗng"):
            self.audit(self.frame.iloc[:0])

    def test_wrong_year_is_not_silently_replaced(self):
        with self.assertRaises(FileNotFoundError):
            self.audit(self.frame, survey_year="1900")

    def test_missing_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Thiếu cột"):
            self.audit(self.frame.drop(columns="OCCP"))

    def test_wkw_2018_uses_category_codes(self):
        frame = self.frame.rename(columns={"WKWN": "WKW"})
        frame["WKW"] = 6
        self.audit(frame, survey_year="2018")
        frame["WKW"] = 52
        with self.assertRaisesRegex(ValueError, "sai miền/kiểu"):
            self.audit(frame, survey_year="2018")

    def test_valid_code_outside_cohort_is_rejected(self):
        for col, value in [("AGEP", 17), ("PERNP", 0), ("COW", 8)]:
            with self.subTest(col=col):
                frame = self.frame.copy()
                frame[col] = value
                with self.assertRaisesRegex(ValueError, "ngoài cohort"):
                    self.audit(frame)

    def test_cow8_is_excluded_by_scope(self):
        frame = pd.concat([self.frame, self.frame], ignore_index=True)
        frame.loc[1, "COW"] = 8
        with contextlib.redirect_stdout(io.StringIO()):
            result = apply_data_homogenization(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["PERNP"], 50000)

    def test_local_texas_2024(self):
        path = Path(__file__).resolve().parent / "data/pums/2024/1-Year/psam_p48.csv"
        if not path.is_file():
            self.skipTest("Không có dữ liệu Texas 2024 cục bộ")
        raw = pd.read_csv(path, usecols=list(self.frame.columns))
        with contextlib.redirect_stdout(io.StringIO()):
            cohort = apply_data_homogenization(raw)
        before = cohort.copy(deep=True)
        result, output = self.audit(cohort)
        pd.testing.assert_frame_equal(result, before)
        self.assertIn("0 ô sai miền/kiểu trên 0 dòng", output)


if __name__ == "__main__":
    unittest.main()
