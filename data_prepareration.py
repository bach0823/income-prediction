"""
Pipeline Chuẩn Bị Dữ Liệu (Data Preparation) cho bài toán phân loại mức thu nhập cá nhân.
Quy trình thực thi theo chuẩn khoa học dữ liệu:
1. Nạp dữ liệu vi mô (Data Loading) từ Census Bureau (ACS PUMS).
2. Chọn bản ghi phù hợp với bài toán:
   - Lựa chọn không gian thuộc tính nghiên cứu đồng nhất (14 biến cốt lõi).
   - Chỉ giữ AGEP >= 18, WKHP > 0, PERNP > 0, COW != 8.
   - Mã 9920 chỉ được khảo sát, không thêm bộ lọc dư thừa.
3. Kiểm tra & Đánh giá Nhiễu và Điểm ngoại lai (Noise & Outliers theo 2 nhóm biến: Phân loại & Số học).
4. Thống kê mô tả & Khảo sát phân phối tập dữ liệu đồng nhất sạch (info, describe, value_counts).
5. Khảo sát & Xử lý giá trị khuyết thiếu (Missing Values Audit & Imputation):
   - Khảo sát tỷ lệ khuyết thiếu (isna trên các thuộc tính).
   - Điền giá trị khuyết thiếu bằng 0 cho ENG (chỉ nói tiếng Anh) và FOD1P (chưa có bằng cử nhân).
6. Bàn giao dữ liệu đã xử lý cho data_transformation.py để tạo các phiên bản dữ liệu và nhãn thu nhập.
"""

import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from folktables import ACSDataSource

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------
# 1. Cấu hình mặc định
# ---------------------------------------------------------
DEFAULT_STUDY_FEATURES = [
    "AGEP",
    "SEX",
    "MAR",
    "ENG",
    "SCHL",
    "FOD1P",
    "COW",
    "WKWN",
    "WKHP",
    "OCCP",
    "INDP",
    "RAC1P",
    "CIT",
    "PERNP",
]

DATA_PATH = r"G:\My Drive\Môn\KPDL\DataMining\BAITAPLON\Project\data\pums"


def resolve_data_dir() -> str:
    """Trả về đường dẫn PUMS cố định trên máy chạy dự án."""
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu PUMS: {DATA_PATH}")
    return DATA_PATH


# ---------------------------------------------------------
# 2. Tải và Nạp Dữ Liệu (Data Loading)
# ---------------------------------------------------------
def load_raw_dataset(
    survey_year: str = "2024",
    horizon: str = "1-Year",
    survey: str = "person",
    states: list = ["TX"],
    download: bool = False,
) -> pd.DataFrame:
    """
    Nạp dữ liệu vi mô ACS PUMS từ Census Bureau qua thư viện folktables.
    """
    root_pums = resolve_data_dir()
    print(f"[*] Thư mục dữ liệu PUMS: {root_pums}")
    print(
        f"[*] Đang nạp ACS PUMS {survey_year} ({horizon}, survey={survey}, states={states})..."
    )

    data_source = ACSDataSource(
        survey_year=survey_year, horizon=horizon, survey=survey, root_dir=root_pums
    )

    df_raw = data_source.get_data(states=states, download=download)
    print(f"[✓] Đã nạp thành công: {df_raw.shape[0]:,} dòng x {df_raw.shape[1]:,} cột.")
    return df_raw


# ---------------------------------------------------------
# 3. Chọn thuộc tính và bản ghi phù hợp với bài toán
# ---------------------------------------------------------
def select_study_features(
    df: pd.DataFrame, features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Thu hẹp không gian dữ liệu vào đúng 14 thuộc tính nghiên cứu đồng nhất.
    Tương thích ngược: tự động dùng 'WKW' nếu tập dữ liệu là bản 2018 chưa có 'WKWN'.
    """
    if features is None:
        features = list(DEFAULT_STUDY_FEATURES)

    features = [c.upper() if c.lower() == "pernp" else c for c in features]

    cols_to_select = []
    for col in features:
        if col in df.columns:
            cols_to_select.append(col)
        elif col == "WKWN" and "WKW" in df.columns:
            print(
                "[!] Cảnh báo: Tập dữ liệu chứa biến 'WKW' (2018) thay vì 'WKWN' (2024). Tự động dùng 'WKW'."
            )
            cols_to_select.append("WKW")
        else:
            raise KeyError(f"Thiếu thuộc tính bắt buộc '{col}' trong tập dữ liệu!")

    print(
        f"\n--- [3.1] LỰA CHỌN KHÔNG GIAN THUỘC TÍNH NGHIÊN CỨU ĐỒNG NHẤT ({len(cols_to_select)} CỘT) ---"
    )
    df_selected = df[cols_to_select].copy()
    print(
        f"[✓] Kích thước sau khi giữ lại thuộc tính: {df_selected.shape[0]:,} dòng x {df_selected.shape[1]} cột."
    )
    print(f"    Danh sách cột: {list(df_selected.columns)}")
    return df_selected


def show_homogeneity_checks(df: pd.DataFrame) -> None:
    """Khảo sát ngắn các điều kiện lọc và một số mã/giá trị cần chú ý."""
    n = len(df)
    print("\n--- KHẢO SÁT TRƯỚC KHI LỌC ---")
    for col, condition, label in [
        ("AGEP", df["AGEP"] >= 18, "AGEP >= 18 (người trưởng thành)"),
        ("WKHP", df["WKHP"] > 0, "WKHP > 0 (có giờ làm)"),
        ("PERNP", df["PERNP"] > 0, "PERNP > 0 (thu nhập dương)"),
        ("COW", df["COW"] != 8, "COW != 8 (ngoài nhóm lao động gia đình không lương)"),
    ]:
        print(f"{label:<58}: {condition.sum():>8,} ({condition.mean() * 100:5.2f}%)")

    # Các trường hợp này chỉ được khảo sát, không tự động loại bỏ.
    basic_cohort = (df["AGEP"] >= 18) & (df["WKHP"] > 0) & (df["PERNP"] > 0)
    cow8 = df.loc[basic_cohort & (df["COW"] == 8), "PERNP"]
    print(f"\nCOW == 8 trong nhóm cơ bản: {len(cow8):,} dòng")
    if not cow8.empty:
        print("Phân vị PERNP:", cow8.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict())

    for label, mask in [
        ("PERNP <= 100 USD/năm", df["PERNP"] <= 100),
        ("WKHP > 84 giờ/tuần", df["WKHP"] > 84),
    ]:
        print(f"{label:<58}: {mask.sum():>8,} ({mask.mean() * 100:5.2f}%)")

    for col in ["INDP", "OCCP"]:
        count = (df[col] == 9920).sum()
        print(f"Mã 9920 trong {col}: {count:,} dòng ({count / n * 100:.2f}%)")


def apply_data_homogenization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chọn bản ghi phù hợp với bài toán (giữ tên hàm để tương thích nơi gọi cũ):
    - Điều kiện 1: Đối tượng người trưởng thành (AGEP >= 18).
    - Điều kiện 2: Có giờ làm trong 12 tháng qua (WKHP > 0), không suy ra việc làm hiện tại.
    - Điều kiện 3: Có thu nhập lao động dương (PERNP > 0).
    - Điều kiện 4: Loại lao động gia đình không hưởng lương (COW != 8) theo phạm vi đề tài, không kết luận dữ liệu sai.

    """
    print("\n" + "=" * 85)
    print("[3.3] CHỌN BẢN GHI PHÙ HỢP VỚI BÀI TOÁN")
    print("=" * 85)

    n_init = len(df)

    # 1. Các tiêu chí đồng nhất bắt buộc
    cond_age = df["AGEP"] >= 18
    cond_wkhp_pos = df["WKHP"] > 0
    cond_pernp_pos = df["PERNP"] > 0
    # COW và PERNP có thể phản ánh các công việc/thời điểm khác nhau.
    # Loại mã 8 theo phạm vi đề tài, không khẳng định PERNP dương là bất khả thi.
    cond_cow_paid = df["COW"] != 8

    # Bốn điều kiện đã loại hết mã 9920 trong Texas 2024; không thêm bộ lọc riêng.
    homogeneous_mask = cond_age & cond_wkhp_pos & cond_pernp_pos & cond_cow_paid

    print("Thống kê theo từng điều kiện (có thể trùng dòng; không cộng các số này):")
    print(
        f"- Vi phạm AGEP < 18               : {(~cond_age).sum():>7,} dòng ({(~cond_age).mean() * 100:5.2f}%)"
    )
    print(
        f"- Vi phạm WKHP <= 0               : {(~cond_wkhp_pos).sum():>7,} dòng ({(~cond_wkhp_pos).mean() * 100:5.2f}%)"
    )
    print(
        f"- Vi phạm PERNP <= 0              : {(~cond_pernp_pos).sum():>7,} dòng ({(~cond_pernp_pos).mean() * 100:5.2f}%)"
    )
    print(
        f"- Ngoài phạm vi COW == 8         : {(~cond_cow_paid).sum():>7,} dòng ({(~cond_cow_paid).mean() * 100:5.2f}%)"
    )

    print("-" * 65)

    n_excluded = (~homogeneous_mask).sum()
    print(
        f"=> Tổng số bản ghi không đồng nhất bị loại: {n_excluded:>7,} dòng ({n_excluded / n_init * 100:5.2f}%)"
    )

    df_clean = df.loc[homogeneous_mask].copy().reset_index(drop=True)
    print(
        f"[✓] Kích thước tập dữ liệu đồng nhất chuẩn: {df_clean.shape[0]:,} dòng / {n_init:,} dòng gốc ({len(df_clean) / n_init * 100:5.2f}%)."
    )

    return df_clean


# ---------------------------------------------------------
# Bước 2.1.1.2: Kiểm tra & Đánh giá Nhiễu theo 2 nhóm biến
# ---------------------------------------------------------
def load_pums_domains(survey_year: str, dictionary_path: Optional[str] = None) -> dict:
    """Đọc các khoảng mã hợp lệ (VAL) từ data dictionary ACS PUMS."""
    if dictionary_path is None:
        root = Path(__file__).resolve().parent / "data" / "pums"
        filename = f"PUMS_Data_Dictionary_{survey_year}.csv"
        candidates = [
            root / f"data_dict_{survey_year}" / filename,
            root / survey_year / "data_dict" / filename,
        ]
        dictionary_file = next((p for p in candidates if p.is_file()), None)
        if dictionary_file is None:
            raise FileNotFoundError(f"Không tìm thấy data dictionary {survey_year}: {candidates}")
        dictionary_path = str(dictionary_file)

    domains = {}
    with open(dictionary_path, encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if len(row) < 7 or row[0].strip() != "VAL":
                continue
            try:
                lower, upper = int(row[4]), int(row[5])
            except ValueError:
                # Các mã b/bb/bbbb biểu thị không áp dụng/khuyết thiếu.
                continue
            domains.setdefault(row[1].strip(), []).append((lower, upper))
    return domains


def check_noise_by_variable_groups(
    df: pd.DataFrame,
    survey_year: str = "2024",
    dictionary_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Đối chiếu mã dữ liệu với data dictionary ACS và khảo sát giá trị cực đoan.
    Giá trị thiếu được báo riêng; giá trị cực đoan hợp lệ không bị tự động xóa.
    """
    if df.empty:
        raise ValueError("Tập dữ liệu rỗng.")

    domains = load_pums_domains(str(survey_year), dictionary_path)
    wkw_col = "WKWN" if "WKWN" in df.columns else "WKW"
    categorical = ["SEX", "MAR", "CIT", "RAC1P", "COW", "ENG", "SCHL", "FOD1P", "OCCP", "INDP"]
    numerical = ["AGEP", "WKHP", "PERNP"]
    (numerical if wkw_col == "WKWN" else categorical).append(wkw_col)
    columns = categorical + numerical

    missing_columns = set(columns) - set(df.columns)
    missing_domains = set(columns) - set(domains)
    if missing_columns:
        raise ValueError(f"Thiếu cột dữ liệu: {sorted(missing_columns)}")
    if missing_domains:
        raise ValueError(f"Data dictionary thiếu miền giá trị: {sorted(missing_domains)}")

    print(f"\n--- KIỂM TRA MIỀN GIÁ TRỊ THEO ACS PUMS {survey_year} ---")
    numeric_data = {}
    invalid_rows = pd.Series(False, index=df.index)

    for group_name, group_columns in [
        ("BIẾN PHÂN LOẠI", categorical),
        ("BIẾN SỐ", numerical),
    ]:
        print(f"\n{group_name}")
        for col in group_columns:
            values = pd.to_numeric(df[col], errors="coerce")
            numeric_data[col] = values

            in_domain = pd.Series(False, index=df.index)
            for lower, upper in domains[col]:
                in_domain |= values.between(lower, upper)

            valid = in_domain & np.isfinite(values) & (values % 1 == 0)
            invalid = df[col].notna() & ~valid
            invalid_rows |= invalid

            print(
                f"{col:<7} | khác nhau: {df[col].nunique():>5,} "
                f"| thiếu: {df[col].isna().sum():>6,} "
                f"| sai mã/kiểu: {invalid.sum():>5,}"
            )
            if invalid.any():
                print("  Ví dụ:", df.loc[invalid, col].astype(str).unique()[:5].tolist())

    # Kiểm tra lại phạm vi cohort; đây không phải kiểm tra mã hợp lệ trong dictionary.
    cohort_invalid = (
        (numeric_data["AGEP"] < 18)
        | (numeric_data["WKHP"] <= 0)
        | (numeric_data["PERNP"] <= 0)
        | (numeric_data["COW"] == 8)
    ).fillna(False)

    print(f"\nSố ô khuyết thiếu: {df[columns].isna().sum().sum():,}")
    print(f"Số dòng ngoài cohort: {cohort_invalid.sum():,}")

    print("\nPhân vị các biến số:")
    print(
        pd.DataFrame(numeric_data)[numerical]
        .replace([np.inf, -np.inf], np.nan)
        .quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
        .T.round(2).to_string()
    )

    print("\nKhảo sát một số giá trị cực đoan (không tự động loại):")
    extreme_groups = {
        "AGEP >= 85": numeric_data["AGEP"] >= 85,
        "WKHP > 84": numeric_data["WKHP"] > 84,
        "WKHP == 99": numeric_data["WKHP"] == 99,
        "0 < PERNP <= 100": (numeric_data["PERNP"] > 0) & (numeric_data["PERNP"] <= 100),
        "PERNP > 500000": numeric_data["PERNP"] > 500000,
    }
    for label, mask in extreme_groups.items():
        print(f"{label:<22}: {mask.sum():>7,} dòng ({mask.mean() * 100:.2f}%)")

    invalid_count = int(invalid_rows.sum())
    print(f"\nKết quả: {invalid_count:,} dòng có giá trị sai miền/kiểu.")
    if invalid_count or cohort_invalid.any():
        raise ValueError("Có dữ liệu sai miền/kiểu hoặc ngoài cohort; cần kiểm tra trước khi tiếp tục.")

    print(f"[✓] Giữ nguyên {len(df):,} dòng. Không cắt bỏ giá trị cực đoan hợp lệ.")
    return df



# ---------------------------------------------------------
# 4. Thống Kê & Khảo Sát Tập Dữ Liệu Đồng Nhất
# ---------------------------------------------------------
def show_info(df: pd.DataFrame) -> None:
    """In thông tin tổng quan cấu trúc, kiểu dữ liệu, số lượng quan sát và bộ nhớ."""
    print("\n" + "=" * 85)
    print("4. THÔNG TIN CẤU TRÚC TẬP DỮ LIỆU ĐỒNG NHẤT (INFO)")
    print("=" * 85)
    df.info(verbose=True, show_counts=True)


def show_describe(df: pd.DataFrame) -> pd.DataFrame:
    """In thống kê mô tả (describe) cho các biến số/thứ bậc: AGEP, WKWN/WKW, WKHP, PERNP."""
    print("\n" + "=" * 85)
    print("5. THỐNG KÊ MÔ TẢ (DESCRIBE CHO CÁC BIẾN SỐ: AGEP, WKWN, WKHP, PERNP)")
    print("=" * 85)
    num_cols = [c for c in ["AGEP", "WKWN", "WKW", "WKHP", "PERNP"] if c in df.columns]
    desc = df[num_cols].describe().round(2)
    print(desc.to_string())
    return desc


def show_value_counts(df: pd.DataFrame, top_n: int = 15) -> None:
    """In phân phối tần suất giá trị (value_counts) cho các biến mã phân loại."""
    print("\n" + "=" * 85)
    print("PHÂN PHỐI TẦN SUẤT GIÁ TRỊ (VALUE COUNTS CHO CÁC BIẾN MÃ DANH MỤC)")
    print("=" * 85)
    n_total = len(df)
    num_cols = {"AGEP", "WKWN", "WKW", "WKHP", "PERNP"}
    cat_cols = [c for c in df.columns if c not in num_cols]

    for col in cat_cols:
        n_unique = df[col].nunique(dropna=False)
        print(f"\n--- Thuộc tính: {col} (Số giá trị độc nhất: {n_unique:,}) ---")
        vc = df[col].value_counts(dropna=False)

        # Hiển thị đầy đủ 100% nếu số lượng mã <= 30 (SCHL, COW, ENG, MAR, SEX, RAC1P, CIT)
        if len(vc) <= 30:
            display_vc = vc.sort_index()
            for val, count in display_vc.items():
                pct = count / n_total * 100
                val_str = (
                    "NaN"
                    if pd.isna(val)
                    else (
                        str(int(val))
                        if isinstance(val, (int, float)) and not pd.isna(val)
                        else str(val)
                    )
                )
                print(f"  Mã {val_str:<12}: {count:>7,} ({pct:5.2f}%)")
        else:
            display_vc = vc.head(top_n)
            for val, count in display_vc.items():
                pct = count / n_total * 100
                val_str = (
                    "NaN"
                    if pd.isna(val)
                    else (
                        str(int(val))
                        if isinstance(val, (int, float)) and not pd.isna(val)
                        else str(val)
                    )
                )
                print(f"  Mã {val_str:<12}: {count:>7,} ({pct:5.2f}%)")

            remaining_count = vc.iloc[top_n:].sum()
            remaining_pct = remaining_count / n_total * 100
            print(
                f"  ... và {len(vc) - top_n:,} mã khác: {remaining_count:>7,} ({remaining_pct:5.2f}%)"
            )


# ---------------------------------------------------------
# 5. Khảo Sát Giá Trị Khuyết Thiếu (Missing Values Audit)
# ---------------------------------------------------------
def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra và thống kê chi tiết tỷ lệ khuyết thiếu (isna/isnull) trên tất cả thuộc tính
    để làm căn cứ xác định chiến lược xử lý (Missing Values Strategy).
    """
    print("\n" + "=" * 85)
    print("7. KHẢO SÁT GIÁ TRỊ KHUYẾT THIẾU (MISSING VALUES AUDIT)")
    print("=" * 85)

    n_total = len(df)
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / n_total) * 100

    missing_df = (
        pd.DataFrame(
            {
                "Thuộc tính": df.columns,
                "Kiểu dữ liệu": [str(df[col].dtype) for col in df.columns],
                "Số dòng khuyết": missing_counts.values,
                "Tỷ lệ khuyết (%)": missing_pcts.round(2).values,
                "Có NaN không?": [df[col].isna().any() for col in df.columns],
            }
        )
        .sort_values(by="Số dòng khuyết", ascending=False)
        .reset_index(drop=True)
    )

    print(missing_df.to_string(index=False))

    has_nan_cols = missing_df[missing_df["Có NaN không?"]]["Thuộc tính"].tolist()
    no_nan_cols = missing_df[~missing_df["Có NaN không?"]]["Thuộc tính"].tolist()

    print("\n" + "-" * 60)
    print(
        f"✓ Số thuộc tính CÓ giá trị khuyết thiếu: {len(has_nan_cols)} cột -> {has_nan_cols}"
    )
    print(
        f"✓ Số thuộc tính KHÔNG có khuyết thiếu  : {len(no_nan_cols)} cột -> {no_nan_cols}"
    )
    print("-" * 60)

    # Kiểm tra liên hệ cấu trúc của FOD1P với bằng cấp Cử nhân trở lên (SCHL >= 21)
    if "FOD1P" in df.columns and "SCHL" in df.columns:
        print(
            "\nKiểm tra tính khuyết thiếu có cấu trúc của FOD1P (Bảng chéo với SCHL >= 21):"
        )
        ct = pd.crosstab(
            df["SCHL"] >= 21,
            df["FOD1P"].notna(),
            rownames=["SCHL >= 21 (Cử nhân+)"],
            colnames=["FOD1P.notna()"],
        )
        print(ct.to_string())

    return missing_df


# ---------------------------------------------------------
# 6. Xử Lý Giá Trị Khuyết Thiếu (Missing Values Imputation)
# ---------------------------------------------------------
def impute_missing_values(
    df: pd.DataFrame, fill_cols: Optional[List[str]] = None, fill_value: int = 0
) -> pd.DataFrame:
    """
    Xử lý giá trị khuyết thiếu bằng cách thay thế bằng 0:
    - ENG: Thay NaN bằng 0 (Ý nghĩa: Người nói tiếng Anh bản ngữ / chỉ nói tiếng Anh tại nhà).
    - FOD1P: Thay NaN bằng 0 (Ý nghĩa: Nhóm chưa có bằng Cử nhân SCHL < 21 / không có chuyên ngành).
    - Chuyển đổi kiểu dữ liệu của các cột đã điền sang số nguyên (int64).
    - Kiểm tra và xác nhận không còn giá trị NaN nào trên toàn bộ tập dữ liệu.
    """
    print("\n" + "=" * 85)
    print("8. XỬ LÝ GIÁ TRỊ KHUYẾT THIẾU (MISSING VALUES IMPUTATION -> THAY BẰNG 0)")
    print("=" * 85)

    if fill_cols is None:
        fill_cols = [c for c in ["ENG", "FOD1P"] if c in df.columns]

    df_imputed = df.copy()

    for col in fill_cols:
        n_missing_before = df_imputed[col].isna().sum()
        pct_missing = (n_missing_before / len(df_imputed)) * 100
        df_imputed[col] = df_imputed[col].fillna(fill_value).astype(np.int64)
        n_zeros = (df_imputed[col] == fill_value).sum()
        print(
            f"  + Thuộc tính {col:<6}: Đã điền {n_missing_before:>7,} giá trị NaN bằng {fill_value} ({pct_missing:5.2f}%) "
            f"-> Tổng số dòng mang giá trị {fill_value}: {n_zeros:>7,}"
        )

    total_missing_after = df_imputed.isna().sum().sum()
    print("-" * 65)
    print(
        f"[✓] Tổng số giá trị NaN còn lại trên toàn bộ dữ liệu: {total_missing_after}"
    )
    assert total_missing_after == 0, "Vẫn còn giá trị NaN sau khi xử lý!"
    print(
        f"[✓] Kích thước dữ liệu sau khi xử lý missing: {df_imputed.shape[0]:,} dòng x {df_imputed.shape[1]} cột."
    )

    return df_imputed


if __name__ == "__main__":
    survey_year = sys.argv[1] if len(sys.argv) > 1 else "2024"
    df_raw = load_raw_dataset(survey_year=survey_year)

    # Bước 1: Lựa chọn không gian thuộc tính nghiên cứu đồng nhất
    df_features = select_study_features(df_raw)

    # Bước 2: Khảo sát các điều kiện chọn đối tượng nghiên cứu
    show_homogeneity_checks(df_features)

    # Bước 3: Chọn bản ghi phù hợp với bài toán
    df_cohort = apply_data_homogenization(df_features)

    # Bước 4: Kiểm tra & Đánh giá Nhiễu theo 2 nhóm biến (Noise & Outliers)
    df_clean = check_noise_by_variable_groups(df_cohort, survey_year=survey_year)

    # Bước 5: Khảo sát giá trị khuyết thiếu (Missing Values)
    missing_summary = check_missing_values(df_clean)

    # Bước 6: Xử lý giá trị khuyết thiếu (thay thế bằng 0)
    df_imputed = impute_missing_values(df_clean, fill_value=0)

    # Bước 7: Thống kê mô tả & Khảo sát phân phối tập dữ liệu hoàn thiện
    show_info(df_imputed)
    show_describe(df_imputed)
    show_value_counts(df_imputed)

    # II.3–II.4 được tách module để không trộn tiền xử lý với biến đổi/thực nghiệm.
    if survey_year == "2024":
        from data_transformation import (
            export_transformed_data,
            prepare_transformed_data,
        )

        project_dir = Path(__file__).resolve().parent
        dictionary_path = (
            project_dir / "data/pums/data_dict_2024/PUMS_Data_Dictionary_2024.csv"
        )
        prepared = prepare_transformed_data(df_imputed, dictionary_path)
        export_transformed_data(prepared, project_dir / "data/processed/texas_2024")
    else:
        print(
            "Chỉ thực hiện phần biến đổi A/B cho năm 2024; dữ liệu năm khác dừng sau làm sạch."
        )
