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
6. II.3–II.4: tạo ANNUAL_HOURS, hai phiên bản chi tiết/gom nhóm, chia 70/30 và gán nhãn tam phân vị train.
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
    """
    Khảo sát các điều kiện chọn đối tượng nghiên cứu và mã đặc biệt
    trước khi thực hiện lọc dữ liệu đồng nhất.
    """
    print("\n" + "=" * 85)
    print("[3.2] KHẢO SÁT CÁC ĐIỀU KIỆN CHỌN ĐỐI TƯỢNG NGHIÊN CỨU")
    print("=" * 85)

    n_total = len(df)

    # 1. Đối tượng nghiên cứu cơ bản
    mask_age = df["AGEP"] >= 18
    mask_wkhp = df["WKHP"] > 0
    mask_pernp = df["PERNP"] > 0
    print(
        f"1. Bộ lọc độ tuổi (AGEP >= 18)                 : {mask_age.sum():>7,} dòng ({mask_age.mean() * 100:5.2f}%)"
    )
    print(
        f"2. Giờ làm việc dương (WKHP > 0)               : {mask_wkhp.sum():>7,} dòng ({mask_wkhp.mean() * 100:5.2f}%)"
    )
    print(
        f"3. Thu nhập lao động dương (PERNP > 0)         : {mask_pernp.sum():>7,} dòng ({mask_pernp.mean() * 100:5.2f}%)"
    )

    # Khảo sát mã 8 trong cohort cơ bản để thấy tác động riêng của giới hạn phạm vi.
    print(
        "\n--- Kiểm tra nhóm COW == 8 (Làm việc gia đình không hưởng lương nhưng PERNP > 0) ---"
    )
    cow8 = df[mask_age & mask_wkhp & mask_pernp & (df["COW"] == 8)]
    print(
        f"Số dòng COW == 8 trong cohort cơ bản (tỷ lệ trên dữ liệu đầu vào): {len(cow8):,} ({len(cow8) / n_total * 100:.2f}%)"
    )
    if len(cow8) > 0:
        print(cow8["PERNP"].describe().round(2).to_string())
        print("\nPhân vị 10% (Deciles) PERNP của nhóm COW == 8:")
        print(cow8["PERNP"].quantile(np.arange(0.1, 1.0, 0.1)).round(2).to_string())

    # 3. Khảo sát thu nhập rất thấp, chưa đủ căn cứ kết luận là sai (PERNP <= 100 USD/năm)
    print("\n--- Kiểm tra nhóm thu nhập rất thấp (PERNP <= 100 USD/năm) ---")
    low_inc = df[df["PERNP"] <= 100]
    print(
        f"Số bản ghi có PERNP <= 100: {len(low_inc):,} ({len(low_inc) / n_total * 100:.2f}%)"
    )

    # 4. Thời gian làm việc cực đoan (WKHP > 84 giờ/tuần)
    print("\n--- Kiểm tra số giờ làm việc cực đoan (WKHP > 84 giờ/tuần) ---")
    extreme_wkhp = df[df["WKHP"] > 84]
    print(
        f"Số bản ghi có WKHP > 84: {len(extreme_wkhp):,} ({len(extreme_wkhp) / n_total * 100:.2f}%)"
    )

    # 5. Mã đặc biệt 9920 ở INDP và OCCP (Thất nghiệp / Không có kinh nghiệm 5 năm)
    print("\n--- Kiểm tra mã đặc biệt 9920 (Thất nghiệp >= 5 năm) ở INDP và OCCP ---")
    for col in ["INDP", "OCCP"]:
        if col in df.columns:
            cnt_9920 = (df[col] == 9920).sum()
            print(
                f"- Cột {col:<5}: {cnt_9920} dòng mang mã 9920 ({cnt_9920 / n_total * 100:.2f}%)"
            )


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
    """Đọc đúng các mã/khoảng mã công bố, không suy luận từ min/max."""
    if dictionary_path is None:
        root = Path(__file__).resolve().parent / "data" / "pums"
        filename = f"PUMS_Data_Dictionary_{survey_year}.csv"
        candidates = [
            root / f"data_dict_{survey_year}" / filename,
            root / survey_year / "data_dict" / filename,
        ]
        dictionary_path = next(
            (str(path) for path in candidates if path.is_file()), None
        )
        if dictionary_path is None:
            raise FileNotFoundError(
                f"Không có từ điển PUMS {survey_year}: {candidates}"
            )
    domains = {}
    with open(dictionary_path, encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source):
            if len(row) < 7 or row[0].strip() != "VAL":
                continue
            try:
                lower, upper = int(row[4]), int(row[5])
            except ValueError:
                # Mã b/bb/bbbb là N/A, được thống kê riêng dưới dạng missing.
                continue
            domains.setdefault(row[1].strip(), []).append((lower, upper))
    return domains


def check_noise_by_variable_groups(
    df: pd.DataFrame,
    survey_year: str = "2024",
    dictionary_path: Optional[str] = None,
) -> pd.DataFrame:
    """Kiểm tra miền và cohort; giữ cực đoan hợp lệ, dừng khi sai miền.

    Chạy trước điền khuyết: mã 0 tự tạo cho ENG/FOD1P không phải mã PUMS gốc.
    Missing được báo riêng và xử lý ở bước tiếp theo, không coi là mã sai.
    """
    if df.empty:
        raise ValueError("Không thể kiểm tra nhiễu trên tập dữ liệu rỗng.")
    survey_year = str(survey_year)
    domains = load_pums_domains(survey_year, dictionary_path)
    wkw_col = "WKWN" if "WKWN" in df.columns else "WKW"
    cat_cols = [
        "SEX",
        "MAR",
        "CIT",
        "RAC1P",
        "COW",
        "ENG",
        "SCHL",
        "FOD1P",
        "OCCP",
        "INDP",
    ]
    num_cols = ["AGEP", "WKHP", "PERNP"]
    # WKW là mã nhóm 1..6; WKWN mới là số tuần thực tế 1..52.
    (num_cols if wkw_col == "WKWN" else cat_cols).append(wkw_col)
    required = cat_cols + num_cols
    missing_columns = set(required) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột kiểm tra nhiễu: {sorted(missing_columns)}")
    missing_domains = set(required) - set(domains)
    if missing_domains:
        raise ValueError(
            f"Từ điển PUMS {survey_year} thiếu miền: {sorted(missing_domains)}"
        )

    print("\n" + "=" * 85)
    print(f"KIỂM TRA NHIỄU THEO TỪ ĐIỂN PUMS {survey_year} (TRƯỚC ĐIỀN KHUYẾT)")
    print("=" * 85)
    invalid_rows = pd.Series(False, index=df.index)
    numeric = {}
    invalid_cells = 0
    for title, columns in [("BIẾN PHÂN LOẠI", cat_cols), ("BIẾN SỐ", num_cols)]:
        print(f"\n--- {title} ---")
        for col in columns:
            values = pd.to_numeric(df[col], errors="coerce")
            numeric[col] = values
            in_domain = pd.Series(False, index=df.index)
            for lower, upper in domains[col]:
                in_domain |= values.between(lower, upper)
            # Không ép int trước kiểm tra: 1.5, chuỗi lỗi và vô cực phải bị phát hiện.
            valid = (in_domain & np.isfinite(values) & values.mod(1).eq(0)).fillna(
                False
            )
            invalid = df[col].notna() & ~valid
            invalid_rows |= invalid
            invalid_cells += int(invalid.sum())
            print(
                f"  + {col:<8}: {df[col].nunique():>4} giá trị khác nhau; "
                f"missing: {df[col].isna().sum():>7,}; sai miền/kiểu: {invalid.sum():>7,}"
            )
            if invalid.any():
                print(
                    f"    Ví dụ sai: {df.loc[invalid, col].astype(str).unique()[:10].tolist()}"
                )

    # Phạm vi nghiên cứu tách biệt với miền dữ liệu gốc: mã hợp lệ vẫn có thể ngoài cohort.
    cohort_violations = (
        (numeric["AGEP"] < 18)
        | (numeric["WKHP"] <= 0)
        | (numeric["PERNP"] <= 0)
        | (numeric["COW"] == 8)
    )
    print(f"\nNgoài phạm vi cohort: {cohort_violations.sum():,} dòng")
    print(
        f"Tổng missing (xử lý riêng ở bước sau): {df[required].isna().sum().sum():,} ô"
    )
    print("\n--- PHÂN VỊ BIẾN SỐ ---")
    print(
        pd.DataFrame(numeric)[num_cols]
        .replace([np.inf, -np.inf], np.nan)
        .quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
        .T.round(2)
        .to_string()
    )

    print("\n--- CÁC NHÓM CỰC ĐOAN CẦN KHẢO SÁT, KHÔNG TỰ ĐỘNG XÓA ---")
    extreme_groups = {
        "AGEP >= 85": numeric["AGEP"] >= 85,
        "WKHP > 84": numeric["WKHP"] > 84,
        "WKHP == 99 (99 giờ trở lên)": numeric["WKHP"] == 99,
        "0 < PERNP <= 100": (numeric["PERNP"] > 0) & (numeric["PERNP"] <= 100),
        "PERNP > 500000": numeric["PERNP"] > 500000,
    }
    for label, mask in extreme_groups.items():
        print(f"  + {label:<32}: {mask.sum():>7,} dòng ({mask.mean() * 100:.2f}%)")

    print(
        f"\nKết quả: {invalid_cells:,} ô sai miền/kiểu trên {invalid_rows.sum():,} dòng."
    )
    if invalid_rows.any() or cohort_violations.any():
        raise ValueError(
            "Phát hiện dữ liệu sai miền/kiểu hoặc ngoài cohort. "
            "Dừng pipeline để kiểm tra; không tự xóa, ép kiểu hay chia bins che lỗi."
        )
    print(
        f"[✓] Giá trị không khuyết nằm trong miền PUMS {survey_year} và phạm vi cohort."
    )
    print("Top-coding bảo vệ riêng tư, không bảo đảm mọi giá trị đều chính xác.")
    print("Không bắt buộc chia bins; không cắt đuôi PERNP chỉ vì thu nhập cực đoan.")
    print(f"=> Giữ nguyên {len(df):,} dòng; missing được xử lý ở bước tiếp theo.")
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
