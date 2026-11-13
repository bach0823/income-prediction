"""
Chuẩn bị dữ liệu ACS PUMS 2024 cho bài toán dự đoán thu nhập người trưởng thành.

Các bước:
1. Nạp dữ liệu Texas từ ACS PUMS.
2. Giữ lại 14 thuộc tính nghiên cứu.
3. Chọn người trưởng thành có giờ làm và thu nhập lao động dương,
   đồng thời loại lao động gia đình không hưởng lương (COW = 8).
4. Kiểm tra và xử lý giá trị khuyết thiếu ở ENG và FOD1P.
5. In một số thống kê cơ bản.
6. Chuyển dữ liệu sang data_transformation.py để tạo nhãn thu nhập.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from folktables import ACSDataSource


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Đường dẫn thư mục chứa dữ liệu ACS PUMS trên máy chạy dự án.
DATA_PATH = r"G:\My Drive\Môn\KPDL\DataMining\BAITAPLON\Project\data\pums"

# 14 thuộc tính được sử dụng trong nghiên cứu.
STUDY_FEATURES = [
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


def load_raw_dataset(survey_year="2024"):
    """Nạp dữ liệu ACS PUMS của Texas."""
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {DATA_PATH}")

    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=DATA_PATH,
    )

    df = data_source.get_data(states=["TX"], download=False)
    print(f"Đã nạp dữ liệu: {df.shape[0]:,} dòng, {df.shape[1]} cột.")
    return df


def select_study_features(df):
    """Chỉ giữ lại các thuộc tính được sử dụng trong nghiên cứu."""
    missing_columns = [col for col in STUDY_FEATURES if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dữ liệu thiếu các cột cần thiết: {missing_columns}")

    df = df[STUDY_FEATURES].copy()
    print(f"Kích thước sau khi chọn thuộc tính: {df.shape}")
    return df


def apply_data_homogenization(df):
    """
    Chọn nhóm đối tượng nghiên cứu:
    - AGEP >= 18: từ 18 tuổi trở lên.
    - WKHP > 0: có giờ làm trong 12 tháng qua.
    - PERNP > 0: có thu nhập lao động dương.
    - COW != 8: không thuộc nhóm lao động gia đình không hưởng lương.
    """
    initial_rows = len(df)

    condition = (
        (df["AGEP"] >= 18)
        & (df["WKHP"] > 0)
        & (df["PERNP"] > 0)
        & (df["COW"] != 8)
    )

    df = df.loc[condition].copy().reset_index(drop=True)

    print(f"Số dòng ban đầu: {initial_rows:,}")
    print(f"Số dòng sau khi lọc: {len(df):,}")
    print(f"Số dòng bị loại: {initial_rows - len(df):,}")

    if df.empty:
        raise ValueError("Không còn bản ghi nào sau khi lọc dữ liệu.")

    return df


def check_missing_values(df):
    """Thống kê số lượng giá trị khuyết thiếu ở từng cột."""
    print("\nSố lượng giá trị khuyết thiếu:")
    print(df.isnull().sum())


def impute_missing_values(df):
    """
    Thay giá trị khuyết thiếu của ENG và FOD1P bằng 0.
    Nếu các cột khác vẫn còn thiếu dữ liệu thì dừng để kiểm tra,
    không tự động thay thế bằng 0.
    """
    df = df.copy()

    for col in ["ENG", "FOD1P"]:
        df[col] = df[col].fillna(0)

    if df.isnull().sum().sum() > 0:
        raise ValueError(
            "Dữ liệu vẫn còn giá trị khuyết thiếu ở các cột khác ENG và FOD1P."
        )

    print("\nĐã xử lý giá trị khuyết thiếu ở ENG và FOD1P.")
    return df


def show_basic_statistics(df):
    """In thông tin và thống kê mô tả cơ bản."""
    print("\nThông tin tập dữ liệu:")
    df.info()

    print("\nThống kê mô tả:")
    print(df[["AGEP", "WKWN", "WKHP", "PERNP"]].describe())


if __name__ == "__main__":
    survey_year = sys.argv[1] if len(sys.argv) > 1 else "2024"

    # 1. Nạp dữ liệu
    df = load_raw_dataset(survey_year)

    # 2. Chọn thuộc tính nghiên cứu
    df = select_study_features(df)

    # 3. Lọc nhóm đối tượng nghiên cứu
    df = apply_data_homogenization(df)

    # 4. Kiểm tra và xử lý giá trị khuyết thiếu
    check_missing_values(df)
    df = impute_missing_values(df)

    # 5. Thống kê cơ bản
    show_basic_statistics(df)

    # 6. Tạo nhãn thu nhập và lưu dữ liệu đã biến đổi.
    if survey_year == "2024":
        from data_transformation import prepare_transformed_data, export_transformed_data

        project_dir = Path(__file__).resolve().parent
        dictionary_path = project_dir / "data/pums/data_dict_2024/PUMS_Data_Dictionary_2024.csv"
        output_dir = project_dir / "data/processed/texas_2024"

        export_transformed_data(
            prepare_transformed_data(df, dictionary_path),
            output_dir,
        )
    else:
        print("Chỉ hỗ trợ biến đổi dữ liệu năm 2024.")
