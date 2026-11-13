"""
Chuẩn bị dữ liệu ACS PUMS 2024 cho bài toán dự đoán thu nhập người trưởng thành.

Các bước:
1. Nạp dữ liệu Texas từ ACS PUMS.
2. Giữ lại 14 thuộc tính nghiên cứu.
3. Lọc nhóm người trưởng thành có giờ làm và thu nhập lao động dương,
   đồng thời loại lao động gia đình không hưởng lương.
4. Xử lý giá trị khuyết thiếu ở ENG và FOD1P.
5. In thống kê mô tả cơ bản.
6. Chuyển dữ liệu sang data_transformation.py để tạo nhãn thu nhập.
"""

import sys
from pathlib import Path

from folktables import ACSDataSource


# Đường dẫn thư mục gốc của dự án và dữ liệu ACS PUMS.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = str(PROJECT_DIR / "data/pums")

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

    return df


def impute_missing_values(df):
    """Thay giá trị khuyết thiếu của ENG và FOD1P bằng 0."""
    df = df.copy()

    for col in ["ENG", "FOD1P"]:
        df[col] = df[col].fillna(0)

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

    # 4. Xử lý giá trị khuyết thiếu
    df = impute_missing_values(df)

    # 5. Thống kê cơ bản
    show_basic_statistics(df)

    # 6. Tạo nhãn thu nhập và lưu dữ liệu đã biến đổi
    if survey_year == "2024":
        from data_transformation import prepare_transformed_data, export_transformed_data

        export_transformed_data(
            prepare_transformed_data(
                df,
                PROJECT_DIR / "data/pums/data_dict_2024/PUMS_Data_Dictionary_2024.csv",
            ),
            PROJECT_DIR / "data/processed/texas_2024",
        )
    else:
        print("Chỉ hỗ trợ biến đổi dữ liệu năm 2024.")
