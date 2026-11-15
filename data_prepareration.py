"""
Chuẩn bị dữ liệu ACS PUMS 2024 cho bài toán dự đoán thu nhập người trưởng thành.

Các bước:
1. Nạp dữ liệu Texas từ ACS PUMS.
2. Giữ lại 14 thuộc tính nghiên cứu.
3. Lọc nhóm đối tượng nghiên cứu.
4. Xử lý giá trị khuyết thiếu ở ENG và FOD1P.
5. In thống kê mô tả cơ bản.
6. Tạo nhãn thu nhập và lưu dữ liệu đã biến đổi.
"""

from folktables import ACSDataSource


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


# 1. Nạp dữ liệu ACS PUMS 2024 của Texas
data_source = ACSDataSource(
    survey_year="2024",
    horizon="1-Year",
    survey="person",
    root_dir="data/pums",
)

df = data_source.get_data(states=["TX"], download=False)
print(f"Đã nạp dữ liệu: {df.shape[0]:,} dòng, {df.shape[1]} cột.")


# 2. Chọn 14 thuộc tính nghiên cứu
df = df[STUDY_FEATURES].copy()
print(f"Kích thước sau khi chọn thuộc tính: {df.shape}")


# 3. Lọc nhóm đối tượng nghiên cứu
initial_rows = len(df)

df = df.loc[
    (df["AGEP"] >= 18)
    & (df["WKHP"] > 0)
    & (df["PERNP"] > 0)
    & (df["COW"] != 8)
].copy().reset_index(drop=True)

print(f"Số dòng sau khi lọc: {len(df):,}")
print(f"Số dòng bị loại: {initial_rows - len(df):,}")


# 4. Xử lý giá trị khuyết thiếu
df["ENG"] = df["ENG"].fillna(0)
df["FOD1P"] = df["FOD1P"].fillna(0)

print("Đã xử lý giá trị khuyết thiếu ở ENG và FOD1P.")


# 5. Thống kê mô tả cơ bản
print("\nThông tin tập dữ liệu:")
df.info()

print("\nThống kê mô tả:")
print(df[["AGEP", "WKWN", "WKHP", "PERNP"]].describe())


# 6. Tạo nhãn thu nhập và lưu dữ liệu đã biến đổi
from data_transformation import prepare_transformed_data, export_transformed_data

prepared_data = prepare_transformed_data(
    df,
    "data/pums/data_dict_2024/PUMS_Data_Dictionary_2024.csv",
)

export_transformed_data(prepared_data, "data/processed/texas_2024")
