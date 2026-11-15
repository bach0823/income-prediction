"""Biến đổi ACS PUMS Texas 2024 và chuẩn bị dữ liệu cho CART/RF."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_FEATURES = [
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
]

LABEL_NAMES = {
    0: "Thấp",
    1: "Trung bình",
    2: "Cao",
}

FOD_GROUP_NAMES = {
    0: "Chưa có bằng cử nhân",
    11: "Nông nghiệp và các ngành liên quan",
    13: "Môi trường và tài nguyên",
    14: "Kiến trúc",
    15: "Nghiên cứu khu vực, dân tộc và văn minh",
    19: "Truyền thông và báo chí",
    20: "Công nghệ truyền thông",
    21: "Máy tính và hệ thống thông tin",
    22: "Dịch vụ thẩm mỹ và nghệ thuật ẩm thực",
    23: "Giáo dục",
    24: "Kỹ thuật",
    25: "Công nghệ kỹ thuật",
    26: "Ngôn ngữ học và ngoại ngữ",
    29: "Khoa học gia đình và người tiêu dùng",
    32: "Dự bị luật và nghiên cứu pháp lý",
    33: "Ngôn ngữ và văn học Anh",
    34: "Giáo dục khai phóng và nhân văn",
    35: "Khoa học thư viện",
    36: "Sinh học",
    37: "Toán và thống kê",
    38: "Công nghệ quân sự",
    40: "Đa ngành và liên ngành",
    41: "Thể chất, công viên và giải trí",
    48: "Triết học và nghiên cứu tôn giáo",
    49: "Thần học và nghề nghiệp tôn giáo",
    50: "Khoa học vật chất",
    51: "Công nghệ hạt nhân, bức xạ công nghiệp và sinh học",
    52: "Tâm lý học",
    53: "Tư pháp hình sự và phòng cháy",
    54: "Hành chính, chính sách công và công tác xã hội",
    55: "Khoa học xã hội",
    56: "Dịch vụ xây dựng",
    57: "Công nghệ điện, cơ khí, chính xác và sản xuất",
    59: "Khoa học và công nghệ vận tải",
    60: "Nghệ thuật",
    61: "Y tế và sức khỏe",
    62: "Kinh doanh",
    64: "Lịch sử",
}


def load_group_mappings(dictionary_path):
    """Tạo bảng ánh xạ mã chi tiết sang mã nhóm."""

    entries = {
        "OCCP": [],
        "INDP": [],
        "FOD1P": [],
    }

    # Đọc mã thuộc tính từ Data Dictionary
    with open(dictionary_path, encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if len(row) < 7 or row[0] != "VAL" or row[1] not in entries:
                continue

            if not row[4].isdigit() or row[4] != row[5]:
                continue

            code = int(row[4])

            # Xác định nhóm của mã
            if row[1] == "FOD1P":
                group_key = code // 100
                group_label = FOD_GROUP_NAMES[group_key]

            elif code == 9920:
                group_key = "SPECIAL_9920"
                group_label = "Mã đặc biệt 9920"

            else:
                group_key = row[6].split("-")[0].strip()
                group_label = group_key

            entries[row[1]].append(
                (code, row[6], group_key, group_label)
            )

    # Tạo bảng ánh xạ cho từng thuộc tính
    mappings = {}

    for column, records in entries.items():
        table = pd.DataFrame(
            records,
            columns=[
                "raw_code",
                "raw_label",
                "group_key",
                "group_label",
            ],
        )

        if column == "FOD1P":
            table.loc[len(table)] = [
                0,
                "N/A (less than bachelor's degree)",
                0,
                FOD_GROUP_NAMES[0],
            ]

            table["group_code"] = table["group_key"].astype(int)

        else:
            group_min_codes = table.groupby("group_key")["raw_code"].min()
            group_order = group_min_codes.sort_values().index

            group_codes = {}

            for number, key in enumerate(group_order, start=1):
                group_codes[key] = number

            table["group_code"] = (
                table["group_key"].map(group_codes).astype(int)
            )

        mappings[column] = table.sort_values("raw_code").reset_index(drop=True)

    return mappings


def prepare_transformed_data(df, dictionary_path, test_size=0.3, random_state=42):
    """Tạo hai phiên bản dữ liệu A và B, cùng nhãn và cùng cách chia train/test."""

    # 1. Chọn các thuộc tính nghiên cứu và chuyển sang dạng số
    data = df[BASE_FEATURES + ["PERNP"]].apply(
        pd.to_numeric,
        errors="raise",
    ).reset_index(drop=True)

    # 2. Tạo bảng ánh xạ mã chi tiết sang mã nhóm
    mappings = load_group_mappings(dictionary_path)

    # 3. Chia dữ liệu thành tập train và test
    train_rows, test_rows = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # 4. Tính ngưỡng thu nhập từ tập train
    q1 = data["PERNP"].iloc[train_rows].quantile(1 / 3)
    q2 = data["PERNP"].iloc[train_rows].quantile(2 / 3)

    # 5. Phân loại thu nhập thành ba nhóm
    labels = pd.Series(
        np.select(
            [
                data["PERNP"] <= q1,
                data["PERNP"] <= q2,
            ],
            [0, 1],
            default=2,
        ),
        index=data.index,
        name="INCOME_CLASS",
    )

    # 6. Phiên bản A: giữ nguyên mã thuộc tính chi tiết
    detailed = data[BASE_FEATURES].astype("int64").copy()

    # Ước tính tổng số giờ làm việc trong năm
    # WKHP = 99 biểu thị 99 giờ hoặc nhiều hơn mỗi tuần
    detailed["ANNUAL_HOURS"] = detailed["WKHP"] * detailed["WKWN"]

    # 7. Phiên bản B: thay mã chi tiết bằng mã nhóm
    grouped = detailed.copy()

    for column, table in mappings.items():
        group_codes = table.set_index("raw_code")["group_code"]

        grouped = grouped.rename(columns={column: f"{column}_GROUP"})
        grouped[f"{column}_GROUP"] = (
            data[column].map(group_codes).astype("int64")
        )

    # 8. Trả về dữ liệu và thông tin cần thiết
    return {
        "variants": {
            "A_detailed": detailed,
            "B_grouped": grouped,
        },
        "labels": labels,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "thresholds": (float(q1), float(q2)),
        "mappings": mappings,
        "random_state": random_state,
        "test_size": test_size,
    }


def export_transformed_data(prepared, output_dir):
    """Lưu dữ liệu train/test, bảng ánh xạ và metadata."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = prepared["labels"]

    positions = {
        "train": prepared["train_rows"],
        "test": prepared["test_rows"],
    }

    # 1. Tạo thông tin chung
    metadata = {
        "survey_year": "2024",
        "state": "TX",
        "rows": len(labels),
        "random_state": prepared["random_state"],
        "test_size": prepared["test_size"],
        "split_method": "random shuffle before fitting tertiles; no stratification",
        "thresholds_train_only": {
            "q1": prepared["thresholds"][0],
            "q2": prepared["thresholds"][1],
        },
        "label_names": LABEL_NAMES,
        "label_rule": "0: PERNP <= q1; 1: q1 < PERNP <= q2; 2: PERNP > q2",
        "annual_hours": "WKHP * WKWN (estimate; WKHP 99 means 99 or more)",
        "encoding": "integer codes; no native nominal handling, one-hot or target encoding",
        "variants": {},
        "splits": {},
    }

    # 2. Lưu nhãn train/test và thống kê số lượng từng lớp
    for split, indices in positions.items():
        labels.iloc[indices].to_csv(
            output / f"y_{split}.csv",
            index=False,
        )

        counts = labels.iloc[indices].value_counts().reindex(
            [0, 1, 2],
            fill_value=0,
        )

        class_counts = {}

        for label, count in counts.items():
            class_counts[int(label)] = int(count)

        metadata["splits"][split] = {
            "rows": len(indices),
            "class_counts": class_counts,
        }

    # 3. Lưu hai phiên bản dữ liệu A và B
    for variant, features in prepared["variants"].items():
        metadata["variants"][variant] = {
            "features": features.columns.tolist(),
        }

        for split, indices in positions.items():
            features.iloc[indices].to_csv(
                output / f"X_{variant}_{split}.csv",
                index=False,
            )

    # 4. Lưu các bảng ánh xạ
    for column, table in prepared["mappings"].items():
        table.to_csv(
            output / f"mapping_{column}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # 5. Lưu metadata
    with open(output / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Đã xuất dữ liệu: {output}")
