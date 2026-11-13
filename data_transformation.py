"""II.3–II.4: biến đổi ACS PUMS Texas 2024 và chuẩn bị so sánh CART/RF.

Không huấn luyện mô hình. Các phép gom nhóm chỉ dựa vào từ điển, không dùng nhãn.
Gọi qua: python data_prepareration.py 2024
"""

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
LABEL_NAMES = {0: "Thấp", 1: "Trung bình", 2: "Cao"}
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
    entries = {
        "OCCP": [],
        "INDP": [],
        "FOD1P": [],
    }

    # Đọc các mã từ từ điển ACS
    with open(dictionary_path, encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source):
            if len(row) < 7 or row[0] != "VAL" or row[1] not in entries:
                continue

            if not row[4].isdigit():
                continue

            code = int(row[4])

            if row[4] != row[5]:
                raise ValueError(f"Cần mã đơn để lập ánh xạ: {row}")

            # Xác định nhóm của từng mã
            if row[1] == "FOD1P":
                key = code // 100

                if key not in FOD_GROUP_NAMES:
                    raise ValueError(f"Chưa có tên nhóm FOD1P {key} cho mã {code}")

                label = FOD_GROUP_NAMES[key]

            elif code == 9920:
                key = "SPECIAL_9920"
                label = "Mã đặc biệt 9920"

            else:
                if "-" not in row[6]:
                    raise ValueError(f"Không nhận diện được nhóm: {row}")

                key = row[6].split("-")[0]

                if not key.isalpha():
                    raise ValueError(f"Không nhận diện được nhóm: {row}")

                label = key

            entries[row[1]].append((code, row[6], key, label))

    # Tạo bảng ánh xạ cho từng biến
    mappings = {}

    for column, records in entries.items():
        if not records:
            raise ValueError(f"Từ điển thiếu {column}")

        table = pd.DataFrame(
            records,
            columns=["raw_code", "raw_label", "group_key", "group_label"]
        )

        if table["raw_code"].duplicated().any():
            raise ValueError(f"Trùng mã/ánh xạ chồng lấn trong {column}")

        # Gán mã nhóm
        if column == "FOD1P":
            table.loc[len(table)] = [
                0,
                "N/A (less than bachelor's degree)",
                0,
                FOD_GROUP_NAMES[0],
            ]

            table["group_code"] = table["group_key"].astype(int)

        else:
            group_order = (
                table.groupby("group_key")["raw_code"]
                .min()
                .sort_values()
                .index
            )

            group_codes = {
                key: number
                for number, key in enumerate(group_order, start=1)
            }

            table["group_code"] = table["group_key"].map(group_codes).astype(int)

        mappings[column] = table.sort_values("raw_code").reset_index(drop=True)

    return mappings


def prepare_transformed_data(df, dictionary_path, test_size=0.3, random_state=42):
    """Tạo hai phiên bản dữ liệu A và B, cùng nhãn và cùng cách chia train/test."""

    # 1. Kiểm tra các cột cần thiết
    required = BASE_FEATURES + ["PERNP"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Thiếu cột biến đổi (cần WKWN thực tế, không dùng WKW): {sorted(missing)}"
        )

    if len(df) < 10:
        raise ValueError("Cần ít nhất 10 dòng để chia dữ liệu và tạo ba lớp.")

    # 2. Chuyển dữ liệu sang dạng số và kiểm tra dữ liệu
    data = df[required].apply(pd.to_numeric, errors="raise").reset_index(drop=True)

    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Dữ liệu biến đổi phải hữu hạn và đã điền khuyết.")

    if data.mod(1).ne(0).any().any():
        raise ValueError("Không ép kiểu che giá trị thập phân sai mã.")

    if not data.WKHP.between(1, 99).all() or not data.WKWN.between(1, 52).all():
        raise ValueError("WKHP/WKWN sai miền để tính ANNUAL_HOURS.")

    if (
        not data.AGEP.between(18, 99).all()
        or (data.PERNP <= 0).any()
        or (data.COW == 8).any()
    ):
        raise ValueError("Còn bản ghi ngoài phạm vi bài toán.")

    # 3. Đọc bảng ánh xạ các nhóm mã
    mappings = load_group_mappings(dictionary_path)

    # 4. Chia train/test trước khi tính ngưỡng thu nhập
    train_rows, test_rows = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Chỉ dùng thu nhập trong tập train để tính Q1 và Q2
    quantiles = data.PERNP.iloc[train_rows].quantile([1 / 3, 2 / 3])
    q1 = quantiles.iloc[0]
    q2 = quantiles.iloc[1]

    # 5. Phân chia thu nhập thành ba lớp
    labels = pd.Series(
        np.select(
            [
                data.PERNP <= q1,
                data.PERNP <= q2,
            ],
            [0, 1],
            default=2,
        ),
        index=data.index,
        name="INCOME_CLASS",
    )

    if labels.iloc[train_rows].nunique() != 3:
        raise ValueError(
            "Train không tạo đủ ba lớp; cần kiểm tra phân phối/đồng hạng PERNP."
        )

    # 6. Tạo phiên bản A: giữ các mã chi tiết
    detailed = data[BASE_FEATURES].astype("int64").copy()

    # Đây là số giờ làm việc ước tính trong năm
    # WKHP == 99 là top-code nên không phải số giờ chính xác
    detailed["ANNUAL_HOURS"] = detailed.WKHP * detailed.WKWN

    # 7. Tạo phiên bản B: thay các mã chi tiết bằng mã nhóm
    grouped = detailed.copy()

    for column, table in mappings.items():
        values = data[column].map(
            table.set_index("raw_code")["group_code"]
        )

        if values.isna().any():
            unknown = data.loc[values.isna(), column].unique().tolist()
            raise ValueError(f"Mã {column} không có ánh xạ: {unknown[:10]}")

        grouped = grouped.rename(columns={column: f"{column}_GROUP"})
        grouped[f"{column}_GROUP"] = values.astype("int64")

    # 8. Trả về dữ liệu và thông tin cần thiết cho các bước tiếp theo
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
    """Lưu dữ liệu train/test, bảng ánh xạ và báo cáo biến đổi."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = prepared["labels"]
    positions = {
        "train": prepared["train_rows"],
        "test": prepared["test_rows"],
    }

    # Thông tin chung của dữ liệu đã biến đổi
    metadata = {
        "survey_year": "2024",
        "state": "TX",
        "rows": len(labels),
        "random_state": prepared["random_state"],
        "requested_test_size": prepared["test_size"],
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

    # Lưu nhãn và vị trí dòng của train/test
    for split, indices in positions.items():
        labels.iloc[indices].to_csv(output / f"y_{split}.csv", index=False)

        pd.DataFrame({"prepared_row_id": indices}).to_csv(
            output / f"rows_{split}.csv", index=False
        )

        counts = labels.iloc[indices].value_counts().reindex(
            [0, 1, 2], fill_value=0
        )

        metadata["splits"][split] = {
            "rows": len(indices),
            "class_counts": {
                int(label): int(count)
                for label, count in counts.items()
            },
        }

    # Lưu hai phiên bản dữ liệu A và B
    domains = []

    for variant, features in prepared["variants"].items():
        metadata["variants"][variant] = {
            "features": features.columns.tolist(),
            "feature_count": features.shape[1],
        }

        for split, indices in positions.items():
            features.iloc[indices].to_csv(
                output / f"X_{variant}_{split}.csv", index=False
            )

        # Ghi lại miền giá trị quan sát của từng đặc trưng
        for column in features:
            domains.append({
                "variant": variant,
                "feature": column,
                "min": int(features[column].min()),
                "max": int(features[column].max()),
                "unique_values": int(features[column].nunique()),
            })

    pd.DataFrame(domains).to_csv(
        output / "feature_domains.csv", index=False, encoding="utf-8-sig"
    )

    # Lưu các bảng ánh xạ mã chi tiết sang mã nhóm
    for column, table in prepared["mappings"].items():
        table.to_csv(
            output / f"mapping_{column}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Đếm số nhóm thực tế xuất hiện trong dữ liệu
    metadata["observed_groups"] = {}

    for column in prepared["mappings"]:
        group_column = f"{column}_GROUP"
        group_count = prepared["variants"]["B_grouped"][group_column].nunique()
        metadata["observed_groups"][column] = int(group_count)

    # Lưu metadata dưới dạng JSON
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Tạo báo cáo mô tả quá trình biến đổi dữ liệu
    report = [
        "KẾT QUẢ BIẾN ĐỔI DỮ LIỆU TEXAS 2024 — II.3 VÀ II.4",
        f"Tổng bản ghi sau làm sạch: {len(labels):,}",
        "II.3: A giữ mã chi tiết; B gom OCCP/INDP theo tiền tố nhãn từ điển, FOD1P theo mã // 100.",
        "Cả A và B có ANNUAL_HOURS = WKHP * WKWN, là số giờ làm ước tính trong năm.",
        "Không chia bins tuổi/giờ/tuần, không chuẩn hóa, không one-hot/target encoding.",
        "scikit-learn xử lý mã nguyên bằng ngưỡng số, không phải phép chia nominal theo tập con.",
        "II.4: Xáo trộn ngẫu nhiên trước khi tính ngưỡng; không phân tầng theo nhãn chưa tồn tại.",
        f"Seed: {prepared['random_state']}; tam phân vị chỉ tính trên train (nội suy tuyến tính mặc định pandas).",
        f"Ngưỡng: q1 = {prepared['thresholds'][0]:,.2f}; q2 = {prepared['thresholds'][1]:,.2f}",
        "Nhãn 0 (Thấp): PERNP <= q1; 1 (Trung bình): q1 < PERNP <= q2; 2 (Cao): PERNP > q2.",
        "Giữ người cùng thu nhập cùng nhãn; không ép số lượng ba lớp bằng nhau.",
        "Đây là tam phân vị mẫu không trọng số, không phải ngưỡng thu nhập dân số Texas.",
        "PERNP không xuất vào X; y và ID vị trí dòng được lưu riêng. A/B có cùng dòng, thứ tự và nhãn.",
    ]

    for split, info in metadata["splits"].items():
        report.append(f"{split}: {info['rows']:,} dòng")

        for code, count in info["class_counts"].items():
            percentage = count / info["rows"] * 100
            report.append(
                f"  {code} ({LABEL_NAMES[code]}): {count:,} ({percentage:.2f}%)"
            )

    report.append(f"Số nhóm quan sát: {metadata['observed_groups']}")
    report.extend([
        "Kiểm chứng ánh xạ: mỗi mã từ điển chỉ có một nhóm, mã lạ gây lỗi thay vì tự gán nhóm khác.",
        "FOD1P_GROUP = 0 là chưa có bằng cử nhân; 37 nhóm ngành + nhóm 0 trên dữ liệu hiện có.",
        "mapping_*.csv chứa từng mã, nhãn gốc, khóa nhóm và mã nhóm để đối chiếu/trình bày phụ lục.",
        "feature_domains.csv chứa miền giá trị quan sát; không được dùng min/max này làm quy tắc học từ test.",
        "rows_*.csv là vị trí 0-based sau làm sạch, không phải ID Census, không đưa vào mô hình.",
        "Chỉ chuẩn bị dữ liệu; chưa có kết quả CART/RF. Tinh chỉnh trên train/CV, test dành đánh giá cuối.",
    ])

    text = "\n".join(report) + "\n"

    (output / "ket_qua_bien_doi.txt").write_text(
        text, encoding="utf-8"
    )

    print("\n" + text)
    print(f"Đã xuất dữ liệu: {output.resolve()}")
