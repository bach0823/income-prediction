import os
import sys
import ftplib
import time

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_pums_data_dictionary(year="2024", dict_dir=None, guide_dir=None):
    """
    Tải PUMS Data Dictionary (CSV, PDF, TXT) và User Guide trực tiếp từ FTP Server của US Census Bureau.
    - year: Năm khảo sát (mặc định '2024').
    - dict_dir: Thư mục lưu Data Dictionary (mặc định: data/pums/data_dict_{year}).
    - guide_dir: Thư mục lưu User Guide (mặc định: data/pums/{year}/1-Year).
    """
    if dict_dir is None:
        dict_dir = os.path.join(PROJECT_ROOT, "data", "pums", f"data_dict_{year}")
    if guide_dir is None:
        guide_dir = os.path.join(PROJECT_ROOT, "data", "pums", year, "1-Year")

    os.makedirs(dict_dir, exist_ok=True)
    os.makedirs(guide_dir, exist_ok=True)

    ftp_host = "ftp2.census.gov"
    dict_ftp_path = "programs-surveys/acs/tech_docs/pums/data_dict"
    guide_ftp_path = "programs-surveys/acs/tech_docs/pums"

    dict_files = [
        f"PUMS_Data_Dictionary_{year}.csv",
        f"PUMS_Data_Dictionary_{year}.pdf",
        f"PUMS_Data_Dictionary_{year}.txt"
    ]
    guide_file = f"{year}ACS_PUMS_User_Guide.pdf"

    print(f"--- KẾT NỐI TỚI FTP SERVER: {ftp_host} (NĂM {year}) ---")
    ftp = ftplib.FTP(ftp_host, timeout=60)
    ftp.login()

    downloaded_files = []

    # 1. Tải bộ Data Dictionary (CSV, PDF, TXT)
    print(f"\n1. Chuyển đến thư mục từ điển: {dict_ftp_path}")
    ftp.cwd("/" + dict_ftp_path)
    for filename in dict_files:
        dest_path = os.path.join(dict_dir, filename)
        try:
            file_size = ftp.size(filename)
        except Exception:
            file_size = 0

        # Bỏ qua nếu file đã tồn tại và đủ kích thước
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == file_size and file_size > 0:
            print(f"  ✓ Đã tồn tại sẵn: {filename} ({file_size:,} bytes)")
            downloaded_files.append(dest_path)
            continue

        print(f"  [*] Đang tải {filename} ({file_size:,} bytes)...")
        start_time = time.time()
        with open(dest_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write, blocksize=32768)
        elapsed = time.time() - start_time
        print(f"  ✓ Hoàn thành: {dest_path} trong {elapsed:.2f}s")
        downloaded_files.append(dest_path)

    # 2. Tải User Guide (Technical Manual)
    print(f"\n2. Chuyển đến thư mục hướng dẫn kỹ thuật: {guide_ftp_path}")
    ftp.cwd("/" + guide_ftp_path)
    guide_dest = os.path.join(guide_dir, guide_file)
    try:
        guide_size = ftp.size(guide_file)
    except Exception:
        guide_size = 0

    if os.path.exists(guide_dest) and os.path.getsize(guide_dest) == guide_size and guide_size > 0:
        print(f"  ✓ Đã tồn tại sẵn: {guide_file} ({guide_size:,} bytes)")
        downloaded_files.append(guide_dest)
    else:
        print(f"  [*] Đang tải {guide_file} ({guide_size:,} bytes)...")
        start_time = time.time()
        with open(guide_dest, "wb") as f:
            ftp.retrbinary(f"RETR {guide_file}", f.write, blocksize=32768)
        elapsed = time.time() - start_time
        print(f"  ✓ Hoàn thành: {guide_dest} trong {elapsed:.2f}s")
        downloaded_files.append(guide_dest)

    ftp.quit()
    print(f"\n[✓] Đã tải về thành công toàn bộ tài liệu và từ điển ACS PUMS {year}!")
    return downloaded_files


if __name__ == '__main__':
    download_pums_data_dictionary(year="2024")
