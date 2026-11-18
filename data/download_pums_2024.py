import os
import sys
import ftplib
import zipfile
import time

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_and_extract_pums_2024(download_docs=True):
    base_dir = os.path.join(PROJECT_ROOT, "data", "pums", "2024", "1-Year")
    os.makedirs(base_dir, exist_ok=True)

    zip_path = os.path.join(base_dir, "csv_ptx.zip")
    csv_path = os.path.join(base_dir, "psam_p48.csv")

    # 1. Tải tài liệu hướng dẫn kỹ thuật và từ điển dữ liệu nếu được yêu cầu
    if download_docs:
        from download_pums_dict import download_pums_data_dictionary
        print("--- KIỂM TRA & TẢI TÀI LIỆU PUMS 2024 (DICT & USER GUIDE) ---")
        download_pums_data_dictionary(year="2024")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100_000_000:
        print(f"\n✓ File CSV 2024 đã tồn tại sẵn tại: {csv_path} ({os.path.getsize(csv_path):,} bytes)")
        return csv_path

    print("\n--- BẮT ĐẦU TẢI DỮ LIỆU ACS PUMS 2024 (TEXAS) TỪ CENSUS BUREAU FTP ---")
    ftp_host = "ftp2.census.gov"
    ftp_dir = "programs-surveys/acs/data/pums/2024/1-Year"
    filename = "csv_ptx.zip"

    print(f"1. Đang kết nối tới FTP server: {ftp_host}...")
    ftp = ftplib.FTP(ftp_host, timeout=60)
    ftp.login()
    ftp.cwd(ftp_dir)

    total_size = ftp.size(filename)
    print(f"   Độ lớn file zip: {total_size:,} bytes (~{total_size / (1024*1024):.2f} MB)")

    downloaded = 0
    start_time = time.time()
    last_print = start_time

    print(f"2. Đang tải {filename}...")
    with open(zip_path, "wb") as f:
        def callback(chunk):
            nonlocal downloaded, last_print
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 3.0 or downloaded == total_size:
                pct = (downloaded / total_size) * 100
                speed = (downloaded / max(now - start_time, 0.001)) / (1024 * 1024)
                print(f"   Tiến độ: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%) - Tốc độ: {speed:.2f} MB/s")
                last_print = now

        ftp.retrbinary(f"RETR {filename}", callback, blocksize=65536)

    ftp.quit()
    elapsed = time.time() - start_time
    print(f"✓ Tải xong file zip trong {elapsed:.1f} giây!")

    print("3. Đang giải nén psam_p48.csv...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        print(f"   Các file trong archive: {namelist}")
        z.extractall(base_dir)

    if os.path.exists(csv_path):
        size_csv = os.path.getsize(csv_path)
        print(f"✓ Giải nén thành công: {csv_path} ({size_csv:,} bytes - ~{size_csv / (1024*1024):.1f} MB)")
    else:
        print("Cảnh báo: Không tìm thấy psam_p48.csv sau khi giải nén!")

    return csv_path


if __name__ == "__main__":
    download_and_extract_pums_2024()
