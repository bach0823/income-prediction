import os
import sys
import ftplib
import time

sys.stdout.reconfigure(encoding='utf-8')

def download_pums_data_dictionary(target_dir=None):
    """
    Tải PUMS Data Dictionary 2018 (CSV, PDF, TXT) trực tiếp từ FTP Server của US Census Bureau.
    """
    if target_dir is None:
        target_dir = os.path.join(r"g:\My Drive\Môn\KPDL\DataMining\BAITAPLON\Project\data\pums\2018", "data_dict")
    
    os.makedirs(target_dir, exist_ok=True)
    
    ftp_host = "ftp2.census.gov"
    ftp_dir = "programs-surveys/acs/tech_docs/pums/data_dict"
    files_to_download = [
        "PUMS_Data_Dictionary_2018.csv",
        "PUMS_Data_Dictionary_2018.pdf",
        "PUMS_Data_Dictionary_2018.txt"
    ]
    
    print(f"--- KẾT NỐI TỚI FTP SERVER: {ftp_host} ---")
    ftp = ftplib.FTP(ftp_host, timeout=60)
    ftp.login()
    ftp.cwd(ftp_dir)
    print(f"✓ Đã chuyển đến thư mục: {ftp_dir}\n")
    
    downloaded_files = []
    for filename in files_to_download:
        dest_path = os.path.join(target_dir, filename)
        file_size = ftp.size(filename)
        print(f"[*] Đang tải {filename} ({file_size:,} bytes)...")
        
        start_time = time.time()
        with open(dest_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write, blocksize=32768)
        
        elapsed = time.time() - start_time
        print(f"✓ Hoàn thành: {dest_path} trong {elapsed:.2f}s ({file_size / (1024 * max(elapsed, 0.001)):.1f} KB/s)")
        downloaded_files.append(dest_path)
        
    ftp.quit()
    print("\n[✓] Đã tải về thành công toàn bộ PUMS Data Dictionary 2018!")
    return downloaded_files

if __name__ == '__main__':
    download_pums_data_dictionary()
