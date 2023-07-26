import os
import re

data_path = "/home/ningj/data/SIMPLE_RECON/cleaning_robot/2023-7-24-11-57/aiimgs/"

image_files = os.listdir(data_path)

for file_name in image_files:
    # 使用正则表达式匹配以一个或多个数字组成的字符串，并分割文件名
    new_file_name = re.sub(r'\d+-', '', file_name)
    
    print(f"Renaming {file_name} to {new_file_name}")
    
    # 重命名文件（注意：这将在您的实际文件系统上重命名文件，务必确保备份原始文件）
    os.rename(data_path + file_name, data_path + new_file_name)