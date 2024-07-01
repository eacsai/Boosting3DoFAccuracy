import os
import shutil

# 定义根目录和目标目录
root_dir = '/home/wangqw/video_dataset/KITTI/raw_data'
target_dir = os.path.join(root_dir, 'testing')

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 定义文件路径列表文件的路径
file_list_path = os.path.join('/home/wangqw/video_program/Boosting3DoFAccuracy/dataLoader', 'test2_files.txt')

# 读取文件路径列表文件
with open(file_list_path, 'r') as file:
    file_paths = file.readlines()

# 复制每个文件到目标目录
for file_path in file_paths:
    # 去除换行符并生成完整路径
    file_path = file_path.strip()
    parts = file_path.split()
    file_path = parts[0]  # 第一部分是文件路径
    new_file_path = os.path.join(
      os.path.dirname(file_path),
      'image_02/data',
      os.path.basename(file_path)
    )
    full_file_path = os.path.join(root_dir, new_file_path)

    # 确保文件存在
    if os.path.isfile(full_file_path):
        # 复制文件到目标目录
        shutil.copy(full_file_path, target_dir)
        print(f'Copied {full_file_path} to {target_dir}')
    else:
        print(f'File not found: {full_file_path}')