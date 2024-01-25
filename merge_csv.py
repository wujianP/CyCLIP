import os
import csv

# 获取当前目录下所有的CSV文件
csv_files = [file for file in os.listdir() if file.endswith('.csv')]

# 如果没有找到CSV文件，输出提示信息并结束程序
if not csv_files:
    print("在当前目录下未找到任何CSV文件。")
    exit()

# 创建一个空的列表，用于存储所有CSV文件的数据
combined_data = []

# 创建一个字典，用于记录每个CSV文件的行数
rows_before_merge = {}

# 逐个读取CSV文件并将数据存储到列表中
for idx, file in enumerate(csv_files):
    file_path = os.path.join(os.getcwd(), file)
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # 如果不是第一个文件，跳过头部
        if idx > 0:
            next(reader, None)
        # 将每一行数据追加到列表中
        rows = list(reader)
        combined_data.extend(rows)
        rows_before_merge[file] = len(rows)

# 将合并后的数据写入一个新的CSV文件
output_file = 'train_all.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入CSV文件的头部
    writer.writerow(combined_data[0])
    # 写入数据
    writer.writerows(combined_data[1:])

# 计算融合后新文件的行数
rows_after_merge = len(combined_data)

# 输出每个CSV文件的行数和融合后新文件的行数
print("每个CSV文件的行数:")
for file, rows in rows_before_merge.items():
    print(f"{file}: {rows} 行")

print("\n融合后新文件的行数:", rows_after_merge)
print("CSV文件合并完成，输出文件名:", output_file)
