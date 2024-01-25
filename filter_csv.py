import csv


def filter_csv(input_file, output_file):
    filtered_rows = 0

    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入表头
        header = next(reader)
        writer.writerow(header)

        # 过滤行
        for row in reader:
            if row and row[0]:  # 如果第一个元素不为空
                writer.writerow(row)
            else:
                filtered_rows += 1

    return filtered_rows


if __name__ == "__main__":
    input_filename = "train.csv"  # 替换为您的输入文件名
    output_filename = "train_filtered.csv"  # 替换为您的输出文件名

    filtered_rows_count = filter_csv(input_filename, output_filename)

    print(f"Filtered CSV saved to {output_filename}")
    print(f"Number of filtered rows: {filtered_rows_count}")
