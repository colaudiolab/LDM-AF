import os
import pandas as pd
import numpy as np


def process_folder(root_dir):
    """
    处理指定文件夹及其子文件夹中的所有.social_engagement.annotation.csv文件
    忽略空值进行判断

    参数:
        root_dir (str): 要处理的文件夹路径
    """

    # 用于存储所有文件的结果
    all_results = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.task_engagement.annotation.csv'):
                file_path = os.path.join(root, file)

                try:
                    # 读取CSV文件（只有一列，无表头）
                    df = pd.read_csv(file_path, header=None)

                    # 获取数据列（假设是第一列）
                    data = df.iloc[:, 0].values

                    # 将空字符串、None等转换为NaN，然后过滤掉NaN
                    # 同时处理可能的字符串空值
                    cleaned_data = []
                    for item in data:
                        if pd.isna(item) or str(item).strip() == '':
                            continue  # 跳过空值
                        cleaned_data.append(item)

                    # 如果清理后数据长度小于128，跳过该文件
                    if len(cleaned_data) < 256:
                        print(f"警告: {file_path} 清理后数据长度不足128，已跳过")
                        continue

                    # 以128为长度划分数据
                    chunks = []
                    for i in range(0, len(cleaned_data), 256):
                        chunk = cleaned_data[i:i + 256]
                        # 只保留长度为128的块
                        if len(chunk) == 256:
                            chunks.append(chunk)

                    # 统计每个chunk内数据是否全部相同（忽略空值）
                    same_count = 0
                    total_chunks = len(chunks)

                    for chunk in chunks:
                        # 进一步清理chunk内的空值（虽然前面已经清理过，但以防万一）
                        valid_values = [v for v in chunk if not (pd.isna(v) or str(v).strip() == '')]

                        # 如果没有有效值，跳过这个chunk
                        if len(valid_values) == 0:
                            continue

                        # 检查valid_values中所有元素是否相同
                        first_value = valid_values[0]
                        if all(v == first_value for v in valid_values):
                            same_count += 1

                    # 计算相同数据的占比
                    ratio = same_count / total_chunks if total_chunks > 0 else 0

                    # 存储结果
                    result = {
                        'file': file_path,
                        'original_length': len(data),
                        'cleaned_length': len(cleaned_data),
                        'total_chunks': total_chunks,
                        'same_chunks': same_count,
                        'ratio': ratio
                    }
                    all_results.append(result)

                    # 打印当前文件的结果
                    print(f"文件: {file}")
                    print(f"  原始数据长度: {len(data)}")
                    print(f"  清理后数据长度: {len(cleaned_data)}")
                    print(f"  总块数: {total_chunks}")
                    print(f"  相同块数: {same_count}")
                    print(f"  相同占比: {ratio:.4f}")
                    print("-" * 50)

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")

    return all_results


if __name__ == "__main__":
    # 设置要处理的文件夹路径
    folder_path = "/media/dxy/e4d05437-15dc-4450-9866-ff9fd650d89e/PInSoRo/cr/train"

    # 验证路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 路径 '{folder_path}' 不存在")
    else:
        # 处理文件夹
        results = process_folder(folder_path)

        # 可选：保存结果到CSV文件
        if results:
            output_df = pd.DataFrame(results)
            output_file = "analysis_results.csv"
            output_df.to_csv(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")

            # 计算总体统计
            total_files = len(results)
            avg_ratio = sum(r['ratio'] for r in results) / total_files
            print(f"\n总体统计:")
            print(f"  处理文件总数: {total_files}")
            print(f"  平均相同占比: {avg_ratio:.4f}")

