import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from train import train_main
from utils.utils import logmsg, concordance_correlation_coefficient
from validate import val_main

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# rng = np.random.default_rng(seed=42)  #默认使用 PCG64生成器2^128次才会序列重复

def run_main(root_path, gen_data_config, train_config, model_config,validate_config,only_run_validate):
    # 创建文件夹
    folder_name = ""
    for key, value in gen_data_config.items():
        # 添加键值对到文件名
        if key != "modal_dim" and key != "tr_dir" and key != "va_dir":
            folder_name += f"{key}_{value}_"
    for key, value in train_config.items():
        # 添加键值对到文件名
        folder_name += f"{key}_{value}_"
    for key, value in model_config.items():
        # 添加键值对到文件名
        folder_name += f"{key}_{value}_"
    # for key, value in validate_config.items():
    #     # 添加键值对到文件名
    #     folder_name += f"{key}_{value}_"

    folder_path = root_path / folder_name.rstrip("_")
    os.makedirs(folder_path,exist_ok=True)


    # 将各个config保存为yaml文件备份
    with open(folder_path / "gen_data_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            gen_data_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    with open(folder_path / "train_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            train_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    with open(folder_path / "model_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            model_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    with open(folder_path / "validate_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(
            validate_config,
            file,
            allow_unicode=True,  # 支持中文
            default_flow_style=False,  # 使用块样式（更易读）
            indent=4,  # 缩进4个空格
            sort_keys=False  # 保持键的原始顺序
        )

    groups = gen_data_config['grps']

    # 训练模型 并保存best
    scalar_list = []
    for group_name in groups:
        logmsg(folder_path / 'printLogs.txt', f"processing group:{group_name}")
        scalar = train_main(gen_data_config, train_config, model_config, folder_path, group_name, only_run_validate)
        scalar_list.append(scalar)

    #验证模型
    outputs_list = []
    labels_list = []
    embed_P_list = []
    for group_name, scaler in zip(groups, scalar_list):
        outputs, labels, val_f1,accuracy,precision,recall,embed_P = val_main(gen_data_config, train_config,model_config,validate_config,folder_path, group_name, scaler)
        logmsg(folder_path / 'printLogs.txt', f"{group_name} eval f1: {val_f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")
        outputs_list.append(outputs)
        labels_list.append(labels)
        embed_P_list.append(embed_P)

    outputs_list = [x for sublst in outputs_list for x in sublst]
    labels_list = [x for sublst in labels_list for x in sublst]

    # 将列表转换为 numpy 数组
    y_true = np.array(labels_list)
    y_pred = np.array(outputs_list)

    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # 多分类常用 macro/micro/weighted
    recall = recall_score(y_true, y_pred, average='macro')
    val_f1 = f1_score(y_true, y_pred, average='macro')
    embed_P = np.concatenate(embed_P_list, axis=0)
    logmsg(folder_path / 'printLogs.txt', f"total eval f1: {val_f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")

    return embed_P, val_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Script')
    parser.add_argument('--root_path', type=str, help='Path for the root folder')
    parser.add_argument('--gen_data_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--train_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--model_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--validate_config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--only_run_validate', action='store_true', default=False)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    gen_data_config_path = Path(args.gen_data_config_path)
    train_config_path = Path(args.train_config_path)
    model_config_path = Path(args.model_config_path)
    validate_config_path = Path(args.validate_config_path)
    only_run_validate = args.only_run_validate

    with open(gen_data_config_path, 'r') as file:
        gen_data_config = yaml.safe_load(file)

    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load(file)

    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    with open(validate_config_path, 'r') as file:
        validate_config = yaml.safe_load(file)

    run_main(root_path, gen_data_config, train_config, model_config,validate_config,only_run_validate)