import os
from pathlib import Path

import numpy as np
import yaml

from utils.utils import concordance_correlation_coefficient

folder_root_path = '/home/dxy/桌面/MM26_pretrained/'
folder_root_path = Path(folder_root_path)
# 只获取直接子文件夹，不递归
folders = [p.name for p in folder_root_path.iterdir() if p.is_dir()]


csv_save_path_root = '/home/dxy/桌面/MM26_提交/best'
NoXi_or_J = 'NoXi_J'

folder_list = []
val_ck_list = []
w16_list = []
w32_list = []
w64_list = []
w128_list = []
w256_list = []
for folder in folders:
    genYamlPath = folder_root_path / folder / "gen_data_config.yaml"
    if not os.path.exists(genYamlPath):
        print(f"{genYamlPath} is not exist \n\n\n\n\n\n")
        continue
    with open(genYamlPath, 'r') as file:
        gen_data_config = yaml.safe_load(file)
    val_dir = gen_data_config['va_dir']
    if isinstance(val_dir, list):
        val_dir = val_dir[0]

    if NoXi_or_J == 'NoXi_J':
        if 'NoXi_J' not in val_dir:
            continue
    else:
        if 'NoXi_J' in val_dir:
            continue

    npz_path = folder_root_path / folder / 'outputs_data.npz'
    if not npz_path.exists():
        print(f'\n\n\n\n{folder} not have outputs_data.npz\n\n\n')
        continue

    npz_data = np.load(npz_path)
    labels = npz_data['labels']
    outputs = npz_data['outputs']


    # if labels.shape[0] < 300000:
    #     continue


    y_true = np.array(labels)
    y_pred = np.array(outputs)

    val_ck = concordance_correlation_coefficient(y_true, y_pred)

    if val_ck < 0.1:
        continue

    w_len = gen_data_config['w']
    if w_len == 16:
        w16_list.append(val_ck)

    elif w_len == 32:
        w32_list.append(val_ck)

    elif w_len == 64:
        w64_list.append(val_ck)

    elif w_len == 128:
        w128_list.append(val_ck)

    elif w_len == 256:
        w256_list.append(val_ck)

print(NoXi_or_J)
print("w16", np.mean(w16_list), np.std(w16_list))
print("w32", np.mean(w32_list), np.std(w32_list))
print("w64", np.mean(w64_list), np.std(w64_list))
print("w128", np.mean(w128_list), np.std(w128_list))
print("w256", np.mean(w256_list), np.std(w256_list))