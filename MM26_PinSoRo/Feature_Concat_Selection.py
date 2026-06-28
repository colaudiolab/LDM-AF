import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import cohen_kappa_score

folder_root_path = '/media/dxy/e4d05437-15dc-4450-9866-ff9fd650d89e/MM26_pretrained_PinSoRo/experiments'
folder_root_path = Path(folder_root_path)
# 只获取直接子文件夹，不递归
folders = [p.name for p in folder_root_path.iterdir() if p.is_dir()]

cr_cc = 'cc'
task_social = 'task' #social #task

folder_list = []
val_ck_list = []
combine1_list = [] #5 egev2 swin w2v vame clip
combine2_list = [] #5 egev2 w2v clip opfa2 oppo
combine3_list = [] #6 egev2 swin w2v vame clip opfa2
combine4_list = [] #7 egev2 swin w2v vame clip opfa2 oppo
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

    if cr_cc not in val_dir:
        continue

    if task_social != gen_data_config['task']:
        continue


    npz_path = folder_root_path / folder / 'outputs_data.npz'
    if not npz_path.exists():
        print(f'\n\n\n\n{folder} not have outputs_data.npz\n\n\n')
        continue

    npz_data = np.load(npz_path)
    labels = npz_data['labels']
    outputs = npz_data['outputs']


    y_true = np.array(labels)
    y_pred = np.array(outputs)

    val_ck = cohen_kappa_score(y_true, y_pred)

    modal_list = gen_data_config['modal']
    if len(modal_list) == 7:
        combine4_list.append(val_ck)

    elif len(modal_list) == 6:
        combine3_list.append(val_ck)

    elif 'swin' in modal_list:
        combine1_list.append(val_ck)

    else:
        combine2_list.append(val_ck)

print(cr_cc, task_social)
print("combine1", np.mean(combine1_list), np.std(combine1_list))
print("combine2", np.mean(combine2_list), np.std(combine2_list))
print("combine3", np.mean(combine3_list), np.std(combine3_list))
print("combine4", np.mean(combine4_list), np.std(combine4_list))