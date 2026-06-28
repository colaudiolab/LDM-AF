#有一个predict.txt来指明哪些模型去推理
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import MM26Dataset_pair
from models.filterNet1 import FilterNet1
from models.filterNet2 import FilterNet2
from models.filterNet3 import FilterNet3
from models.filterNet4 import FilterNet4
from models.filterNet5 import FilterNet5
from models.filterNet6 import FilterNet6
from models.pairFilterNet1 import PairFilterNet1
from models.pairFilterNet2 import PairFilterNet2
from models.pairFilterNet3 import PairFilterNet3
from models.pairFilterNet4 import PairFilterNet4
from models.pairFilterNet5 import PairFilterNet5
from models.pairFilterNet6 import PairFilterNet6
from train import train_main

predict_folders_path = '/root/autodl-tmp/dxy/MM26/analysis.txt'

with open(predict_folders_path, 'r', encoding='utf-8') as f:
    folder_paths = [line.strip() for line in f if line.strip()]  # 去除空行和首尾空格

folder_paths = folder_paths[:]

#根据不同的模型(train.yaml)类型初始化模型加载模型权重 #特殊处理：根据group分开加载将两波人分开处理的模型  没有.pth文件跳过
for predict_folder in folder_paths:
    # if "modal_['egemapsv2', 'w2vbert2', 'clip', 'openface2', 'openpose']_dataset_NoXi_l_8_w_256_s_32_lr_0.0001_bs_32_n_eps_100_model_PairFilterNet1_loss_MSE_em_d_256_is_bid_True_n_180_is_lstm_False_n_l_2" not in predict_folder:
    #     continue

    print(f"处理路径: {predict_folder}")

    if not os.path.exists(predict_folder):
        print(f"{predict_folder} is not exist")
        continue

    genYamlPath = Path(predict_folder) / "gen_data_config.yaml"
    modelYamlPath = Path(predict_folder) / "model_config.yaml"
    trainYamlPath = Path(predict_folder) / "train_config.yaml"
    valYamlPath = Path(predict_folder) / "validate_config.yaml"

    if not os.path.exists(genYamlPath):
        print(f"{genYamlPath} is not exist")
        continue

    if not os.path.exists(modelYamlPath):
        print(f"{modelYamlPath} is not exist")
        continue

    if not os.path.exists(trainYamlPath):
        print(f"{trainYamlPath} is not exist")
        continue

    if not os.path.exists(valYamlPath):
        print(f"{valYamlPath} is not exist")
        continue

    # 没有.pth文件
    if not any(Path(predict_folder).glob("*.pth")):
        print(f"model pth file is not exist")
        continue

    with open(genYamlPath, 'r') as file:
        gen_data_config = yaml.safe_load(file)

    with open(trainYamlPath, 'r') as file:
        train_config = yaml.safe_load(file)

    with open(modelYamlPath, 'r') as file:
        model_config = yaml.safe_load(file)

    with open(valYamlPath, 'r') as file:
        validate_config = yaml.safe_load(file)


    #对group的处理
    is_groups = gen_data_config.get('grps')
    if is_groups == None:
        groups = [None]
    else:
        groups = is_groups

    predict_folder = Path(predict_folder)
    # 训练模型 并保存best
    scalar_list = []
    for group_name in groups:
        scalar = train_main(gen_data_config, train_config, model_config, predict_folder, group_name, only_run_validate=True)
        scalar_list.append(scalar)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #加载模型
    outputs_list = []
    labels_list = []
    embed_P_list = []
    for group_name, scaler in zip(groups, scalar_list):
        if train_config['model'] == 'FilterNet1':
            model = FilterNet1(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'FilterNet2':
            model = FilterNet2(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'FilterNet3':
            model = FilterNet3(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'FilterNet4':
            model = FilterNet4(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'FilterNet5':
            model = FilterNet5(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'FilterNet6':
            model = FilterNet6(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'PairFilterNet1':
            model = PairFilterNet1(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'PairFilterNet2':
            model = PairFilterNet2(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'PairFilterNet3':
            model = PairFilterNet3(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'PairFilterNet4':
            model = PairFilterNet4(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'PairFilterNet5':
            model = PairFilterNet5(gen_data_config, model_config, train_config).to(device)
        elif train_config['model'] == 'PairFilterNet6':
            model = PairFilterNet6(gen_data_config, model_config, train_config).to(device)


        #过滤没有pth的
        try:
            if group_name == None:
                model.load_state_dict(torch.load(Path(predict_folder) / f'best_model.pth', map_location=device))
            else:
                model.load_state_dict(torch.load(Path(predict_folder) / f'{group_name}_best_model.pth', map_location=device))
        except Exception as e:
            # 获取完整的错误堆栈信息
            error_msg = f"Error occurred: {str(e)}\n"
            error_trace = traceback.format_exc()

            # 写入 error.txt 文件
            with open("./error.txt", "a", encoding="utf-8") as f:
                f.write(f"Folder_path: {Path(predict_folder).name}\n")
                f.write(error_msg)
                f.write("Traceback:\n")
                f.write(error_trace)
                f.write("\n" + "=" * 60 + "\n")

            print(f"Error captured and saved to error.txt: {e}")
            continue



        val_dir = gen_data_config['va_dir']
        if isinstance(val_dir, list):
            val_dir = val_dir[0]
        if 'NoXi_J' in val_dir:
            te_dir_path = ['/root/autodl-tmp/dxy/DataSet/NoXiDataSet/NoXi_J_test/test']
            # te_dir_path = ['/root/autodl-tmp/dxy/DataSet/NoXiDataSet/NoXi_J_trainval/train']
        else:
            te_dir_path = ['/root/autodl-tmp/dxy/DataSet/NoXiDataSet/NoXi_test/test_additional','/root/autodl-tmp/dxy/DataSet/NoXiDataSet/NoXi_test/test_base']
            # te_dir_path = ['/root/autodl-tmp/dxy/DataSet/NoXiDataSet/NoXi_trainval/train']


        #开始推理
        if group_name == None:
            group_name = ['novice','expert']
        for te_dir in te_dir_path:
            sub_te_dirs = [str(p) for p in Path(te_dir).iterdir() if p.is_dir()]
            for sub_te_dir in sub_te_dirs:
                for group_name_single in group_name:
                    group_name_single = [group_name_single]

                    p = Path(sub_te_dir)
                    sub_path = p.parent.parent.name + "/" +  p.parent.name + "/" + p.name
                    csv_save_path = Path(predict_folder) / sub_path / f"{group_name_single[0]}.engagement.prediction.csv"
                    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
                    npy_save_path = Path(
                        predict_folder) / sub_path / f"{group_name_single[0]}.engagement.prediction.npy"

                    if npy_save_path.exists():
                        continue

                    val_dataset = MM26Dataset_pair(modal=gen_data_config['modal'], modal_dim=gen_data_config['modal_dim'],
                                                   dir_path=sub_te_dir, group_name=group_name_single,
                                                   w=gen_data_config['w'], l=gen_data_config['l'], s=gen_data_config['w'])

                    val_dataset.pre_data_process(scaler)

                    batch_size = train_config['bs']
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                    model.eval()
                    outputs_list = []
                    embed_P_list = []
                    with torch.no_grad():
                        for batch in tqdm(val_loader):
                            signals_t = batch['datas_t'].to(device)
                            signals_p = batch['datas_p'].to(device)

                            if train_config['model'] == 'FilterNet1' or train_config['model'] == 'FilterNet2' or train_config['model'] == 'FilterNet3' or train_config['model'] == 'FilterNet4' or train_config['model'] == 'FilterNet5' or train_config['model'] == 'FilterNet6':
                                outputs, embed_P = model(signals_t)
                            else:
                                outputs, embed_P = model(signals_t,signals_p)


                            if train_config['model'] == 'FilterNet6':
                                outputs = outputs.flatten().detach().cpu().numpy()
                            else:
                                outputs = outputs[:,
                                val_loader.dataset.l:val_loader.dataset.l + val_loader.dataset.w].flatten().detach().cpu().numpy()
                            outputs_list.append(outputs)

                            embed_P = embed_P.reshape(-1, embed_P.shape[-1]).detach().cpu().numpy()
                            embed_P_list.append(embed_P)

                    outputs_list = [item for sublist in outputs_list for item in sublist][
                        :len(val_loader.dataset.labels)]
                    df = pd.DataFrame(outputs_list)
                    df.to_csv(csv_save_path, index=False, header=False)

                    embed_P_list = np.concatenate(embed_P_list, axis=0)
                    np.save(npy_save_path, embed_P_list)




    # csvSavePath = Path(predict_folder) / "printLogs.txt"

#根据不同的模型加载训练和验证数据获得 scalar

#加载每个session测试数据、对每个session组成dataloader(专门写一个用于测试的dataloader),用scalar处理和其他nan处

#模型推理(不足长度的循环取再截取)

#将结果保存到csv文件，目录放在推理模型的目录下面，用文件夹来指明推理的哪个数据集哪个session(如果该csv文件存在则跳过)

