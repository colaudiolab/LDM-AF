import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import MM26Dataset, MM26Dataset_pair
from models.filterNet1 import FilterNet1, filterNet1_train, filterNet1_eval
from models.filterNet2 import FilterNet2, filterNet2_train, filterNet2_eval
from models.filterNet3 import FilterNet3, filterNet3_train, filterNet3_eval
from models.filterNet4 import FilterNet4, filterNet4_train, filterNet4_eval
from models.filterNet5 import FilterNet5, filterNet5_train, filterNet5_eval
from models.filterNet6 import FilterNet6, filterNet6_train, filterNet6_eval
from models.pairFilterNet1 import PairFilterNet1, pairFilterNet1_train, pairFilterNet1_eval
from models.pairFilterNet2 import PairFilterNet2, pairFilterNet2_train, pairFilterNet2_eval
from models.pairFilterNet3 import PairFilterNet3, pairFilterNet3_eval
from models.pairFilterNet4 import PairFilterNet4, pairFilterNet4_eval
from models.pairFilterNet5 import PairFilterNet5, pairFilterNet5_eval
from utils.loss import CCCLoss, SmoothCCCLoss, SmoothMSELoss

from sklearn.preprocessing import MinMaxScaler

from utils.utils import logmsg

seed = 12
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def val_main(gen_data_config, train_config,model_config,validate_config,folder_path, group_name, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logmsg(folder_path / 'printLogs.txt', "Run eval...")

    # train_dataset = MM26Dataset(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['tr_dir'], w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['s'])
    # val_dataset = MM26Dataset(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['va_dir'], w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['s'])

    val_dataset = MM26Dataset_pair(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['va_dir'], group_name = group_name, w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['w'])

    val_dataset.pre_data_process(scaler)


    batch_size = train_config['bs']
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    if train_config['model'] == 'FilterNet1':
        model = FilterNet1(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = filterNet1_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'FilterNet2':
        model = FilterNet2(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = filterNet2_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'FilterNet3':
        model = FilterNet3(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = filterNet3_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'FilterNet4':
        model = FilterNet4(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = filterNet4_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'FilterNet5':
        model = FilterNet5(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = filterNet5_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'FilterNet6':
        model = FilterNet6(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = filterNet6_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'PairFilterNet1':
        model = PairFilterNet1(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = pairFilterNet1_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'PairFilterNet2':
        model = PairFilterNet2(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = pairFilterNet2_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'PairFilterNet3':
        model = PairFilterNet3(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = pairFilterNet3_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'PairFilterNet4':
        model = PairFilterNet4(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = pairFilterNet4_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)
    elif train_config['model'] == 'PairFilterNet5':
        model = PairFilterNet5(gen_data_config, model_config, train_config).to(device)
        if group_name == None:
            model.load_state_dict(torch.load(folder_path / f'best_model.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(folder_path / f'{group_name}_best_model.pth', map_location=device))
        outputs_list, labels_list, val_ccc, embed_P = pairFilterNet5_eval(model, val_loader, device, validate_config, gen_data_config, folder_path)

    return outputs_list, labels_list, val_ccc, embed_P


if __name__ == '__main__':
    print()





