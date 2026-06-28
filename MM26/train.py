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
from models.filterNet1 import FilterNet1, filterNet1_train
from models.filterNet2 import FilterNet2, filterNet2_train
from models.filterNet3 import FilterNet3, filterNet3_train
from models.filterNet4 import FilterNet4, filterNet4_train
from models.filterNet5 import FilterNet5, filterNet5_train
from models.filterNet6 import FilterNet6, filterNet6_train
from models.pairFilterNet1 import PairFilterNet1, pairFilterNet1_train
from models.pairFilterNet2 import PairFilterNet2, pairFilterNet2_train
from models.pairFilterNet3 import PairFilterNet3, pairFilterNet3_train
from models.pairFilterNet4 import PairFilterNet4, pairFilterNet4_train
from models.pairFilterNet5 import PairFilterNet5, pairFilterNet5_train
from models.pairFilterNet6 import PairFilterNet6, pairFilterNet6_train
from utils.loss import CCCLoss, SmoothCCCLoss, SmoothMSELoss

from sklearn.preprocessing import MinMaxScaler

from utils.utils import logmsg

seed = 12
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def train_main(gen_data_config, train_config,model_config,folder_path, group_name, only_run_validate = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logmsg(folder_path / 'printLogs.txt', "Run training...")

    # train_dataset = MM26Dataset(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['tr_dir'], w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['s'])
    # val_dataset = MM26Dataset(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['va_dir'], w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['s'])

    train_dataset = MM26Dataset_pair(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['tr_dir'], group_name = group_name, w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['s'])
    val_dataset = MM26Dataset_pair(modal = gen_data_config['modal'], modal_dim = gen_data_config['modal_dim'], dir_path  = gen_data_config['va_dir'], group_name = group_name, w = gen_data_config['w'], l = gen_data_config['l'], s = gen_data_config['w'])

    scaler = MinMaxScaler()
    scaler.partial_fit(train_dataset.datas_t)
    scaler.partial_fit(val_dataset.datas_t)

    if only_run_validate:
        return scaler
    train_dataset.pre_data_process(scaler)
    val_dataset.pre_data_process(scaler)


    batch_size = train_config['bs']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_epochs = train_config['n_eps']
    writer = SummaryWriter(log_dir=folder_path / 'logs')

    #加载loss
    if train_config['loss'] == 'CCC':
        criterion = CCCLoss()
    elif train_config['loss'] == 'MSE':
        criterion = nn.MSELoss()
    elif train_config['loss'] == 'Softmax':
        criterion = nn.CrossEntropyLoss()
    elif train_config['loss'] == 'SCCC':
        criterion = SmoothCCCLoss(train_config['alpha'])
    elif train_config['loss'] == 'SMSE':
        criterion = SmoothMSELoss(train_config['alpha'])
    else:
        return -1


    early_stop = 20
    step_size = 100
    gamma = 0.5

    if train_config['model'] == 'FilterNet1':
        model = FilterNet1(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        filterNet1_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop)
    elif train_config['model'] == 'FilterNet2':
        model = FilterNet2(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        filterNet2_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop)
    elif train_config['model'] == 'FilterNet3':
        model = FilterNet3(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        filterNet3_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop)
    elif train_config['model'] == 'FilterNet4':
        model = FilterNet4(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        filterNet4_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop)
    elif train_config['model'] == 'FilterNet5':
        model = FilterNet5(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        filterNet5_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop)
    elif train_config['model'] == 'FilterNet6':
        model = FilterNet6(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        filterNet6_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop)
    elif train_config['model'] == 'PairFilterNet1':
        model = PairFilterNet1(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        pairFilterNet1_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name)
    elif train_config['model'] == 'PairFilterNet2':
        model = PairFilterNet2(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        pairFilterNet2_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop, group_name)
    elif train_config['model'] == 'PairFilterNet3':
        model = PairFilterNet3(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        pairFilterNet3_train(num_epochs, model, optimizer, train_loader, val_loader, criterion, device, writer, folder_path, scheduler, early_stop, group_name)
    elif train_config['model'] == 'PairFilterNet4':
        model = PairFilterNet4(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        pairFilterNet4_train(num_epochs, model, optimizer, train_loader, val_loader, criterion, device, writer, folder_path, scheduler, early_stop, group_name)
    elif train_config['model'] == 'PairFilterNet5':
        model = PairFilterNet5(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        pairFilterNet5_train(num_epochs, model, optimizer, train_loader, val_loader, criterion, device, writer, folder_path, scheduler, early_stop, group_name)
    elif train_config['model'] == 'PairFilterNet6':
        model = PairFilterNet6(gen_data_config, model_config, train_config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logmsg(folder_path / 'printLogs.txt', f"总参数量: {total_params:,}")
        logmsg(folder_path / 'printLogs.txt', f"可训练参数量: {trainable_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        pairFilterNet6_train(num_epochs, model, optimizer, train_loader, val_loader, criterion, device, writer, folder_path, scheduler, early_stop, group_name)
    return scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script')
    parser.add_argument('--yaml_config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    yaml_config_path = Path(args.yaml_config_path)

    with open(yaml_config_path, 'r') as file:
        train_config = yaml.safe_load(file)  # 推荐使用 safe_load 避免安全风险

    train_main(train_config,None,None)






