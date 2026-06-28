import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from scipy.signal import find_peaks

from utils.utils import concordance_correlation_coefficient, logmsg


class RNNBlock(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,is_bidirectional = False):
        #in_dim 2*m
        #mid_dim n
        #out_dim 2*m
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.gru = nn.GRU(input_size = self.in_dim, hidden_size = self.out_dim,num_layers = 2, batch_first=True,bidirectional=is_bidirectional)

    def forward(self,x):
        # x (batch,2*m,n)

        x = self.bn(x) # out x (batch,2*m,n)
        x = x.permute(0, 2, 1) # out x (batch,n,2*m)
        x = self.gru(x)[0] # out x (batch,n,2*m)
        x = x.permute(0, 2, 1) #out x (batch,2*m,n)

        return x


class FilterNet1(nn.Module):
    def __init__(self,gen_data_config, model_config, train_config):
        #in_dim 2*m
        #mid_dim n
        #out_dim 2*m
        super().__init__()
        self.gen_data_config = gen_data_config
        self.model_config = model_config
        self.train_config = train_config
        self.input_dim = sum(gen_data_config['modal_dim'])
        self.w = gen_data_config['w']
        self.l = gen_data_config['l']
        self.seq_len = self.w + 2 * self.l
        self.k = model_config['em_d']
        self.n = model_config['n']
        self.is_bidirectional = model_config['is_bid']

        # 添加可学习参数 A
        self.A = nn.Parameter(torch.zeros(self.input_dim, self.seq_len))  # 零初始化
        nn.init.xavier_uniform_(self.A)  # Xavier初始化

        # self.m = gen_data_config['m']
        # degree_lower = gen_data_config['deg_l']
        # degree_upper = gen_data_config['deg_u']
        # degree_precision = gen_data_config['deg_p']
        # self.n = len(list(np.arange(degree_lower, degree_upper, degree_precision)))
        # self.snapshots = gen_data_config['spshots']
        # self.k = model_config['em_d']
        # self.is_bidirectional = model_config['is_bid']
        self.Ann1 = RNNBlock(self.input_dim, self.n, self.k, self.is_bidirectional)
        self.Xnn2 = RNNBlock(self.input_dim, self.seq_len, self.k, self.is_bidirectional)
        if self.is_bidirectional:
            self.k = 2 * self.k
        self.Bnn3 = RNNBlock(self.k, self.n, self.k)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.n, 2*self.n),
        #     nn.BatchNorm1d(2*self.n),
        #     nn.ReLU(),
        #     nn.Linear(2*self.n, self.n),
        # )
        self.W_v = nn.Linear(self.seq_len, self.n)
        self.W_k = nn.Linear(self.seq_len, self.n)
        self.W_q = nn.Linear(self.seq_len, self.n)
        self.W_b = nn.Linear(self.k, self.input_dim)
        self.scale = self.n ** 0.5

    def forward(self, X):
        X_orig = X.clone()
        A_expanded = self.A.unsqueeze(0).expand(X.size(0), -1, -1)
        A = self.Ann1(A_expanded) #out A (batch,k,n)
        X = self.Xnn2(X) #out Y (batch,k,snapshots)

        A_v = self.W_v(A) #out A_v (batch,k,n)
        A_k = self.W_k(A) #out A_k (batch,k,n)
        X_q = self.W_q(X) #out Y_q (batch,k,n)

        # 计算注意力分数
        scores = torch.matmul(X_q, A_k.transpose(-2, -1)) / self.scale #out scores (batch,k,k)
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        B = torch.matmul(attn_weights, A_v)  # out B (batch_size, k, n)

        B = self.Bnn3(B) #out B (batch_size, k, n)
        B = B.permute(0, 2, 1) #out B (batch_size, n, k)
        B = self.W_b(B) #out B (batch_size, n, 2 * m)

        P = B @ A_expanded # out P (batch_size, n, snapshots)
        P = torch.abs(P)**2
        P = torch.mean(P, dim=1)#out P (batch_size, n)

        # P = self.fc(P)  # out P (batch_size, n)

        P = torch.tanh(P)

        return P


def filterNet1_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop):
    # 训练循环
    best_ccc = 0.0
    not_improve_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            signals = batch['datas_t'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        outputs_list = []
        labels_list = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                signals = batch['datas_t'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                outputs = outputs[:,val_loader.dataset.l:val_loader.dataset.l+val_loader.dataset.w].flatten().detach().cpu().numpy()
                outputs_list.append(outputs)
                labels = labels[:, val_loader.dataset.l:val_loader.dataset.l+val_loader.dataset.w].flatten().detach().cpu().numpy()
                labels_list.append(labels)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

            outputs_list = [item for sublist in outputs_list for item in sublist][:len(val_loader.dataset.labels)]
            labels_list = [item for sublist in labels_list for item in sublist][:len(val_loader.dataset.labels)]
            val_ccc = concordance_correlation_coefficient(labels_list, outputs_list)

            writer.add_scalar('CCC/val', val_ccc, epoch)

            # 保存最佳模型
            if val_ccc > best_ccc:
                best_ccc = val_ccc
                torch.save(model.state_dict(), folder_path / 'best_model.pth')
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    logmsg(folder_path / 'printLogs.txt',f'Early stopped')
                    break

            logmsg(folder_path / 'printLogs.txt',f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val CCC={val_ccc:.4f}, Best CCC={best_ccc:.4f}')

def filterNet1_eval(model, val_loader, device, validate_config,gen_data_config):
    #TODO 对val进行可视化，对test进行推理，得到的score看看val和test分布符不符合trian

    # model.eval()
    # doas_list = []
    # label_list = []
    # doas_known_d_list = []
    # with torch.no_grad():
    #     for batch in tqdm(val_loader):
    #         signal = batch['signal'].to(device)
    #         labels = batch['label'].numpy()
    #         steeringMatrix_A = batch['steeringMatrix_A'].to(device)
    #
    #         outputs = model(steeringMatrix_A, signal)
    #
    #         spectrums = torch.atanh(torch.clamp(outputs, -1+1e-6, 1-1e-6)).detach().cpu().numpy()
    #         for index in range(batch['label'].shape[0]):
    #             spectrum = spectrums[index]
    #             spectrum_mean = np.mean(spectrum)
    #             spectrum_std = np.std(spectrum)
    #             spectrum_max = np.max(spectrum)
    #             # 滤波寻找峰值
    #             doas = find_peaks(spectrum,height=spectrum_mean+validate_config['hght']*spectrum_std,prominence=spectrum_max*validate_config['prmice'])[0]
    #             doas_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
    #             label_list.append((np.where(labels[index] == 1)[0] * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
    #
    #             doas = find_peaks(spectrum)[0]
    #             d = len(np.where(labels[index] == 1)[0])
    #             doas = doas[np.argsort(spectrum[doas])[-d:]]
    #             doas_known_d_list.append((doas * gen_data_config['deg_p'] - gen_data_config['deg_u']) * np.pi / 180)
    #
    #         # outputs = outputs.detach().cpu().numpy()
    #         # for index in range(batch['label'].shape[0]):
    #         #     doas_list.append((np.where(outputs[index] > validate_config['td'])[0] - gen_data_config['deg_u']) * np.pi / 180)
    #         #     label_list.append((np.where(labels[index] == 1)[0]-gen_data_config['deg_u']) * np.pi / 180)
    #
    #
    # return doas_list, label_list, doas_known_d_list

    return 0