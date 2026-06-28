import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from scipy.signal import find_peaks

from utils.utils import concordance_correlation_coefficient, logmsg

class RNNBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, n_l, is_bidirectional=False, is_lstm=False):
        # in_dim 2*m
        # mid_dim n
        # out_dim 2*m
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.n_l = n_l

        if is_lstm:
            self.gru = nn.LSTM(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=self.n_l, batch_first=True,
                               bidirectional=is_bidirectional)
        else:
            self.gru = nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=self.n_l, batch_first=True,
                              bidirectional=is_bidirectional)

        # ==========修改1：GRU输出后增加层归一化，约束RNN输出范围==========
        self.ln_out = nn.LayerNorm(self.out_dim * (2 if is_bidirectional else 1))

    def forward(self, x):
        # x (batch,in_dim,n)
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # (batch,n,in_dim)
        x = self.gru(x)[0]  # (batch,n,out_dim/double_out)
        # RNN输出归一化
        x = self.ln_out(x)
        x = x.permute(0, 2, 1)  # (batch,hidden_dim,n)
        return x


class FilterBlock(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.gen_data_config = gen_data_config
        self.model_config = model_config
        self.train_config = train_config
        self.input_dim = sum(gen_data_config['modal_dim'])
        self.w = gen_data_config['w']
        self.l = gen_data_config['l']
        self.seq_len = self.w + 2 * self.l
        self.k = model_config.get('em_d', 256)
        self.n = model_config.get('n', 180)
        self.is_bidirectional = model_config.get('is_bid', True)
        self.is_lstm = model_config.get('is_lstm', False)
        self.n_l = model_config.get('n_l', 2)

        # 添加可学习参数 A
        self.A = nn.Parameter(torch.zeros(self.input_dim, self.seq_len))
        nn.init.xavier_uniform_(self.A)

        self.Ann1 = RNNBlock(self.input_dim, self.n, self.k, self.n_l, self.is_bidirectional, self.is_lstm)
        self.Xnn2 = RNNBlock(self.input_dim, self.seq_len, self.k, self.n_l, self.is_bidirectional, self.is_lstm)
        if self.is_bidirectional:
            self.k = 2 * self.k
        self.Bnn3 = RNNBlock(self.k, self.n, self.k, self.n_l, is_lstm=self.is_lstm)

        self.W_v = nn.Linear(self.seq_len, self.n)
        self.W_k = nn.Linear(self.seq_len, self.n)
        self.W_q = nn.Linear(self.seq_len, self.n)
        self.W_b = nn.Linear(self.k, self.input_dim)
        self.scale = self.n ** 0.5

        # ==========修改2：P输出层归一化，约束B@X_orig后的数值范围==========
        self.p_ln = nn.LayerNorm(self.seq_len)
        # 矩阵相乘缩放超参数，固定常数可按需调整 10/20
        self.mat_scale = np.sqrt(self.input_dim)

    def forward(self, X):
        X_orig = X.clone()
        A_expanded = self.A.unsqueeze(0).expand(X.size(0), -1, -1)
        A = self.Ann1(A_expanded)  # (batch,k,n)
        X = self.Xnn2(X)  # (batch,k,seq_len)

        A_v = self.W_v(A)
        A_k = self.W_k(A)
        X_q = self.W_q(X)

        # 注意力
        scores = torch.matmul(X_q, A_k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        B = torch.matmul(attn_weights, A_v)  # (batch,k,n)

        B = self.Bnn3(B)
        B = B.permute(0, 2, 1)  # (batch,n,k)
        B = self.W_b(B)  # (batch,n,input_dim)

        # ==========修改3：矩阵相乘增加缩放，防止数值爆炸==========
        P = (B @ X_orig) / self.mat_scale  # 关键：除以维度开根号缩放
        P = self.p_ln(P)  # 输出归一化 (batch,n,seq_len)

        return P


class PairFilterNet6(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.n = model_config['n']
        self.FilterBlock_t = FilterBlock(gen_data_config, model_config, train_config)
        self.FilterBlock_p = FilterBlock(gen_data_config, model_config, train_config)

        # ==========修改4：fc前加LayerNorm + fc输出可学习缩放系数==========
        embed_dim = 2 * self.n
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        # 可学习缩放参数，初始值设0.1，限制fc输出不会过大
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, X_t, X_p):
        P_t = self.FilterBlock_t(X_t)
        P_p = self.FilterBlock_p(X_p)

        P_t = P_t.permute(0, 2, 1)  # (batch,seq_len,n)
        P_p = P_p.permute(0, 2, 1)

        embed_P = torch.cat((P_t, P_p), dim=-1)
        embed_P = self.embed_ln(embed_P)  # 拼接特征归一化

        # fc输出乘可学习缩放，大幅缩小输入tanh的值域
        P = self.fc(embed_P) * self.out_scale  # (batch,seq_len,1)
        P = P.squeeze(-1)

        # 优化：如果仍饱和，可选替换tanh为softsign：x/(1+|x|)，值域[-1,1]但不容易饱和
        P = torch.sigmoid(P)
        # P = nn.functional.softsign(P)

        return P, embed_P


def pairFilterNet6_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name):
    # 训练循环
    best_ccc = 0.0
    not_improve_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            signals_t = batch['datas_t'].to(device)
            signals_p = batch['datas_p'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(signals_t,signals_p)
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
                signals_t = batch['datas_t'].to(device)
                signals_p = batch['datas_p'].to(device)
                labels = batch['labels'].to(device)

                outputs, _ = model(signals_t, signals_p)
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
                if group_name == None:
                    torch.save(model.state_dict(), folder_path / f'best_model.pth')
                else:
                    torch.save(model.state_dict(), folder_path / f'{group_name}_best_model.pth')
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    logmsg(folder_path / 'printLogs.txt',f'Early stopped')
                    break

            logmsg(folder_path / 'printLogs.txt',f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val CCC={val_ccc:.4f}, Best CCC={best_ccc:.4f}')

def pairFilterNet6_eval(model, val_loader, device, validate_config,gen_data_config, folder_path):
    #TODO 对val进行可视化，对test进行推理，得到的score看看val和test分布符不符合trian

    # 验证阶段
    model.eval()
    outputs_list = []
    labels_list = []
    embed_P_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signals_t = batch['datas_t'].to(device)
            signals_p = batch['datas_p'].to(device)
            labels = batch['labels'].to(device)

            outputs, embed_P = model(signals_t, signals_p)

            embed_P = embed_P.reshape(-1, embed_P.shape[-1]).detach().cpu().numpy()
            embed_P_list.append(embed_P)
            outputs = outputs[:,
                      val_loader.dataset.l:val_loader.dataset.l + val_loader.dataset.w].flatten().detach().cpu().numpy()
            outputs_list.append(outputs)
            labels = labels[:,
                     val_loader.dataset.l:val_loader.dataset.l + val_loader.dataset.w].flatten().detach().cpu().numpy()
            labels_list.append(labels)


        outputs_list = [item for sublist in outputs_list for item in sublist][:len(val_loader.dataset.labels)]
        labels_list = [item for sublist in labels_list for item in sublist][:len(val_loader.dataset.labels)]
        embed_P_list = np.concatenate(embed_P_list, axis=0)

        val_ccc = concordance_correlation_coefficient(labels_list, outputs_list)
        val_ccc = None if np.isnan(val_ccc) else val_ccc


        # 创建图形
        plt.figure(figsize=(36, 12))
        # 绘制第一条线（标签数据）- 蓝色
        plt.plot(labels_list, color='blue', label='Labels')
        # 绘制第二条线（输出数据）- 红色
        plt.plot(outputs_list, color='red', label='Outputs')

        # 添加图表元素
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Comparison of Labels and Outputs')
        plt.legend()  # 显示图例
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图像（支持多种格式）
        plt.savefig(folder_path / 'comparison.png', dpi=600, bbox_inches='tight')  # PNG格式
        plt.show()


        #保存outputs_list和labels_list
        file_path = folder_path / 'outputs_data.npz'


        np.savez(
            file_path,
            labels=labels_list,
            outputs=outputs_list
        )

    return outputs_list, labels_list, val_ccc, embed_P_list