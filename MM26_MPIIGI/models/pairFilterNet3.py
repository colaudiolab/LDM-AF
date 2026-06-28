import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.utils import concordance_correlation_coefficient, logmsg


class RNNBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, n_l,
                 is_bidirectional=False, is_lstm=False, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.n_l = n_l
        self.is_bidirectional = is_bidirectional
        self.is_lstm = is_lstm

        # 使用 LayerNorm 替代 BatchNorm1d
        self.norm = nn.LayerNorm(in_dim)

        rnn_cls = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=out_dim,
            num_layers=n_l,
            batch_first=True,
            bidirectional=is_bidirectional,
            dropout=dropout if n_l > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, in_dim, seq_len)
        x = x.transpose(1, 2)  # -> (batch, seq_len, in_dim)
        x = self.norm(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # -> (batch, out_dim, seq_len)
        return x


class FilterBlock(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.input_dim = sum(gen_data_config['modal_dim'])
        self.w = gen_data_config['w']
        self.l = gen_data_config['l']
        self.seq_len = self.w + 2 * self.l
        self.k = model_config.get('em_d', 256)
        self.n = model_config.get('n', 180)
        self.is_bidirectional = model_config.get('is_bid', True)
        self.is_lstm = model_config.get('is_lstm', False)
        self.n_l = model_config.get('n_l', 2)
        self.dropout_rate = model_config.get('dropout', 0.1)

        # 可学习参数 A
        self.A = nn.Parameter(torch.zeros(self.input_dim, self.seq_len))
        nn.init.xavier_uniform_(self.A)

        # RNN 模块
        self.Ann1 = RNNBlock(self.input_dim, self.n, self.k, self.n_l,
                             self.is_bidirectional, self.is_lstm, self.dropout_rate)
        self.Xnn2 = RNNBlock(self.input_dim, self.seq_len, self.k, self.n_l,
                             self.is_bidirectional, self.is_lstm, self.dropout_rate)

        if self.is_bidirectional:
            self.k *= 2

        self.Bnn3 = RNNBlock(self.k, self.n, self.k, self.n_l,
                             self.is_bidirectional, self.is_lstm, self.dropout_rate)

        # 注意力投影层
        self.W_v = nn.Linear(self.seq_len, self.n)
        self.W_k = nn.Linear(self.seq_len, self.n)
        self.W_q = nn.Linear(self.seq_len, self.n)
        self.W_b = nn.Linear(2 * self.k, self.input_dim)

        # 注意力归一化
        self.attn_norm = nn.LayerNorm(self.n)
        self.scale = self.n ** 0.5

    def forward(self, X):
        batch_size = X.size(0)
        A_expanded = self.A.unsqueeze(0).expand(batch_size, -1, -1)

        # 通过 RNN 提取特征
        A = self.Ann1(A_expanded)  # (batch, k, n)
        X = self.Xnn2(X)  # (batch, k, seq_len)

        # 注意力机制
        A_v = self.W_v(A)  # (batch, k, n)
        A_k = self.W_k(A)  # (batch, k, n)
        X_q = self.W_q(X)  # (batch, k, n)

        scores = torch.matmul(X_q, A_k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        B = torch.matmul(attn_weights, A_v)  # (batch, k, n)

        # 残差连接 + 层归一化
        B = self.attn_norm(B + X_q)

        # 进一步特征变换
        B = self.Bnn3(B)  # (batch, k, n)
        B = B.transpose(1, 2)  # (batch, n, k)
        B = self.W_b(B)  # (batch, n, input_dim)

        # 与原始 A 结合
        P = torch.bmm(B, A_expanded)  # (batch, n, seq_len)
        return P


class PairFilterNet3(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.n = model_config['n']
        self.FilterBlock_1 = FilterBlock(gen_data_config, model_config, train_config)
        self.FilterBlock_2 = FilterBlock(gen_data_config, model_config, train_config)

        # 输出头：全局平均池化 + 全连接层
        self.fc = nn.Sequential(
            nn.Linear(4 * self.n, self.n),
            nn.ReLU(),
            nn.Dropout(model_config.get('dropout', 0.1)),
            nn.Linear(self.n, 1),
            nn.Sigmoid()  # 输出范围 [0,1]
        )

    def forward(self, X_1, X_2,X_3,X_4):
        P_1 = self.FilterBlock_1(X_1)  # (batch, n, seq_len)
        P_2 = self.FilterBlock_2(X_2)  # (batch, n, seq_len)
        P_3 = self.FilterBlock_2(X_3)  # (batch, n, seq_len)
        P_4 = self.FilterBlock_2(X_4)  # (batch, n, seq_len)  #0.2568  #0.3944

        P_1 = P_1.permute(0, 2, 1)
        P_2 = P_2.permute(0, 2, 1)
        P_3 = P_3.permute(0, 2, 1)
        P_4 = P_4.permute(0, 2, 1)
        # 拼接并输出
        embed_P = torch.cat([P_1, P_2, P_3, P_4], dim=-1)  # (batch, 2*n)


        return self.fc(embed_P).squeeze(-1), embed_P  # (batch,)




def pairFilterNet3_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name):
    # 训练循环
    best_ccc = 0.0
    not_improve_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            signals_1 = batch['datas_1'].to(device)
            signals_2 = batch['datas_2'].to(device)
            signals_3 = batch['datas_3'].to(device)
            signals_4 = batch['datas_4'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs,_ = model(signals_1,signals_2,signals_3,signals_4)
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
                signals_1 = batch['datas_1'].to(device)
                signals_2 = batch['datas_2'].to(device)
                signals_3 = batch['datas_3'].to(device)
                signals_4 = batch['datas_4'].to(device)
                labels = batch['labels'].to(device)

                outputs,_ = model(signals_1, signals_2,signals_3,signals_4)
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
                torch.save(model.state_dict(), folder_path / f'{group_name}_best_model.pth')
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    logmsg(folder_path / 'printLogs.txt',f'Early stopped')
                    break

            logmsg(folder_path / 'printLogs.txt',f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val CCC={val_ccc:.4f}, Best CCC={best_ccc:.4f}')

def pairFilterNet3_eval(model, val_loader, device, validate_config,gen_data_config,folder_path):
    #TODO 对val进行可视化，对test进行推理，得到的score看看val和test分布符不符合trian

    # 验证阶段
    model.eval()
    outputs_list = []
    labels_list = []
    embed_P_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signals_1 = batch['datas_1'].to(device)
            signals_2 = batch['datas_2'].to(device)
            signals_3 = batch['datas_3'].to(device)
            signals_4 = batch['datas_4'].to(device)
            labels = batch['labels'].to(device)

            outputs, embed_P = model(signals_1, signals_2,signals_3,signals_4)

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

        if val_ccc != None:
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

            if not file_path.exists():
                np.savez(
                    file_path,
                    labels=labels_list,
                    outputs=outputs_list
                )

    return outputs_list, labels_list, val_ccc, embed_P_list