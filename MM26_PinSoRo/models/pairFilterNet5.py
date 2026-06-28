import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
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
        self.dropout = dropout

        # 时序用LayerNorm优于BatchNorm1d
        self.norm = nn.LayerNorm(in_dim)

        if is_lstm:
            self.gru = nn.LSTM(input_size=self.in_dim,
                               hidden_size=self.out_dim,
                               num_layers=self.n_l,
                               batch_first=True,
                               bidirectional=is_bidirectional,
                               dropout=dropout if n_l > 1 else 0)
        else:
            self.gru = nn.GRU(input_size=self.in_dim,
                              hidden_size=self.out_dim,
                              num_layers=self.n_l,
                              batch_first=True,
                              bidirectional=is_bidirectional,
                              dropout=dropout if n_l > 1 else 0)

        # 双向维度适配
        self.rnn_out_dim = self.out_dim * 2 if is_bidirectional else self.out_dim
        self.proj = nn.Linear(self.rnn_out_dim, self.out_dim)

        # 残差适配
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, in_dim, seq_len)
        residual = self.res_proj(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)  # (batch, seq_len, in_dim)
        x = self.norm(x)
        x, _ = self.gru(x)
        x = self.proj(x)
        x = self.drop(x)
        x = x + residual  # 残差
        return x.permute(0, 2, 1)  # (batch, out_dim, seq_len)


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
        self.k = model_config['em_d']
        self.n = model_config['n']
        self.is_bidirectional = model_config['is_bid']
        self.is_lstm = model_config['is_lstm']
        self.n_l = model_config['n_l']
        self.dropout = model_config.get('dropout', 0.1)

        # 保留你原可学习矩阵A，初始化优化
        self.A = nn.Parameter(torch.randn(self.input_dim, self.seq_len))
        nn.init.xavier_uniform_(self.A)

        self.Ann1 = RNNBlock(self.input_dim, self.n, self.k,
                             self.n_l, self.is_bidirectional, self.is_lstm, self.dropout)
        self.Xnn2 = RNNBlock(self.input_dim, self.seq_len, self.k,
                             self.n_l, self.is_bidirectional, self.is_lstm, self.dropout)

        # if self.is_bidirectional:
        #     self.k = 2 * self.k

        self.Bnn3 = RNNBlock(self.k, self.n, self.k,
                             self.n_l, is_lstm=self.is_lstm, dropout=self.dropout)

        # 注意力映射
        self.W_v = nn.Linear(self.seq_len, self.n)
        self.W_k = nn.Linear(self.seq_len, self.n)
        self.W_q = nn.Linear(self.seq_len, self.n)
        self.W_b = nn.Linear(self.k, self.input_dim)
        self.scale = self.n ** 0.5

        # 最终时序映射 + Sigmoid约束0~1
        self.out_proj = nn.Linear(self.seq_len, self.seq_len)

    def forward(self, X):
        X_orig = X.clone()
        # 广播A: (batch, input_dim, seq_len)
        A_expanded = self.A.unsqueeze(0).expand(X.size(0), -1, -1)

        A = self.Ann1(A_expanded)  # (batch, k, n)
        X = self.Xnn2(X)  # (batch, k, seq_len)

        A_v = self.W_v(A)
        A_k = self.W_k(A)
        X_q = self.W_q(X)

        # 注意力计算不变
        scores = torch.matmul(X_q, A_k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        B = torch.matmul(attn_weights, A_v)  # (batch, k, n)

        B = self.Bnn3(B)  # (batch, k, n)
        B = B.permute(0, 2, 1)  # (batch, n, k)
        B = self.W_b(B)  # (batch, n, input_dim)

        # ========== 完全保留你核心：B @ A_expanded ==========
        P = B @ A_expanded  # (batch, n, seq_len)

        # 时序维度映射 + 约束到0~1
        P = P.permute(0, 2, 1)  # (batch, seq_len, n)
        # P = self.out_proj(P)  # (batch, seq_len, seq_len)
        P = P.mean(dim=-1)  # (batch, seq_len)

        # # 输出严格映射到 [0,1]
        # P = torch.sigmoid(P)

        return P


class PairFilterNet5(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.n = model_config['n']
        self.FilterBlock_t = FilterBlock(gen_data_config, model_config, train_config)
        self.FilterBlock_p = FilterBlock(gen_data_config, model_config, train_config)
        self.FilterBlock_e = FilterBlock(gen_data_config, model_config, train_config)
        self.seq_len = gen_data_config['w'] + 2 * gen_data_config['l']
        self.task = gen_data_config['task']

        # 时序融合输出
        self.fc = nn.Sequential(
            nn.Linear(3 * self.seq_len, self.seq_len),
            nn.LayerNorm(self.seq_len),
            nn.GELU(),
            nn.Dropout(model_config.get('dropout', 0.1)),
            nn.Linear(self.seq_len, 4 if self.task == "task" else 5)
        )

    def forward(self, X_t, X_p, X_e):
        P_t = self.FilterBlock_t(X_t)  # (batch, seq_len)
        P_p = self.FilterBlock_p(X_p)  # (batch, seq_len)
        P_e = self.FilterBlock_p(X_e)  # (batch, seq_len)

        embed_P = torch.cat((P_t, P_p, P_e), dim=-1)
        P = self.fc(embed_P)

        return P, embed_P


def pairFilterNet5_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name):
    # 训练循环
    best_f1 = 0.0
    not_improve_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            signals_t = batch['datas_t'].to(device)
            signals_p = batch['datas_p'].to(device)
            signals_e = batch['datas_e'].to(device)
            labels = batch['labels'].to(device)
            labels = torch.mode(labels, dim=1).values.long() #处理labels 取众数

            optimizer.zero_grad()
            outputs,_ = model(signals_t,signals_p,signals_e)
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
                signals_e = batch['datas_e'].to(device)
                labels = batch['labels'].to(device)
                labels = torch.mode(labels, dim=1).values.long()  # 处理labels 取众数

                outputs,_ = model(signals_t,signals_p,signals_e)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                outputs = outputs.argmax(dim=1).unsqueeze(1).expand(-1, val_loader.dataset.w).flatten().detach().cpu().numpy()
                outputs_list.append(outputs)
                labels = batch['labels'].to(device)
                labels = labels.flatten().detach().cpu().numpy()
                labels_list.append(labels)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

            outputs_list = [item for sublist in outputs_list for item in sublist][:len(val_loader.dataset.labels)]
            labels_list = [item for sublist in labels_list for item in sublist][:len(val_loader.dataset.labels)]

            # 将列表转换为 numpy 数组
            y_true = np.array(labels_list)
            y_pred = np.array(outputs_list)

            # 基础指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')  # 多分类常用 macro/micro/weighted
            recall = recall_score(y_true, y_pred, average='macro')
            val_f1 = f1_score(y_true, y_pred, average='macro')

            writer.add_scalar('val/F1', val_f1, epoch)
            writer.add_scalar('val/recall', recall, epoch)
            writer.add_scalar('val/precision', precision, epoch)
            writer.add_scalar('val/accuracy', accuracy, epoch)

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), folder_path / f'{group_name}_best_model.pth')
                not_improve_epoch = 0
            else:
                not_improve_epoch += 1
                if not_improve_epoch >= early_stop:
                    logmsg(folder_path / 'printLogs.txt',f'Early stopped')
                    break

            logmsg(folder_path / 'printLogs.txt',f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {val_f1:.4f}, Best F1-score: {best_f1:.4f}')
            logmsg(folder_path / 'printLogs.txt', classification_report(y_true, y_pred))


def pairFilterNet5_eval(model, val_loader, device, validate_config,gen_data_config,folder_path):
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
            signals_e = batch['datas_e'].to(device)
            labels = batch['labels'].to(device)

            outputs, embed_P = model(signals_t, signals_p,signals_e)

            embed_P = embed_P.reshape(-1, embed_P.shape[-1]).detach().cpu().numpy()
            embed_P_list.append(embed_P)

            outputs = outputs.argmax(dim=1).unsqueeze(1).expand(-1,val_loader.dataset.w).flatten().detach().cpu().numpy()
            outputs_list.append(outputs)
            labels = labels.flatten().detach().cpu().numpy()
            labels_list.append(labels)


        outputs_list = [item for sublist in outputs_list for item in sublist][:len(val_loader.dataset.labels)]
        labels_list = [item for sublist in labels_list for item in sublist][:len(val_loader.dataset.labels)]
        embed_P_list = np.concatenate(embed_P_list, axis=0)

        # 将列表转换为 numpy 数组
        y_true = np.array(labels_list)
        y_pred = np.array(outputs_list)

        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')  # 多分类常用 macro/micro/weighted
        recall = recall_score(y_true, y_pred, average='macro')
        val_f1 = f1_score(y_true, y_pred, average='macro')

        # 创建图形
        plt.figure(figsize=(36, 12))

        # 散点图：标签
        plt.scatter(range(len(labels_list)), [x + 0.2 for x in labels_list],
                    color='blue', label='Labels')

        # 散点图：输出
        plt.scatter(range(len(outputs_list)), outputs_list,
                    color='red', label='Outputs')

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
        np.savez(
            folder_path / 'outputs_data.npz',
            labels=labels_list,
            outputs=outputs_list
        )

    return outputs_list, labels_list, val_f1,accuracy,precision,recall,embed_P_list