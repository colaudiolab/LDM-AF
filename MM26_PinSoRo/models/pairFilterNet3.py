import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
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
        self.k = model_config['em_d']
        self.n = model_config['n']
        self.is_bidirectional = model_config['is_bid']
        self.is_lstm = model_config['is_lstm']
        self.n_l = model_config['n_l']
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
        self.l = gen_data_config['l']
        self.w = gen_data_config['w']
        self.task = gen_data_config['task']
        self.FilterBlock_t = FilterBlock(gen_data_config, model_config, train_config)
        self.FilterBlock_p = FilterBlock(gen_data_config, model_config, train_config)
        self.FilterBlock_e = FilterBlock(gen_data_config, model_config, train_config)
        self.fc = nn.Linear(3 * self.n, 1)
        self.classfier = nn.Sequential(
            nn.Linear(2 * self.l + self.w, self.w),
            nn.ReLU(),
            nn.Dropout(model_config.get('dropout', 0.1)),
            nn.Linear(self.w, 4 if self.task == "task" else 5),
        )

    def forward(self, X_t, X_p, X_e):
        P_t = self.FilterBlock_t(X_t)  # (batch, n, seq_len)
        P_p = self.FilterBlock_p(X_p)  # (batch, n, seq_len)
        P_e = self.FilterBlock_p(X_e)

        P_t = P_t.permute(0, 2, 1)
        P_p = P_p.permute(0, 2, 1)
        P_e = P_e.permute(0, 2, 1)
        # 拼接并输出
        embed_P = torch.cat([P_t, P_p, P_e], dim=-1)  # (batch, 2*n)
        P = self.fc(embed_P)
        P = P.squeeze(-1)

        P = self.classfier(P)
        return P, embed_P




def pairFilterNet3_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name):
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


def pairFilterNet3_eval(model, val_loader, device, validate_config,gen_data_config,folder_path):
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