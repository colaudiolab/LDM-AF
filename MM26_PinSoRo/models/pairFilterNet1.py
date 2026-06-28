import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from scipy.signal import find_peaks

from utils.utils import concordance_correlation_coefficient, logmsg

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report


class RNNBlock(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,n_l,is_bidirectional = False, is_lstm = False):
        #in_dim 2*m
        #mid_dim n
        #out_dim 2*m
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.bn = nn.BatchNorm1d(self.in_dim)
        self.n_l = n_l

        if is_lstm:
            self.gru = nn.LSTM(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=self.n_l, batch_first=True, bidirectional=is_bidirectional)
        else:
            self.gru = nn.GRU(input_size = self.in_dim, hidden_size = self.out_dim,num_layers = self.n_l, batch_first=True,bidirectional=is_bidirectional)

    def forward(self,x):
        # x (batch,2*m,n)

        x = self.bn(x) # out x (batch,2*m,n)
        x = x.permute(0, 2, 1) # out x (batch,n,2*m)
        x = self.gru(x)[0] # out x (batch,n,2*m)
        x = x.permute(0, 2, 1) #out x (batch,2*m,n)

        return x

class FilterBlock(nn.Module):
    def __init__(self,gen_data_config, model_config, train_config):
        super().__init__()
        # in_dim 2*m
        # mid_dim n
        # out_dim 2*m
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
        self.Ann1 = RNNBlock(self.input_dim, self.n, self.k, self.n_l, self.is_bidirectional, self.is_lstm)
        self.Xnn2 = RNNBlock(self.input_dim, self.seq_len, self.k, self.n_l, self.is_bidirectional, self.is_lstm)
        if self.is_bidirectional:
            self.k = 2 * self.k
        self.Bnn3 = RNNBlock(self.k, self.n, self.k, self.n_l, is_lstm = self.is_lstm)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.n, 2*self.n),
        #     nn.BatchNorm1d(2*self.n),
        #     nn.ReLU(),
        #     nn.Linear(2*self.n, self.n),
        # )
        # self.fc = nn.Linear(self.n, 1)
        self.W_v = nn.Linear(self.seq_len, self.n)
        self.W_k = nn.Linear(self.seq_len, self.n)
        self.W_q = nn.Linear(self.seq_len, self.n)
        self.W_b = nn.Linear(self.k, self.input_dim)
        self.scale = self.n ** 0.5

    def forward(self, X):
        X_orig = X.clone()
        A_expanded = self.A.unsqueeze(0).expand(X.size(0), -1, -1)
        A = self.Ann1(A_expanded)  # out A (batch,k,n)
        X = self.Xnn2(X)  # out Y (batch,k,snapshots)

        A_v = self.W_v(A)  # out A_v (batch,k,n)
        A_k = self.W_k(A)  # out A_k (batch,k,n)
        X_q = self.W_q(X)  # out Y_q (batch,k,n)

        # 计算注意力分数
        scores = torch.matmul(X_q, A_k.transpose(-2, -1)) / self.scale  # out scores (batch,k,k)
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        B = torch.matmul(attn_weights, A_v)  # out B (batch_size, k, n)

        B = self.Bnn3(B)  # out B (batch_size, k, n)
        B = B.permute(0, 2, 1)  # out B (batch_size, n, k)
        B = self.W_b(B)  # out B (batch_size, n, 2 * m)

        P = B @ A_expanded  # out P (batch_size, n, snapshots)

        # P = torch.abs(P)**2
        # P = torch.mean(P, dim=1)#out P (batch_size, n)

        # P = P.permute(0, 2, 1)
        # P = self.fc(P)  # out P (batch_size, n)
        # P = P.squeeze(-1)
        #
        # P = torch.tanh(P)

        return P


class PairFilterNet1(nn.Module):
    def __init__(self,gen_data_config, model_config, train_config):
        #in_dim 2*m
        #mid_dim n
        #out_dim 2*m
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
            nn.Linear(self.w, 4 if self.task == "task"  else 5),
        )




    def forward(self, X_t,X_p,X_e):
        P_t = self.FilterBlock_t(X_t)
        P_p = self.FilterBlock_p(X_p)
        P_e = self.FilterBlock_p(X_e)

        P_t = P_t.permute(0, 2, 1)
        P_p = P_p.permute(0, 2, 1)
        P_e = P_e.permute(0, 2, 1)

        embed_P = torch.cat((P_t, P_p, P_e), dim=-1)
        P = self.fc(embed_P)  # out P (batch_size, n)
        P = P.squeeze(-1)

        P = self.classfier(P)

        return P, embed_P


def pairFilterNet1_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name):
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


def pairFilterNet1_eval(model, val_loader, device, validate_config,gen_data_config,folder_path):
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