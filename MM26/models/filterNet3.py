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


class FeatureWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))

    def forward(self, x):
        # x: (bs, num_class, d)
        # self.W: (1, num_class, d)
        x = (self.W * x).sum(-1)
        # point-wise mul and sum in the last dim, x will be (bs, num_class)

        if self.bias:
            x = x + self.b
            # broadcast when adding bias, x will be (bs, num_class)

        return x


class FilterNet3(nn.Module):
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
        self.fc = FeatureWiseLinear(self.seq_len, self.n)
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


        # P = torch.abs(P)**2
        # P = torch.mean(P, dim=1)#out P (batch_size, n)

        embed_P = P.permute(0, 2, 1)
        P = self.fc(embed_P)  # out P (batch_size, n)

        P = torch.tanh(P)

        return P, embed_P


def filterNet3_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop):
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
            outputs,_ = model(signals)
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

                outputs,_ = model(signals)
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

def filterNet3_eval(model, val_loader, device, validate_config,gen_data_config, folder_path):
    #TODO 对val进行可视化，对test进行推理，得到的score看看val和test分布符不符合trian

    model.eval()
    outputs_list = []
    labels_list = []
    embed_P_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            signals = batch['datas_t'].to(device)
            labels = batch['labels'].to(device)

            outputs, embed_P = model(signals)

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