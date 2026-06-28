import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.utils import concordance_correlation_coefficient, logmsg


class ImprovedRNNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 is_bidirectional=False, is_lstm=False, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        self.dropout = dropout

        # 使用LayerNorm替代BatchNorm，更适合时序数据
        self.norm = nn.LayerNorm(in_dim)

        # RNN层
        rnn_kwargs = {
            'input_size': in_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': is_bidirectional,
            'dropout': dropout if num_layers > 1 else 0
        }

        if is_lstm:
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.GRU(**rnn_kwargs)

        # 投影层，处理双向RNN输出维度翻倍问题
        rnn_out_dim = hidden_dim * 2 if is_bidirectional else hidden_dim
        self.proj = nn.Linear(rnn_out_dim, out_dim)

        # 残差连接投影（当输入输出维度不同时）
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # 激活函数
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, feature_dim) - 统一使用时序优先的维度顺序
        residual = self.residual_proj(x)

        # 归一化
        x = self.norm(x)

        # RNN前向传播
        x, _ = self.rnn(x)

        # 投影到目标维度
        x = self.proj(x)

        # 激活和dropout
        x = self.activation(x)
        x = self.dropout_layer(x)

        # 残差连接
        x = x + residual

        return x


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, seq_len_q, seq_len_kv, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = feature_dim ** 0.5

        # 注意力投影层
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        # 输出投影
        self.W_o = nn.Linear(feature_dim, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: (batch, seq_len_q, feature_dim)
        # key/value: (batch, seq_len_kv, feature_dim)

        # 投影
        q = self.W_q(query)  # (batch, seq_len_q, d)
        k = self.W_k(key)  # (batch, seq_len_kv, d)
        v = self.W_v(value)  # (batch, seq_len_kv, d)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, seq_len_q, seq_len_kv)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, v)  # (batch, seq_len_q, d)

        # 输出投影
        output = self.W_o(output)

        # 残差连接
        output = output + query

        return output, attn_weights


class ImprovedFilterBlock(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.gen_data_config = gen_data_config
        self.model_config = model_config
        self.train_config = train_config

        # 基础参数
        self.input_dim = sum(gen_data_config['modal_dim'])
        self.w = gen_data_config['w']
        self.l = gen_data_config['l']
        self.seq_len = self.w + 2 * self.l
        self.embedding_dim = model_config.get('em_d', 256)
        self.hidden_dim = model_config.get('n', 180)
        self.is_bidirectional = model_config.get('is_bid', True)
        self.is_lstm = model_config.get('is_lstm', False)
        self.num_layers = model_config.get('n_l', 2)
        self.dropout = model_config.get('dropout', 0.1)

        # 动态参数生成器（替代原固定参数A）
        self.param_generator = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.input_dim * self.seq_len)
        )

        # RNN编码器
        self.a_encoder = ImprovedRNNBlock(
            in_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embedding_dim,
            num_layers=self.num_layers,
            is_bidirectional=self.is_bidirectional,
            is_lstm=self.is_lstm,
            dropout=self.dropout
        )

        self.x_encoder = ImprovedRNNBlock(
            in_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embedding_dim,
            num_layers=self.num_layers,
            is_bidirectional=self.is_bidirectional,
            is_lstm=self.is_lstm,
            dropout=self.dropout
        )

        # 时序注意力层
        self.temporal_attn = TemporalAttention(
            feature_dim=self.embedding_dim,
            seq_len_q=self.seq_len,
            seq_len_kv=self.hidden_dim,
            dropout=self.dropout
        )

        # 后续RNN处理
        self.post_rnn = ImprovedRNNBlock(
            in_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embedding_dim,
            num_layers=self.num_layers,
            is_bidirectional=self.is_bidirectional,
            is_lstm=self.is_lstm,
            dropout=self.dropout
        )

        # 输出投影
        self.output_proj = nn.Linear(self.embedding_dim, self.input_dim)

        # 最终时序输出投影
        self.final_proj = nn.Linear(self.input_dim, 1)

    def forward(self, X):
        # X: (batch, feature_dim, seq_len) - 保持原输入格式
        batch_size = X.size(0)

        # 转置为时序优先格式: (batch, seq_len, feature_dim)
        X = X.permute(0, 2, 1)

        # 生成动态参数A: (batch, seq_len, feature_dim)
        global_feat = torch.mean(X, dim=1)  # (batch, feature_dim)
        A_flat = self.param_generator(global_feat)  # (batch, input_dim * seq_len)
        A = A_flat.view(batch_size, self.seq_len, self.input_dim)  # (batch, seq_len, input_dim)

        # 编码
        A_encoded = self.a_encoder(A)  # (batch, seq_len, embedding_dim)
        X_encoded = self.x_encoder(X)  # (batch, seq_len, embedding_dim)

        # 时序注意力: X作为query, A作为key/value
        attn_output, attn_weights = self.temporal_attn(X_encoded, A_encoded, A_encoded)

        # 后续处理
        B = self.post_rnn(attn_output)  # (batch, seq_len, embedding_dim)



        # 投影回输入维度
        B = self.output_proj(B)  # (batch, seq_len, input_dim)

        # 生成最终时序输出
        output = self.final_proj(B)  # (batch, seq_len, 1)
        output = output.squeeze(-1)  # (batch, seq_len)

        # # 应用sigmoid确保输出在[0,1]范围内
        # output = torch.sigmoid(output)

        return output, attn_weights


class PairFilterNet4(nn.Module):
    def __init__(self, gen_data_config, model_config, train_config):
        super().__init__()
        self.gen_data_config = gen_data_config
        self.model_config = model_config
        self.train_config = train_config

        # 两个FilterBlock共享权重（可选，根据任务决定）
        self.filter_block_1 = ImprovedFilterBlock(gen_data_config, model_config, train_config)
        self.filter_block_2 = ImprovedFilterBlock(gen_data_config, model_config, train_config)

        # 融合层
        self.w = gen_data_config['w']
        self.l = gen_data_config['l']
        self.seq_len = self.w + 2 * self.l

        self.fusion = nn.Sequential(
            nn.Linear(4 * self.seq_len, self.seq_len),
            nn.LayerNorm(self.seq_len),
            nn.GELU(),
            nn.Dropout(model_config.get('dropout', 0.1)),
            nn.Linear(self.seq_len, self.seq_len)
        )

    def forward(self, X_1, X_2, X_3, X_4):
        # X_t, X_p: (batch, feature_dim, seq_len)

        # 分别处理两个输入
        output_1, attn_1 = self.filter_block_1(X_1)  # (batch, seq_len)
        output_2, attn_2 = self.filter_block_2(X_2)  # (batch, seq_len)
        output_3, attn_3 = self.filter_block_2(X_3)  # (batch, seq_len)
        output_4, attn_4 = self.filter_block_2(X_4)  # (batch, seq_len)

        # 融合两个输出
        embed_P = torch.cat([output_1, output_2, output_3, output_4], dim=-1)  # (batch, 2*seq_len)
        final_output = self.fusion(embed_P)  # (batch, seq_len)

        # 再次应用sigmoid确保输出在[0,1]范围内
        final_output = torch.sigmoid(final_output)

        return final_output, embed_P #, (attn_t, attn_p)




def pairFilterNet4_train(num_epochs,model,optimizer,train_loader,val_loader, criterion, device,writer,folder_path,scheduler,early_stop,group_name):
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

                outputs, _= model(signals_1, signals_2,signals_3,signals_4)
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

def pairFilterNet4_eval(model, val_loader, device, validate_config,gen_data_config, folder_path):
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


# 配置示例
if __name__ == "__main__":
    # 示例配置
    gen_data_config = {
        'modal_dim': [10, 10],  # 两个模态，每个10维
        'w': 100,  # 窗口大小
        'l': 20  # 左右上下文长度
    }

    model_config = {
        'em_d': 64,  # 嵌入维度
        'n': 128,  # RNN隐藏层维度
        'is_bid': True,  # 是否双向
        'is_lstm': False,  # 是否使用LSTM
        'n_l': 2,  # RNN层数
        'dropout': 0.1  # dropout概率
    }

    train_config = {
        'lr': 1e-3,
        'batch_size': 32
    }

    # 创建模型
    model = PairFilterNet4(gen_data_config, model_config, train_config)

    # 测试前向传播
    batch_size = 8
    input_dim = sum(gen_data_config['modal_dim'])
    seq_len = gen_data_config['w'] + 2 * gen_data_config['l']

    X_t = torch.randn(batch_size, input_dim, seq_len)
    X_p = torch.randn(batch_size, input_dim, seq_len)

    output, (attn_t, attn_p) = model(X_t, X_p)

    print(f"输入形状: X_t={X_t.shape}, X_p={X_p.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"注意力权重形状: attn_t={attn_t.shape}")
