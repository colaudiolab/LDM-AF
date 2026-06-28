import math
import os

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset


class MM26Dataset_pair(Dataset):
    def __init__(self, task, modal, modal_dim, dir_path, group_name,w=32, l=32 , s = 32, is_only_purple = False):
        if not isinstance(dir_path, list):
            dir_path = [dir_path]

        self.modal = modal
        self.modal_dim = modal_dim
        self.dir_path = dir_path
        self.s = s
        self.l = l
        self.w = w
        self.group_name = group_name
        self.task = task
        self.is_only_purple = is_only_purple


        self.datas_t = None
        self.datas_p = None
        self.datas_e = None
        mask_save = None
        row_index = 0
        global_n = None  # 全局样本基准长度，第一次y的长度

        for modality, feat_dim in zip(self.modal, self.modal_dim):
            X_t, X_p, X_e, y, is_test = self.load_data_for_modality(self.dir_path, modality, feat_dim)
            if is_test:
                curr_n = y.shape[0]
                # ==========核心：统一样本维度到global_n==========
                if global_n is None:
                    # 第一次循环：用当前y长度作为全局基准
                    global_n = curr_n
                else:
                    # 非第一次：对齐样本数到global_n
                    if curr_n < global_n:
                        # 样本少：后面补nan
                        pad_n = global_n - curr_n
                        X_t = np.pad(X_t, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        X_p = np.pad(X_p, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        X_e = np.pad(X_e, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        y = np.pad(y, (0, pad_n), constant_values=np.nan)
                    else:
                        # 样本多：截断前global_n行
                        X_t = X_t[:global_n, :]
                        X_p = X_p[:global_n, :]
                        X_e = X_e[:global_n, :]
                        y = y[:global_n]
                # ==============================================


                mask_t = ~np.all(np.isnan(X_t), axis=1)
                mask_p = ~np.all(np.isnan(X_p), axis=1)
                mask_e = ~np.all(np.isnan(X_e), axis=1)
                mask = np.logical_and(mask_t, mask_p)
                mask = np.logical_and(mask, mask_e)
                if mask_save is None:
                    mask_save = mask
                else:
                    mask_save = np.logical_and(mask_save, mask)

                total_feat = sum(self.modal_dim)
                if self.datas_t is None and self.datas_p is None:
                    self.datas_t = np.zeros((global_n, total_feat), dtype=np.float32)
                    self.datas_p = np.zeros((global_n, total_feat), dtype=np.float32)
                    self.datas_e = np.zeros((global_n, total_feat), dtype=np.float32)

                self.datas_t[:,row_index : row_index + feat_dim] = X_t
                self.datas_p[:, row_index: row_index + feat_dim] = X_p
                self.datas_e[:, row_index: row_index + feat_dim] = X_e
                row_index += feat_dim

            else:
                mask_t = ~np.all(np.isnan(X_t), axis=1)
                mask_p = ~np.all(np.isnan(X_p), axis=1)
                mask_e = ~np.all(np.isnan(X_e), axis=1)
                mask = np.logical_and(mask_t, mask_p)
                mask = np.logical_and(mask, mask_e)
                if mask_save is None:
                    mask_save = mask
                else:
                    mask_save = np.logical_and(mask_save, mask)

                if self.datas_t is None and self.datas_p is None:
                    self.datas_t = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)
                    self.datas_p = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)
                    self.datas_e = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)

                self.datas_t[:, row_index: row_index + feat_dim] = X_t
                self.datas_p[:, row_index: row_index + feat_dim] = X_p
                self.datas_e[:, row_index: row_index + feat_dim] = X_e
                row_index += feat_dim

        self.datas_t = self.datas_t[mask_save]
        self.datas_p = self.datas_p[mask_save]
        self.datas_e = self.datas_e[mask_save]
        self.labels = y[mask_save]

    def pre_data_process(self,scaler):
        self.datas_t = scaler.transform(self.datas_t)
        self.datas_p = scaler.transform(self.datas_p)
        self.datas_e = scaler.transform(self.datas_e)
        self.datas_t = np.nan_to_num(self.datas_t)
        self.datas_p = np.nan_to_num(self.datas_p)
        self.datas_e = np.nan_to_num(self.datas_e)
        self.labels = np.nan_to_num(self.labels)


    def y_str2float(self, y_str):
        if self.task == "social":
            map_dict = {
                "solitary": 0,
                "onlooker": 1,
                "parallel": 2,
                "associative": 3,
                "cooperative": 4
            }
        elif self.task == "task":
            map_dict = {
                "goaloriented": 0,
                "aimless": 1,
                "adultseeking": 2,
                "noplay": 3
            }
        else:
            raise ValueError("self.task is not social or task")

        if y_str not in map_dict:
            raise ValueError(f"Unknown label '{y_str}' for task '{self.task}'")

        return map_dict[y_str]


    def walk_and_load(self, root_dir_list, modality, feat_dim):
        stream_map, anno_map = {}, {}
        for root_dir in root_dir_list:
            for dirpath, _, files in os.walk(root_dir):
                session_id = os.path.basename(dirpath)
                for fname in files:
                    key = f"{fname.split('.')[0]};{session_id}"
                    full_path = os.path.join(dirpath, fname)
                    if modality in fname:
                        stream_map[key] = full_path
                    if fname.endswith(f'{self.task}_engagement.annotation.csv'):
                        anno_map[key] = full_path
        X_t, X_p, X_e, y = [], [], [], []
        is_test = False
        if len(anno_map) != 0:
            is_test = False
            for key, anno_path in anno_map.items():
                group = key.split(';')[0]
                # 这里在做group的筛选
                if group not in self.group_name:
                    continue

                stream_path_t = stream_map.get(key)

                if "purple" in key:
                    key = key.replace("purple", "yellow")
                elif "yellow" in key:
                    key = key.replace("yellow", "purple")

                stream_path_p = stream_map.get(key)

                if "purple" in key:
                    key = key.replace("purple", "env")
                elif "yellow" in key:
                    key = key.replace("yellow", "env")

                stream_path_e = stream_map.get(key)

                if not stream_path_t or not stream_path_p:
                    continue

                a_t = np.fromfile(stream_path_t, dtype=np.float32).reshape(-1, feat_dim)
                a_p = np.fromfile(stream_path_p, dtype=np.float32).reshape(-1, feat_dim)
                if not stream_path_e:
                    a_e = np.zeros_like(a_p)
                else:
                    a_e = np.fromfile(stream_path_p, dtype=np.float32).reshape(-1, feat_dim)

                annos = []
                try:
                    with open(anno_path, 'r', encoding='utf-8') as f:
                        annos = [line.strip() for line in f]
                except UnicodeDecodeError:
                    with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                        annos = [line.strip() for line in f]

                n = len(annos)  # min(len(a), len(annos))
                for i in range(n):
                    val = annos[i]
                    if val in ('', 'nan', '-nan(ind)'):
                        continue
                    try:
                        y_val = self.y_str2float(val)
                    except ValueError:
                        print('valueError')
                        continue

                    if i < len(a_t):
                        X_t.append(a_t[i])
                    else:
                        nan_array = np.full(a_t[0].shape, np.nan)
                        X_t.append(nan_array)

                    if i < len(a_p):
                        X_p.append(a_p[i])
                    else:
                        nan_array = np.full(a_p[0].shape, np.nan)
                        X_p.append(nan_array)

                    if i < len(a_e):
                        X_e.append(a_e[i])
                    else:
                        nan_array = np.full(a_e[0].shape, np.nan)
                        X_e.append(nan_array)

                    y.append(y_val)
            return np.array(X_t, dtype=np.float32), np.array(X_p, dtype=np.float32), np.array(X_e,
                                                                                              dtype=np.float32), np.array(
                y, dtype=np.float32),is_test


        else:
            is_test = True
            for key, anno_path in stream_map.items():
                key_origin = key
                group = key.split(';')[0]

                if self.group_name != None:
                    # 这里在做group的筛选
                    if group not in self.group_name:
                        continue

                if self.is_only_purple and group != 'purple':
                    continue

                stream_path_t = stream_map.get(key)

                if "purple" in key:
                    key = key.replace("purple", "yellow")
                elif "yellow" in key:
                    key = key.replace("yellow", "purple")

                stream_path_p = stream_map.get(key)

                if "purple" in key:
                    key = key.replace("purple", "env")
                elif "yellow" in key:
                    key = key.replace("yellow", "env")

                stream_path_e = stream_map.get(key)

                if not stream_path_t or not stream_path_p:
                    continue

                a_t = np.fromfile(stream_path_t, dtype=np.float32).reshape(-1, feat_dim)
                a_p = np.fromfile(stream_path_p, dtype=np.float32).reshape(-1, feat_dim)
                if not stream_path_e:
                    a_e = np.zeros_like(a_p)
                else:
                    a_e = np.fromfile(stream_path_p, dtype=np.float32).reshape(-1, feat_dim)


                annos = []
                anno_path = anno_map.get(key_origin)
                if anno_path == None:
                    annos = [None for i in range(a_t.shape[0])]
                else:
                    try:
                        with open(anno_path, 'r', encoding='utf-8') as f:
                            annos = [line.strip() for line in f]
                    except UnicodeDecodeError:
                        with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                            annos = [line.strip() for line in f]

                n = len(annos) #min(len(a), len(annos))
                for i in range(n):
                    val = annos[i]
                    if val in ('', 'nan', '-nan(ind)'):
                        continue
                    try:
                        y_val = val if val == None else self.y_str2float(val)
                    except ValueError:
                        print('valueError')
                        continue

                    if i < len(a_t):
                        X_t.append(a_t[i])
                    else:
                        nan_array = np.full(a_t[0].shape, np.nan)
                        X_t.append(nan_array)

                    if i < len(a_p):
                        X_p.append(a_p[i])
                    else:
                        nan_array = np.full(a_p[0].shape, np.nan)
                        X_p.append(nan_array)

                    if i < len(a_e):
                        X_e.append(a_e[i])
                    else:
                        nan_array = np.full(a_e[0].shape, np.nan)
                        X_e.append(nan_array)

                    y.append(y_val)

            y = [np.nan if v is None else v for v in y]
            return np.array(X_t, dtype=np.float32), np.array(X_p, dtype=np.float32), np.array(X_e, dtype=np.float32), np.array(y, dtype=np.float32),is_test


    def load_data_for_modality(self, dir_path, modality, feat_dim):
        if modality == 'egemapsv2' or modality == 'egev2':
            modality = ".audio.egemapsv2.stream~"
        elif modality == 'clip':
            modality = ".clip.stream~"
        elif modality == 'openface2' or modality == 'opfa2':
            modality = ".openface2.stream~"
        elif modality == 'openpose' or modality == 'oppo':
            modality = ".openpose.stream~"
        elif modality == 'w2vbert2' or modality == 'w2vb2':
            modality = ".audio.w2vbert2_embeddings.stream~"
        elif modality == 'xlm_roberta' or modality == 'xlm_r':
            modality = ".xlm_roberta_embeddings.stream~"
        elif modality == 'dino':
            modality = ".dino.stream~"
        elif modality == 'swin':
            modality = ".swin.stream~"
        elif modality == 'videomae' or modality == 'vmae':
            modality = ".videomae.stream~"
        elif modality == 'imagebind' or modality == 'imgbi':
            modality = ".imagebind.stream~"
        print("Loading data for modality", modality)
        X_t, X_p,X_e, y, is_test = self.walk_and_load(dir_path, modality, feat_dim)
        return X_t, X_p,X_e, y, is_test

    def __len__(self):
        return math.ceil((len(self.labels) -self.w) / self.s) + 1


    def __getitem__(self, index):
        indices = (np.arange(index * self.s - self.l, index * self.s + self.w + self.l) % len(self.labels))
        data_t = self.datas_t[indices]
        data_p = self.datas_p[indices]
        data_e = self.datas_e[indices]
        labels = self.labels[indices[self.l: self.l + self.w]]

        return {
            'datas_t': torch.from_numpy(data_t).mT,
            'datas_p': torch.from_numpy(data_p).mT,
            'datas_e': torch.from_numpy(data_e).mT,
            'labels': torch.from_numpy(labels),
        }






class MM26Dataset(Dataset):
    def __init__(self, modal, modal_dim, dir_path, w=32, l=32 , s = 32):
        self.modal = modal
        self.modal_dim = modal_dim
        self.dir_path = dir_path
        self.s = s
        self.l = l
        self.w = w


        self.datas = None
        mask_save = None
        row_index = 0
        for modality, feat_dim in zip(self.modal, self.modal_dim):
            X, y = self.load_data_for_modality(self.dir_path, modality, feat_dim)
            mask = ~np.all(np.isnan(X), axis=1)
            if mask_save is None:
                mask_save = mask
            else:
                mask_save = np.logical_and(mask_save, mask)

            if self.datas is None:
                self.datas = np.zeros((len(y),sum(modal_dim)), dtype=np.float32)

            self.datas[:,row_index : row_index + feat_dim] = X
            row_index += feat_dim

        # datas = np.hstack(modalities_data_list)

        self.datas = self.datas[mask_save]
        self.labels = y[mask_save]


    def pre_data_process(self,scaler):
        self.datas = scaler.transform(self.datas)
        self.datas = np.nan_to_num(self.datas)
        self.labels = np.nan_to_num(self.labels)


    def walk_and_load(self, root_dir, modality, feat_dim):
        stream_map, anno_map = {}, {}
        for dirpath, _, files in os.walk(root_dir):
            session_id = os.path.basename(dirpath)
            for fname in files:
                key = f"{fname.split('.')[0]};{session_id}"
                full_path = os.path.join(dirpath, fname)
                if modality in fname:
                    stream_map[key] = full_path
                if fname.endswith('.engagement.annotation.csv'):
                    anno_map[key] = full_path
        X, y = [], []
        for key, anno_path in anno_map.items():
            stream_path = stream_map.get(key)
            if not stream_path:
                continue
            a = np.fromfile(stream_path, dtype=np.float32).reshape(-1, feat_dim)
            annos = []
            try:
                with open(anno_path, 'r', encoding='utf-8') as f:
                    annos = [line.strip() for line in f]
            except UnicodeDecodeError:
                with open(anno_path, 'r', encoding='latin1', errors='ignore') as f:
                    annos = [line.strip() for line in f]
            n = len(annos) #min(len(a), len(annos))
            for i in range(n):
                val = annos[i]
                if val in ('', 'nan', '-nan(ind)'):
                    continue
                try:
                    y_val = float(val)
                except ValueError:
                    print('valueError')
                    continue

                if i < len(a):
                    X.append(a[i])
                else:
                    nan_array = np.full(a[0].shape, np.nan)
                    X.append(nan_array)
                y.append(y_val)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


    def load_data_for_modality(self, dir_path, modality, feat_dim):
        print("Loading data for modality", modality)
        if modality == 'egemapsv2':
            modality = ".audio.egemapsv2.stream~"
        elif modality == 'clip':
            modality = ".clip.stream~"
        elif modality == 'openface2':
            modality = ".openface2.stream~"
        elif modality == 'openpose':
            modality = ".openpose.stream~"
        elif modality == 'openpose':
            modality = ".openpose.stream~"
        elif modality == 'w2vbert2':
            modality = ".audio.w2vbert2_embeddings.stream~"
        elif modality == 'xlm_roberta':
            modality = ".xlm_roberta_embeddings.stream~"
        elif modality == 'dino':
            modality = ".dino.stream~"
        elif modality == 'swin':
            modality = ".swin.stream~"
        elif modality == 'videomae':
            modality = ".videomae.stream~"
        elif modality == 'imagebind':
            modality = ".imagebind.stream~"
        X, y = self.walk_and_load(dir_path, modality, feat_dim)
        return X, y

    def __len__(self):
        return math.ceil((len(self.labels) -self.w) / self.s) + 1


    def __getitem__(self, index):
        indices = (np.arange(index * self.s - self.l, index * self.s + self.w + self.l) % len(self.labels))
        label = self.labels[indices]
        data = self.datas[indices]

        return {
            'datas': torch.from_numpy(data).mT,
            'labels': torch.from_numpy(label),
        }
