import math
import os

import numpy as np
import torch
from scipy import stats
from torch.utils.data import Dataset


class MM26Dataset_pair(Dataset):
    def __init__(self, modal, modal_dim, dir_path, group_name,w=32, l=32 , s = 32):
        if not isinstance(dir_path, list):
            dir_path = [dir_path]

        self.modal = modal
        self.modal_dim = modal_dim
        self.dir_path = dir_path
        self.s = s
        self.l = l
        self.w = w
        self.group_name = group_name
        self.is_null = False


        self.datas_1 = None
        self.datas_2 = None
        self.datas_3 = None
        self.datas_4 = None
        mask_save = None
        row_index = 0
        global_n = None
        for modality, feat_dim in zip(self.modal, self.modal_dim):
            X_1, X_2,X_3, X_4, y, is_test = self.load_data_for_modality(self.dir_path, modality, feat_dim)
            if is_test:
                if X_1.size == 0 and X_2.size == 0 and X_3.size == 0 and X_4.size == 0 and y.size == 0:
                    self.is_null = True
                    return
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
                        X_1 = np.pad(X_1, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        X_2 = np.pad(X_2, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        X_3 = np.pad(X_3, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        X_4 = np.pad(X_4, ((0, pad_n), (0, 0)), constant_values=np.nan)
                        y = np.pad(y, (0, pad_n), constant_values=np.nan)
                    else:
                        # 样本多：截断前global_n行
                        X_1 = X_1[:global_n, :]
                        X_2 = X_2[:global_n, :]
                        X_3 = X_3[:global_n, :]
                        X_4 = X_4[:global_n, :]
                        y = y[:global_n]
                # ==============================================


                mask_1 = ~np.all(np.isnan(X_1), axis=1)
                mask_2 = ~np.all(np.isnan(X_2), axis=1)
                mask_3 = ~np.all(np.isnan(X_3), axis=1)
                mask_4 = ~np.all(np.isnan(X_4), axis=1)
                mask = np.logical_and(mask_1, mask_2)
                mask = np.logical_and(mask, mask_3)
                mask = np.logical_and(mask, mask_4)
                if mask_save is None:
                    mask_save = mask
                else:
                    mask_save = np.logical_and(mask_save, mask)

                total_feat = sum(self.modal_dim)
                if self.datas_1 is None and self.datas_2 is None:
                    self.datas_1 = np.zeros((global_n, total_feat), dtype=np.float32)
                    self.datas_2 = np.zeros((global_n, total_feat), dtype=np.float32)
                    self.datas_3 = np.zeros((global_n, total_feat), dtype=np.float32)
                    self.datas_4 = np.zeros((global_n, total_feat), dtype=np.float32)

                self.datas_1[:, row_index : row_index + feat_dim] = X_1
                self.datas_2[:, row_index: row_index + feat_dim] = X_2
                self.datas_3[:, row_index: row_index + feat_dim] = X_3
                self.datas_4[:, row_index: row_index + feat_dim] = X_4
                row_index += feat_dim


            else:
                mask_1 = ~np.all(np.isnan(X_1), axis=1)
                mask_2 = ~np.all(np.isnan(X_2), axis=1)
                mask_3 = ~np.all(np.isnan(X_3), axis=1)
                mask_4 = ~np.all(np.isnan(X_4), axis=1)
                mask = np.logical_and(mask_1, mask_2)
                mask = np.logical_and(mask, mask_3)
                mask = np.logical_and(mask, mask_4)
                if mask_save is None:
                    mask_save = mask
                else:
                    mask_save = np.logical_and(mask_save, mask)

                if self.datas_1 is None and self.datas_2 is None:
                    self.datas_1 = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)
                    self.datas_2 = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)
                    self.datas_3 = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)
                    self.datas_4 = np.zeros((len(y), sum(modal_dim)), dtype=np.float32)

                self.datas_1[:, row_index : row_index + feat_dim] = X_1
                self.datas_2[:, row_index: row_index + feat_dim] = X_2
                self.datas_3[:, row_index: row_index + feat_dim] = X_3
                self.datas_4[:, row_index: row_index + feat_dim] = X_4
                row_index += feat_dim


        self.datas_1 = self.datas_1[mask_save]
        self.datas_2 = self.datas_2[mask_save]
        self.datas_3 = self.datas_3[mask_save]
        self.datas_4 = self.datas_4[mask_save]
        self.labels = y[mask_save]

    def pre_data_process(self,scaler):
        self.datas_1 = scaler.transform(self.datas_1)
        self.datas_2 = scaler.transform(self.datas_2)
        self.datas_3 = scaler.transform(self.datas_3)
        self.datas_4 = scaler.transform(self.datas_4)
        self.datas_1 = np.nan_to_num(self.datas_1)
        self.datas_2 = np.nan_to_num(self.datas_2)
        self.datas_3 = np.nan_to_num(self.datas_3)
        self.datas_4 = np.nan_to_num(self.datas_4)
        self.labels = np.nan_to_num(self.labels)



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
                    if fname.endswith(f'.engagement.annotation.csv'):
                        anno_map[key] = full_path
        X_1, X_2, X_3, X_4, y = [], [], [], [], []
        if len(anno_map) != 0:
            is_test = False
            for key, anno_path in anno_map.items():
                group = key.split(';')[0]
                # 这里在做group的筛选
                if group not in self.group_name:
                    continue

                stream_path_1 = stream_map.get(key)

                if "subjectPos1" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos1", "subjectPos2"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos1", "subjectPos3"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos1", "subjectPos4"))

                if "subjectPos2" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos2", "subjectPos3"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos2", "subjectPos4"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos2", "subjectPos1"))

                if "subjectPos3" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos3", "subjectPos4"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos3", "subjectPos1"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos3", "subjectPos2"))

                if "subjectPos4" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos3", "subjectPos1"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos3", "subjectPos2"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos3", "subjectPos3"))


                if not stream_path_1:
                    continue

                a_1 = np.fromfile(stream_path_1, dtype=np.float32).reshape(-1, feat_dim)

                if not stream_path_2:
                    a_2 = np.zeros_like(a_1)
                else:
                    a_2 = np.fromfile(stream_path_2, dtype=np.float32).reshape(-1, feat_dim)

                if not stream_path_3:
                    a_3 = np.zeros_like(a_1)
                else:
                    a_3 = np.fromfile(stream_path_3, dtype=np.float32).reshape(-1, feat_dim)

                if not stream_path_4:
                    a_4 = np.zeros_like(a_1)
                else:
                    a_4 = np.fromfile(stream_path_4, dtype=np.float32).reshape(-1, feat_dim)


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
                        y_val = val if val == None else float(val)
                    except ValueError:
                        print('valueError')
                        continue

                    if i < len(a_1):
                        X_1.append(a_1[i])
                    else:
                        nan_array = np.full(a_1[0].shape, np.nan)
                        X_1.append(nan_array)

                    if i < len(a_2):
                        X_2.append(a_2[i])
                    else:
                        nan_array = np.full(a_2[0].shape, np.nan)
                        X_2.append(nan_array)

                    if i < len(a_3):
                        X_3.append(a_3[i])
                    else:
                        nan_array = np.full(a_3[0].shape, np.nan)
                        X_3.append(nan_array)

                    if i < len(a_4):
                        X_4.append(a_4[i])
                    else:
                        nan_array = np.full(a_4[0].shape, np.nan)
                        X_4.append(nan_array)


                    y.append(y_val)
            return np.array(X_1, dtype=np.float32), np.array(X_2, dtype=np.float32), np.array(X_3, dtype=np.float32), np.array(X_4, dtype=np.float32), np.array(y, dtype=np.float32),is_test




        else:
            is_test = True
            for key, anno_path in stream_map.items():
                key_origin = key
                group = key.split(';')[0]
                # 这里在做group的筛选
                if group not in self.group_name:
                    continue

                stream_path_1 = stream_map.get(key)

                if "subjectPos1" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos1", "subjectPos2"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos1", "subjectPos3"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos1", "subjectPos4"))

                if "subjectPos2" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos2", "subjectPos3"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos2", "subjectPos4"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos2", "subjectPos1"))

                if "subjectPos3" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos3", "subjectPos4"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos3", "subjectPos1"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos3", "subjectPos2"))

                if "subjectPos4" in key:
                    stream_path_2 = stream_map.get(key.replace("subjectPos3", "subjectPos1"))
                    stream_path_3 = stream_map.get(key.replace("subjectPos3", "subjectPos2"))
                    stream_path_4 = stream_map.get(key.replace("subjectPos3", "subjectPos3"))

                if not stream_path_1:
                    continue

                a_1 = np.fromfile(stream_path_1, dtype=np.float32).reshape(-1, feat_dim)

                if not stream_path_2:
                    a_2 = np.zeros_like(a_1)
                else:
                    a_2 = np.fromfile(stream_path_2, dtype=np.float32).reshape(-1, feat_dim)

                if not stream_path_3:
                    a_3 = np.zeros_like(a_1)
                else:
                    a_3 = np.fromfile(stream_path_3, dtype=np.float32).reshape(-1, feat_dim)

                if not stream_path_4:
                    a_4 = np.zeros_like(a_1)
                else:
                    a_4 = np.fromfile(stream_path_4, dtype=np.float32).reshape(-1, feat_dim)

                annos = []
                anno_path = anno_map.get(key_origin)
                if anno_path == None:
                    annos = [None for i in range(a_1.shape[0])]
                else:
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
                        y_val = val if val == None else float(val)
                    except ValueError:
                        print('valueError')
                        continue

                    if i < len(a_1):
                        X_1.append(a_1[i])
                    else:
                        nan_array = np.full(a_1[0].shape, np.nan)
                        X_1.append(nan_array)

                    if i < len(a_2):
                        X_2.append(a_2[i])
                    else:
                        nan_array = np.full(a_2[0].shape, np.nan)
                        X_2.append(nan_array)

                    if i < len(a_3):
                        X_3.append(a_3[i])
                    else:
                        nan_array = np.full(a_3[0].shape, np.nan)
                        X_3.append(nan_array)

                    if i < len(a_4):
                        X_4.append(a_4[i])
                    else:
                        nan_array = np.full(a_4[0].shape, np.nan)
                        X_4.append(nan_array)

                    y.append(y_val)

            y = [np.nan if v is None else v for v in y]
            return np.array(X_1, dtype=np.float32), np.array(X_2, dtype=np.float32), np.array(X_3,
                                                                                              dtype=np.float32), np.array(
                X_4, dtype=np.float32), np.array(y, dtype=np.float32),is_test



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
        X_1, X_2,X_3, X_4, y, is_test = self.walk_and_load(dir_path, modality, feat_dim)
        return X_1, X_2,X_3, X_4, y, is_test

    def __len__(self):
        return math.ceil((len(self.labels) -self.w) / self.s) + 1


    def __getitem__(self, index):
        indices = (np.arange(index * self.s - self.l, index * self.s + self.w + self.l) % len(self.labels))
        data_1 = self.datas_1[indices]
        data_2 = self.datas_2[indices]
        data_3 = self.datas_3[indices]
        data_4 = self.datas_4[indices]
        labels = self.labels[indices]

        return {
            'datas_1': torch.from_numpy(data_1).mT,
            'datas_2': torch.from_numpy(data_2).mT,
            'datas_3': torch.from_numpy(data_3).mT,
            'datas_4': torch.from_numpy(data_4).mT,
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
