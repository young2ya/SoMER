"""
Data loading augmentation utilities.

This module provides:
    - NPZ dataset loading helpers (expects each .npz to contain 'x' and 'y')
    - Dataset slicing into train/val/test/unlabeled splits
    - PyTorch Dataset wrapper (BearingDataset)
    - Method-specific transforms for SSL baselines
"""

import os
import random
import numpy as np

import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset


############################################################################
# 1. 데이터 관련 함수 (파일 불러오기, 병합, 슬라이싱 등)
############################################################################
### SLRA
def _npz_Loader(folder_path, each_num):
    """
    주어진 폴더 내 .npz 파일들 중 무작위로 each_num개를 골라 로드하고
    x, y를 병합하여 반환하는 함수수
    """
    # 폴더 안의 .npz 파일 리스트
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".npz")]
    # 무작위로 each_num개 샘플링링
    file_names = random.sample(file_names, each_num)

    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        X = npz['x']
        y = npz['y']

        npy_list.append(X)
        label_list.append(y)

    # 하나의 리스트에 쌓은 뒤, 최종적으로 vstack/hstack를 호출출
    npy_arr = np.vstack(npy_list)
    label_arr = np.hstack(label_list)

    return npy_arr, label_arr

### CWRU,hust,pu
def npz_Loader(folder_path):
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".npz")]

    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        x = npz['x']
        y = npz['y']

        npy_list.append(x)
        label_list.append(y)

    npy_arr = np.vstack(npy_list)
    label_arr = np.hstack(label_list)
    return npy_arr, label_arr


def slice_data(args, data_dict, label_dict):
    train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label = [{} for i in range(8)]

    for key in data_dict.keys():
        x = data_dict[key]
        y = label_dict[key]

        train_data[key] = x[:args.num_labeled]
        test_data[key] = x[args.num_labeled:args.num_labeled+args.val_num]
        val_data[key] = x[args.num_labeled+args.val_num:args.num_labeled+args.val_num*2]
        unlabel_data[key] = x[args.num_labeled+args.val_num*2:]

        train_label[key] = y[:args.num_labeled]
        test_label[key] = y[args.num_labeled:args.num_labeled+args.val_num]
        val_label[key] = y[args.num_labeled+args.val_num:args.num_labeled+args.val_num*2]
        unlabel_label[key] = y[args.num_labeled+args.val_num*2:]

    return (train_data, test_data, val_data, unlabel_data, train_label, test_label, val_label, unlabel_label)


def load_npz_files(folder_path_list, each_data_num):
    """
    여러 폴더(타입) 경로 목록을 받아,
    각 폴더마다 npz_loader로 데이터를 불러오고, 딕셔너리 형태로 저장하는 함수

    returns:
        arr_dict: dict[bearing_type] = X_array
        label_dict: dict[bearing_type] = y_array
    """
    arr_dict = {}
    label_dict = {}

    for folder_path in folder_path_list:
        bearing_type = os.path.basename(folder_path)
        print(f"Loading bearing type: {bearing_type}...")
        # X_arr, y_arr = npz_Loader(folder_path=folder_path, each_num=each_data_num)
        X_arr, y_arr = npz_Loader(folder_path=folder_path)

        arr_dict[bearing_type] = X_arr
        label_dict[bearing_type] = y_arr

    return arr_dict, label_dict

def concat_data(X_dict, y_dict):
    """
    여러 딕셔너리에 나눠진 데이터를 한 번에 합치는 함수
    X_dict.values(): 각 bearing_type별 X_array
    y_dict.values(): 각 bearing_type별 y_array

    결과 X, y를 반환하며,
    X.shape = (..., height, width)에 1채널 차원을 추가 -> 모델 입력을 위해서
    """
    X_concat = np.concatenate(list(X_dict.values()), axis=0)  # (N, H, W)
    y_concat = np.concatenate(list(y_dict.values()), axis=0)  # (N)

    # (N, H, W, 1) 형태로 만들어서 Pytorch의 (N, C, H, W)로 변환하기 쉽게 함
    X_concat = X_concat[..., np.newaxis]  # (N, H, W, 1)
    return X_concat, y_concat


def get_global_mean_std(X_train, X_unlab):
    """
    arr_dict: dict[bearing_type] = (N, H, W)

    (train + unlabeled)모든 bearing_type의 X_arr를 한 번에 합쳐서 전체 mean, std를 계산
    shape: (N, H, W)에 대해 np.mean / np.std를 구함
    returns:
        (mean, std)
    """
    # 모든 타입 데이터를 하나로 합치기
    X_all = np.concatenate([X_train, X_unlab], axis=0)

    X_2d = X_all[..., 0]

    # 전체 평균/표준편차
    mean = np.mean(X_2d)
    std = np.std(X_2d)
    return mean, std


############################################################################
# 2. Dataset 구현
############################################################################
class BearingDataset(Dataset):
    """
    기본 Dataset.
    X_data: (N, H, W, 1) -> permute -> (N, 1, H, W)
    y_data: (N, )
    """

    def __init__(self, x_data, y_data, transform=None):
        # x_data, y_data를 텐서로 변환
        # permute(0,3,1,2) -> (N, C=1, H, W)
        self.x_data = torch.FloatTensor(x_data).permute(0, 3, 2, 1)
        self.y_data = torch.LongTensor(y_data)
        self.transform = transform
        self.len = len(y_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


############################################################################
# 3. Custom Augmentations
############################################################################
class AddGaussianNoise(object):
    """
    간단한 가우시안 노이즈를 추가하는 증강.
    mean, std를 적당히 조절 가능
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        # x: torch.Tensor, shape=(C,H,W)
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class AddUniformNoise(object):
    """
    Uniform 분포 노이즈
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        noise = (self.high - self.low) * torch.rand_like(x) + self.low
        return x + noise


class SpecTimeMask(object):
    """
    스펙트로그램 시간축 일부 구간을 마스킹하는 증강 기법
    x shape : (C, H, W)
        C: 채널
        H: 주파수 축
        W: 시간 축
    """

    def __init__(self, time_mask_param, num_masks=1, fill_value=0.0):
        """
        time_mask_param: 마스킹 구간의 최대 폭(시간축에서)
        num_masks: 몇 개의 마스크를 적용할지
        fill_value: 마스킹 시 대체할 값
        """
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.fill_value = fill_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, width = x.shape

        for _ in range(self.num_masks):
            mask_width = random.randint(0, self.time_mask_param)

            start = random.randint(0, max(0, width - mask_width))
            x[:, :, start: start + mask_width] = self.fill_value
        return x


class SpecFreqMask(object):
    """
    스펙트로그램 주파수축 일부 구간을 마스킹하는 증강 기법
    x shape: (C, H, W)
        C: 채널
        H: 주파수 축
        W: 시간 축
    """

    def __init__(self, freq_mask_param, num_masks=1, fill_value=0.0):
        """
        freq_mask_param: 마스킹 구간의 최대 폭(주파수 축에서)
        num_masks:몇 개의 마스크를 적용할지
        fill_value: 마스킹 시 대체할 값
        """
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.fill_value = fill_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, height, _ = x.shape

        for _ in range(self.num_masks):
            mask_width = random.randint(0, self.freq_mask_param)
            start = random.randint(0, max(0, height - mask_width))
            x[:, start: start + mask_width, :] = self.fill_value
        return x


class SpecAugmentMask(object):
    """
    SpecAugment-style 마스킹 증강
    스펙트로그램의 시간축, 주파수축에 대해 각각 여러 개 구간을 마스킹
    """

    def __init__(self, freq_mask_param,
                 time_mask_param,
                 num_freq_masks=1,
                 num_time_masks=1,
                 fill_value=0.0):

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.fill_value = fill_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape

        for _ in range(self.num_freq_masks):
            mask_width = random.randint(0, self.freq_mask_param)
            start = random.randint(0, max(0, h - mask_width))
            x[:, start: start + mask_width, :] = self.fill_value

        for _ in range(self.num_time_masks):
            mask_width = random.randint(0, self.time_mask_param)
            start = random.randint(0, max(0, w - mask_width))
            x[:, :, start: start + mask_width] = self.fill_value

        return x


############################################################################
# 3. Augmentation
############################################################################
class Transform_MixMatch(object):

    def __init__(self, mean, std):
        self.transform = tr.Compose([
            AddGaussianNoise(mean=0, std=0.01),
            SpecTimeMask(time_mask_param=10)
        ])

        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        x_1 = self.transform(x)
        x_2 = self.transform(x)
        return self.normalize(x_1), self.normalize(x_2)


class _Transform_FixMatch(object):
    """
    FixMatch transform
    weak transform / strong transform이 다름
    """

    def __init__(self, mean, std):
        # weak transform
        self.transform_w = tr.Compose([
            AddGaussianNoise(mean=0, std=0.01),
            SpecTimeMask(time_mask_param=10)
        ])

        # strong transform
        self.transform_s = tr.Compose([
            AddGaussianNoise(mean=0, std=0.05),
            SpecAugmentMask(time_mask_param=20, freq_mask_param=20)
        ])

        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        weak_aug = self.transform_w(x)
        strong_aug = self.transform_s(x)

        w = self.normalize(weak_aug)
        s = self.normalize(strong_aug)

        return w, s


class Transform_FixMatch(object):
    """
    FixMatch transform
    weak transform / strong transform이 다름
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

        # weak transform
        self.noise_w = AddGaussianNoise(mean=0, std=0.01)
        self.mask_w = SpecTimeMask(time_mask_param=10)

        # strong transform
        self.noise_s = AddGaussianNoise(mean=0, std=0.05)
        self.mask_s = SpecAugmentMask(time_mask_param=10, freq_mask_param=10)

        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        w = self.noise_w(x)
        w = self.normalize(w)
        w = self.mask_w(w)

        s = self.noise_s(x)
        s = self.normalize(s)
        s = self.mask_s(s)

        return w, s


class Transform_Proposed(object):
    def __init__(self, mean, std):
        self.transform = tr.Compose([
            AddGaussianNoise(mean=0, std=0.01),
            SpecTimeMask(time_mask_param=20),
            # SpecAugmentMask(time_mask_param=10,freq_mask_param=10)
        ])

        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        aug1 = self.transform(x)
        aug2 = self.transform(x)
        return self.normalize(aug1), self.normalize(aug2)

class Transform_Proposed_Multi(object):
    def __init__(self, mean, std, mask, n_aug=2):
        self.n_aug = n_aug
        self.transform = tr.Compose([
            AddGaussianNoise(mean=0, std=0.01),
            SpecTimeMask(time_mask_param=mask)
        ])
        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        return [self.normalize(self.transform(x)) for _ in range(self.n_aug)]

class Transform_Proposed_ulb(object):
    def __init__(self, mean, std, mask, n_aug=2):
        self.n_aug = n_aug
        self.transform = tr.Compose([
            AddGaussianNoise(mean=0, std=0.01),
            SpecAugmentMask(time_mask_param=mask, freq_mask_param=mask)
        ])

        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        return [self.normalize(self.transform(x)) for _ in range(self.n_aug)]

def get_transform(args, mean, std):
    """
    return train, unlab, val/test transform
    """
    val_transform = tr.Compose([
        tr.Normalize(mean, std)
    ])

    basic_transform = tr.Compose([
        AddGaussianNoise(0, 0.01),
        tr.Normalize(mean, std)
    ])

    method = args.method
    lab_mask = args.lab_mask
    unlab_mask = args.unlab_mask

    if method == 'supervised':
        return Transform_Proposed_Multi(mean, std, lab_mask), None, val_transform
    elif method == 'pseudo':
        return val_transform, val_transform, val_transform
    elif method == 'hcae':
        return val_transform, val_transform, val_transform
    elif method == 'mixmatch':
        return basic_transform, Transform_MixMatch(mean, std), val_transform
    elif method == 'fixmatch':
        return basic_transform, Transform_FixMatch(mean, std), val_transform
    elif method == 'simmatch':
        return basic_transform, Transform_FixMatch(mean, std), val_transform
    elif method == 'proposed':
        # return Transform_Proposed(mean, std), Transform_Proposed_ulb(mean, std), val_transform
        return Transform_Proposed_Multi(mean, std, lab_mask), Transform_Proposed_ulb(mean, std, unlab_mask), val_transform
    else:
        return val_transform, val_transform, val_transform


############################################################################
# 4. Data Loader
############################################################################
def get_data(args, path_list):
    # 1) 각 폴더에서 npz 파일 로드
    data_dict, label_dict = load_npz_files(path_list, args.each_data_num)

    # 2) 각 type별로 셔플
    shuffle_data_dict, shuffle_label_dict = {}, {}
    for key in data_dict.keys():
        X = data_dict[key]
        y = label_dict[key]
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        shuffle_data_dict[key] = X[idx]
        shuffle_label_dict[key] = y[idx]

    # 3) Slicing
    (train_data, test_data, val_data, unlabel_data,
     train_label, test_label, val_label, unlabel_label) = slice_data(args, shuffle_data_dict, shuffle_label_dict)

    # 4) Concat
    X_train, y_train = concat_data(train_data, train_label)
    X_test, y_test = concat_data(test_data, test_label)
    X_val, y_val = concat_data(val_data, val_label)
    X_unlabel, y_unlabel = concat_data(unlabel_data, unlabel_label)

    # 5) transform 구성
    mean, std = get_global_mean_std(X_train, X_unlabel)

    transform_labeled, transform_unlabeled, transform_val = get_transform(args, mean, std)

    # 6) Dataset 생성
    if args.method == 'supervised':
        trainset = BearingDataset(X_train, y_train, transform=transform_labeled)
        valset = BearingDataset(X_val, y_val, transform=transform_val)
        testset = BearingDataset(X_test, y_test, transform=transform_val)
        unlabelset = None
        return trainset, valset, testset

    else:
        trainset = BearingDataset(X_train, y_train, transform=transform_labeled)
        unlabelset = BearingDataset(X_unlabel, y_unlabel, transform=transform_unlabeled)
        valset = BearingDataset(X_val, y_val, transform=transform_val)
        testset = BearingDataset(X_test, y_test, transform=transform_val)
        return trainset, unlabelset, valset, testset
