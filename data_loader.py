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
# 1. Data functions (file loading, merging, slicing, etc.)
############################################################################
### SLRA (the normal (N) class has far more samples than the fault classes,
### so this loader randomly samples only each_num files per folder to balance class counts)
def npz_Loader_balanced(folder_path, each_num):
    """
    Randomly picks each_num .npz files from the given folder, loads them,
    and returns the concatenated x, y (SLRA-only class balancing).
    """
    # List .npz files in the folder
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".npz")]
    # Randomly sample each_num files
    file_names = random.sample(file_names, each_num)

    npy_list = []
    label_list = []
    for file_name in file_names:
        npz = np.load(file_name)
        X = npz['x']
        y = npz['y']

        npy_list.append(X)
        label_list.append(y)

    # Collect into lists, then vstack/hstack at the end
    npy_arr = np.vstack(npy_list)
    label_arr = np.hstack(label_list)

    return npy_arr, label_arr

### CWRU, HUST, PU (loads every .npz file in the folder as-is — no class balancing)
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
    """
    Splits each bearing_type's (key's) array into train/test/val/unlabeled
    slices using fixed-size windows: [0:num_labeled) for train,
    [num_labeled:num_labeled+val_num) for test,
    [num_labeled+val_num:num_labeled+2*val_num) for val,
    and everything after that for unlabeled.
    """
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


def load_npz_files(args, folder_path_list):
    """
    Loads npz data from each folder (bearing type) in folder_path_list and
    stores it in dictionaries.

    The loader is chosen automatically based on args.dataset:
        - 'slra': npz_Loader_balanced (samples only args.each_data_num files
          per folder to offset the normal (N) class imbalance)
        - otherwise: npz_Loader (loads every file in the folder, no balancing)

    returns:
        arr_dict: dict[bearing_type] = X_array
        label_dict: dict[bearing_type] = y_array
    """
    arr_dict = {}
    label_dict = {}

    for folder_path in folder_path_list:
        bearing_type = os.path.basename(folder_path)
        print(f"Loading bearing type: {bearing_type}...")

        if args.dataset == 'slra':
            X_arr, y_arr = npz_Loader_balanced(folder_path, args.each_data_num)
        else:
            X_arr, y_arr = npz_Loader(folder_path)

        arr_dict[bearing_type] = X_arr
        label_dict[bearing_type] = y_arr

    return arr_dict, label_dict

def concat_data(X_dict, y_dict):
    """
    Merges data spread across multiple per-bearing-type dictionaries into one.
    X_dict.values(): X_array per bearing_type
    y_dict.values(): y_array per bearing_type

    Returns the concatenated X, y. A channel dimension is appended so
    X.shape = (..., height, width, 1), ready for the model input format.
    """
    X_concat = np.concatenate(list(X_dict.values()), axis=0)  # (N, H, W)
    y_concat = np.concatenate(list(y_dict.values()), axis=0)  # (N)

    # Add a channel dim -> (N, H, W, 1), so it converts cleanly to PyTorch's (N, C, H, W)
    X_concat = X_concat[..., np.newaxis]  # (N, H, W, 1)
    return X_concat, y_concat


def get_global_mean_std(X_train, X_unlab):
    """
    arr_dict: dict[bearing_type] = (N, H, W)

    Concatenates train + unlabeled data across all bearing types and computes
    a single global mean/std over the (N, H, W) array.
    returns:
        (mean, std)
    """
    # Merge all bearing types into one array
    X_all = np.concatenate([X_train, X_unlab], axis=0)

    X_2d = X_all[..., 0]

    # Global mean/std
    mean = np.mean(X_2d)
    std = np.std(X_2d)
    return mean, std


############################################################################
# 2. Dataset implementation
############################################################################
class BearingDataset(Dataset):
    """
    Base Dataset.
    X_data: (N, H, W, 1) -> permute -> (N, 1, H, W)
    y_data: (N, )
    """

    def __init__(self, x_data, y_data, transform=None):
        # Convert x_data, y_data to tensors
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
    Simple additive Gaussian noise augmentation.
    mean, std can be tuned as needed.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        # x: torch.Tensor, shape=(C,H,W)
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class SpecTimeMask(object):
    """
    Augmentation that masks a segment along the spectrogram's time axis.
    x shape : (C, H, W)
        C: channel
        H: frequency axis
        W: time axis
    """

    def __init__(self, time_mask_param, num_masks=1, fill_value=0.0):
        """
        time_mask_param: maximum width of the masked segment (time axis)
        num_masks: number of masks to apply
        fill_value: value used to fill the masked region
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
    Augmentation that masks a segment along the spectrogram's frequency axis.
    Currently it's only ever used together with time masking (via
    SpecAugmentMask) as the strong augmentation, so it isn't called on its
    own — kept here for future standalone/combination experiments.
    x shape: (C, H, W)
        C: channel
        H: frequency axis
        W: time axis
    """

    def __init__(self, freq_mask_param, num_masks=1, fill_value=0.0):
        """
        freq_mask_param: maximum width of the masked segment (frequency axis)
        num_masks: number of masks to apply
        fill_value: value used to fill the masked region
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
    SpecAugment-style masking augmentation.
    Masks several segments along both the time and frequency axes of the
    spectrogram.
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


class Transform_FixMatch(object):
    """
    FixMatch transform: builds and returns a weak and a strong augmentation.
    Order of operations is noise -> normalize -> mask, so that the 0.0 fill
    value used by masking corresponds to the true mean in normalized space
    (the standard SpecAugment convention).
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


class Transform_Proposed_Multi(object):
    """
    Augmentation for the labeled data in the 'proposed' method.
    Applies only time-axis masking (SpecTimeMask) to produce n_aug views.
    Mask strength is controlled by args.lab_mask.
    """

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
    """
    Augmentation for the unlabeled data in the 'proposed' method.
    Applies both time- and frequency-axis masking (SpecAugmentMask) together,
    a stronger augmentation than Transform_Proposed_Multi, to produce n_aug
    views. Mask strength is controlled by args.unlab_mask.
    """

    def __init__(self, mean, std, mask, n_aug=2):
        self.n_aug = n_aug
        self.transform = tr.Compose([
            AddGaussianNoise(mean=0, std=0.01),
            SpecAugmentMask(time_mask_param=mask, freq_mask_param=mask)
        ])

        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        return [self.normalize(self.transform(x)) for _ in range(self.n_aug)]

class Transform_Proposed_NoAug(object):
    """
    Ablation variant for 'proposed' (--no-mask-aug): no noise, no masking —
    just normalizes the raw spectrogram. Used for both labeled and unlabeled
    data so Proposed_train can be compared against the same method with
    augmentation disabled. n_aug defaults to 1 (not 2, like the augmented
    transforms) since duplicating an identical unaugmented view would only
    inflate the effective batch size without adding any signal.
    """

    def __init__(self, mean, std, n_aug=1):
        self.n_aug = n_aug
        self.normalize = lambda x: (x - mean) / (std + 1e-8)

    def __call__(self, x):
        return [self.normalize(x) for _ in range(self.n_aug)]

def get_transform(args, mean, std):
    """
    Picks the (labeled transform, unlabeled transform, val/test transform)
    combination that matches args.method. The unlabeled transform is unused
    for pseudo/hcae/supervised, so it's filled with None or val_transform.
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
        if getattr(args, 'no_mask_aug', False):
            # Ablation: no masking augmentation (and no noise, bundled with it)
            return Transform_Proposed_NoAug(mean, std), Transform_Proposed_NoAug(mean, std), val_transform
        n_aug = getattr(args, 'n_aug', 2)
        return (Transform_Proposed_Multi(mean, std, lab_mask, n_aug=n_aug),
                Transform_Proposed_ulb(mean, std, unlab_mask, n_aug=n_aug),
                val_transform)
    else:
        return val_transform, val_transform, val_transform


############################################################################
# 4. Data Loader
############################################################################
def get_data(args, path_list):
    # 1) Load npz files from each folder (loader is chosen automatically per dataset, see load_npz_files)
    data_dict, label_dict = load_npz_files(args, path_list)

    # 2) Shuffle within each bearing type
    shuffle_data_dict, shuffle_label_dict = {}, {}
    for key in data_dict.keys():
        X = data_dict[key]
        y = label_dict[key]
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        shuffle_data_dict[key] = X[idx]
        shuffle_label_dict[key] = y[idx]

    # 3) Slice into train/test/val/unlabeled
    (train_data, test_data, val_data, unlabel_data,
     train_label, test_label, val_label, unlabel_label) = slice_data(args, shuffle_data_dict, shuffle_label_dict)

    # 4) Concatenate bearing types back together
    X_train, y_train = concat_data(train_data, train_label)
    X_test, y_test = concat_data(test_data, test_label)
    X_val, y_val = concat_data(val_data, val_label)
    X_unlabel, y_unlabel = concat_data(unlabel_data, unlabel_label)

    # 5) Build transforms
    mean, std = get_global_mean_std(X_train, X_unlabel)

    transform_labeled, transform_unlabeled, transform_val = get_transform(args, mean, std)

    # 6) Build datasets
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
