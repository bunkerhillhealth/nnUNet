import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json
from sklearn.model_selection import KFold

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


def create_split_file(splits_file, dataset: nnUNetDataset):
    splits = []
    all_keys_sorted = np.sort(list(dataset.keys()))
    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        splits.append({})
        splits[-1]["train"] = list(train_keys)
        splits[-1]["val"] = list(test_keys)
    save_json(splits, splits_file)


def create_random_split(fold, dataset: nnUNetDataset):
    rnd = np.random.RandomState(seed=12345 + fold)
    keys = np.sort(list(dataset.keys()))
    idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
    idx_val = [i for i in range(len(keys)) if i not in idx_tr]
    tr_keys = [keys[i] for i in idx_tr]
    val_keys = [keys[i] for i in idx_val]

    return tr_keys, val_keys
