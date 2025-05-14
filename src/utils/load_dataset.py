# ==== Standard Imports ====
from pathlib import Path
import numpy as np

# ==== Third Party Imports ====
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# ==== Local Project Imports ====
from src.data.dataset import MedicalDataset


def get_dataset(config):
    train_path = Path('../configs/train_indices.npy')
    val_path = Path('../configs/validation_indices.npy')
    test_path = Path('../configs/test_indices.npy')

    # load dataset
    dataset_path = Path(config.data.path)
    dataset = MedicalDataset(dataset_path)

    # extract labels
    labels = [dataset[i][0] for i in range(len(dataset))]

    if not train_path.exists() and not val_path.exists() and not test_path.exists():
        # 80-20 split (80 for training and 20 for validation and testing)
        splitter_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, temp_idx = next(splitter_1.split(X=np.zeros(len(labels)), y=labels))

        # 10-10 split for the 20% (10 for validation and 10 for testing)
        temp_labels = [labels[i] for i in temp_idx]
        splitter_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx_relative, test_idx_relative = next(splitter_2.split(X=np.zeros(len(temp_labels)), y=temp_labels))

        val_idx = [temp_idx[i] for i in val_idx_relative]
        test_idx = [temp_idx[i] for i in test_idx_relative]

        # save indices in file
        np.save(train_path, np.array(train_idx))
        np.save(val_path, np.array(val_idx))
        np.save(test_path, np.array(test_idx))

    return dataset


def get_train_val_dataset(dataset, config):
    train_path = Path('../configs/train_indices.npy')
    val_path = Path('../configs/validation_indices.npy')

    # indices
    train_idx = np.load(train_path, allow_pickle=True)
    val_idx = np.load(val_path, allow_pickle=True)

    # subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # dataloaders
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        pin_memory=config.training.pin_memory
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=config.validation.batch_size,
        shuffle=config.validation.shuffle,
        pin_memory=config.validation.pin_memory
    )

    return train_dl, val_dl


def get_test_dataset(dataset, config):
    test_path = Path('../configs/test_indices.npy')

    # indices
    test_idx = np.load(test_path, allow_pickle=True)

    # subset
    test_dataset = Subset(dataset, test_idx)

    # dataloader
    test_dl = DataLoader(
        test_dataset,
        batch_size=config.testing.batch_size,
        shuffle=config.testing.shuffle,
        pin_memory=config.testing.pin_memory
    )

    return test_dl