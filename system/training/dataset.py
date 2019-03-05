# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2018-01-18 10:28:31
# @Last Modified by:   richman
# @Last Modified time: 2018-04-11
import kaldi_io
import numpy as np
import torch
from torch.utils import data
from imblearn.over_sampling import RandomOverSampler


class ListDataset(torch.utils.data.Dataset):
    """Dataset wrapping List.

    Each sample will be retrieved by indexing List along the first dimension.

    Arguments:
        *lists (List): List that have the same size of the first dimension.
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(a_list) for a_list in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(a_list[index] for a_list in self.lists)

    def __len__(self):
        return len(self.lists[0])


def seq_collate_fn(data_batches):
    """seq_collate_fn

    Helper function for torch.utils.data.Dataloader

    :param data_batches: iterateable
    """
    data_batches.sort(key=lambda x: len(x[0]), reverse=True)

    def merge_seq(dataseq, dim=0):
        lengths = [seq.shape for seq in dataseq]
        # Assuming duration is given in the first dimension of each sequence
        maxlengths = tuple(np.max(lengths, axis=dim))

        # For the case that the lenthts are 2dimensional
        lengths = np.array(lengths)[:, dim]
        # batch_mean = np.mean(np.concatenate(dataseq),axis=0, keepdims=True)
        # padded = np.tile(batch_mean, (len(dataseq), maxlengths[0], 1))
        padded = np.zeros((len(dataseq),) + maxlengths)
        for i, seq in enumerate(dataseq):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, lengths
    features, targets = zip(*data_batches)
    features_seq, feature_lengths = merge_seq(features)
    return torch.from_numpy(features_seq), torch.tensor(targets)


def create_dataloader(
        kaldi_string, label_dict, transform=None,
        batch_size: int = 16, num_workers: int = 1, shuffle: bool = True
):
    """create_dataloader

    :param kaldi_string: copy-feats input
    :param label_dict: Labels from kaldi_data to manual value
    :param transform: Transformation on the data (scaling)
    :param batch_size: Batchsize
    :type batch_size: int
    :param num_workers: Number of workers in parallel
    :type num_workers: int
    """
    def valid_feat(item):
        """valid_feat
        Checks if feature is in labels

        :param item: key value pair from read_mat_ark
        """
        return item[0] in label_dict

    features = []
    labels = []
    # Directly filter out all utterances without labels
    for idx, (k, feat) in enumerate(filter(valid_feat, kaldi_io.read_mat_ark(kaldi_string))):
        if transform:
            feat = transform(feat)
        features.append(feat)
        labels.append(label_dict[k])
    assert len(features) > 0, "No features were found, are the labels correct?"
    # Shuffling means that this is training dataset, so oversample
    if shuffle:
        random_oversampler = RandomOverSampler(random_state=0)
        # Dummy X data, y is the binary label
        _, _ = random_oversampler.fit_resample(
            torch.ones(len(features), 1), [l[1] for l in labels])
        # Get the indices for the sampled data
        indicies = random_oversampler.sample_indices_
        # reindex, new data is oversampled in the minority class
        features, labels = [features[id]
                            for id in indicies], [labels[id] for id in indicies]
    dataset = ListDataset(features, labels)
    # Configure weights to reduce number of unseen utterances
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=seq_collate_fn, shuffle=shuffle)
