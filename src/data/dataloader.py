import torch

from torch_geometric.data import Batch


def dict_collate(batch):
    data_pairs = {k: [] for k in batch[0]}

    for d in batch:
        for k in d:
            data_pairs[k].extend(d[k].to_data_list())

    return {
        k: Batch.from_data_list(data_pairs[k]) for k in data_pairs
    }


class DictDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DictDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=dict_collate, **kwargs
        )