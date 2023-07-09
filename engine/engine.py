import numpy as np
import torch
import faiss
import pickle
import os, sys, json
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def evaluator(ground_truth, rank_index, top_k=10, metrics=['mrr', 'map', 'ndcg'], use_graded_scores=False):

    results = {}

    if 'mrr' in metrics:
        mrr = 0.
        for rank, item in enumerate(rank_index[:top_k]):
            if item in ground_truth:
                mrr = 1.0 / (rank + 1.0)
                break
        results['mrr'] = mrr

    if 'map' in metrics:
        if not ground_truth:
            return 0.
        map = 0.
        num_hits = 0.
        for rank, item in enumerate(rank_index[:top_k]):
            if item in ground_truth and item not in rank_index[:rank]:
                num_hits += 1.
                map += num_hits / (rank + 1.0)
        map = map / max(1.0, len(ground_truth))
        results['map'] = map
    
    if 'ndcg' in metrics:
        ndcg = 0.
        for rank, item in enumerate(rank_index[:top_k]):
            if item in ground_truth:
                if use_graded_scores:
                    grade = 1.0 / (ground_truth.index(item) + 1)
                else:
                    grade = 1.0
                ndcg += grade / np.log2(rank + 2)

        norm = 0.0
        for rank in range(len(ground_truth)):
            if use_graded_scores:
                grade = 1.0 / (rank + 1)
            else:
                grade = 1.0
            norm += grade / np.log2(rank + 2)
        results['ndcg'] = ndcg / max(0.3, norm)

    return results

def get_k_neighbors(query: np.ndarray, database: np.ndarray, k: int):

    ngpus = faiss.get_num_gpus()

    assert query.shape[-1] == database.shape[-1]

    dimention = query.shape[-1]
    # print(ngpus)
    if ngpus == 0:
        index = faiss.IndexFlatL2(dimention)
        index.add(database)
        D, I = index.search(query, k)     # actual search

    elif ngpus == 1:
        res = faiss.StandardGpuResources()  # use a single GPU
        ## Using a flat index
        index_flat = faiss.IndexFlatL2(dimention)  # build a flat (CPU) index
        # make it a flat GPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        gpu_index_flat.add(database)         # add vectors to the index

        D, I = gpu_index_flat.search(query, k)  # actual search

    else:
        cpu_index = faiss.IndexFlatL2(database)
        gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
        )
        gpu_index.add(database)              # add vectors to the index
        D, I = gpu_index.search(query, k) # actual search

    return I[:k]


def get_ground_truth(query, database, top_k=10): # num_class

    pass


def get_scores(query, database, top_k=10):

    rank_index = get_k_neighbors(top_k, query, database)
    ground_truth = get_ground_truth(query, top_k)
    # rank_index [q, k] k numbers of query
    results = evaluator(ground_truth, rank_index, top_k=top_k, metrics=['mrr', 'map', 'ndcg'])
    return results

def search(query, database, top_k=10):
    top_k_index =  get_k_neighbors(query, database, k=top_k)
    top_k_emb= database[top_k_index, :]
    return top_k_index, top_k_emb


if __name__ == '__main__':

    # torch.manual_seed(1234)

    query = 'resnet18'
    print(query)
    
    embeddings = torch.load('pretrained/embeddings.pt')
    embeddings = embeddings.data.detach().numpy().astype('float32')

    model2idx = {}
    idx2model = {}

    # Load model2idx and idx2model from JSON
    with open('maps/model2idx.json', 'r') as f:
        model2idx = json.load(f)
    with open('maps/idx2model.json', 'r') as f:
        idx2model = json.load(f)
    
    idx = model2idx[query]

    query_emb = embeddings[idx].reshape(1, -1)
    # print(query_emb.shape, embeddings.shape)
    top_k_index, top_k_emb = search(query_emb, embeddings)
    
    print(top_k_index)

    for id in top_k_index[0]:
        print(idx2model[str(id)])