import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import faiss
import random
import string
import time
import pickle
import gc
import argparse



def get_KNN(embedding, k, device_idx=0, use_multi_gpu=False, exact=False):
    if not exact:
        d = embedding.shape[1]
        quantizer = faiss.IndexFlatIP(d)
        res = faiss.StandardGpuResources()
        index = faiss.IndexIVFPQ(quantizer, d, 50000, 8, 8, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexIVFFlat(quantizer, d, 50000, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.index_factory(d, "PCA64,Flat", faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.train(embedding)
        gpu_index.add(embedding)
    elif use_multi_gpu:
        d = embedding.shape[1]
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(embedding)
    else:
        d = embedding.shape[1]
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embedding)
    print(gpu_index.ntotal)
    similarity, indices = gpu_index.search(embedding, k)
    del gpu_index
    gc.collect()
    return similarity, indices

def find_new_index(embedding, indices_path, similarity_path, device_idx=0, use_multi_gpu=False, exact=False):
    print('start knn')
    similarity, indices = get_KNN(embedding, 30, device_idx=device_idx, use_multi_gpu=use_multi_gpu, exact=exact)
    if not exact:
        similarity = get_true_cosine_similarity(embedding, indices)
    with open(indices_path, 'wb') as f:
        np.save(f, indices)
    with open(similarity_path, 'wb') as f:
        np.save(f, similarity)
    print('done knn')
    return None

def get_true_cosine_similarity(embedding, indices):
    similarity = []
    for i in range(indices.shape[0]):
        similarity.append(np.dot(embedding[indices[i]], embedding[i].reshape(-1, 1)).reshape(-1))
    return np.array(similarity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_data_dir', type=str, default='../use_data')
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--exact', type=bool, default=False)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_idx)
    embedding = np.load(os.path.join(args.use_data_dir,\
        'embedding.npy')).astype(np.float32)
    indices_path = os.path.join(args.use_data_dir, 'indices.npy')
    similarity_path = os.path.join(args.use_data_dir, 'similarity.npy')
    find_new_index(embedding, indices_path, similarity_path, device_idx=args.device_idx, exact=args.exact, use_multi_gpu=args.use_multi_gpu)
