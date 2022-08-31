import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import faiss
import random
import string
import time
import pickle
import gc
import argparse

batch_size = 256
device_idx = 0
device = torch.device("cuda:"+str(device_idx))

def get_bert_embed(phrase_list, m, tok, normalize=True, summary_method="CLS", tqdm_bar=False):
    m = m.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
#                     with torch.cuda.device('cuda:6'):
                    torch.cuda.empty_cache()
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
            # with torch.cuda.device('cuda:6'):
            torch.cuda.empty_cache()
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    torch.cuda.empty_cache()
    return np.concatenate(output_list, axis=0)

def get_KNN(embeddings, k, use_multi_gpu=True, exact=True):
    if not exact:
        d = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(d)
        res = faiss.StandardGpuResources()
        index = faiss.IndexIVFPQ(quantizer, d, 50000, 8, 8, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexIVFFlat(quantizer, d, 50000, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.index_factory(d, "PCA64,Flat", faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_gpu(res, device_idx, index)
        gpu_index.train(embeddings)
        gpu_index.add(embeddings)
    elif use_multi_gpu:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(embeddings)
    else:
        d = embeddings.shape[1]
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embeddings)
    print(gpu_index.ntotal)
    similarity, indices = gpu_index.search(embeddings, k)
    del gpu_index
    gc.collect()
    return similarity, indices

def find_new_index(model_name_or_path, indices_path, similarity_path, embedding_path, ner_phrase2idx_path, exact):
    print('start finding new index...')
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, config=config)
    print('start loading phrases...')
    with open(ner_phrase2idx_path, 'rb') as f:
        phrase2idx = pickle.load(f)
    phrase_list = list(phrase2idx.keys())
    embeddings = get_bert_embed(phrase_list, model, tokenizer, summary_method="MEAN", tqdm_bar=True)
    del model
    torch.cuda.empty_cache()
    with open(embedding_path, 'wb') as f:
        np.save(f, embeddings)
    print('start knn')
    embeddings = np.load(embedding_path)
    similarity, indices = get_KNN(embeddings, 30, use_multi_gpu=False, exact=exact)
    if not exact:
        similarity = get_true_cosine_similarity(embeddings, indices)
    with open(indices_path, 'wb') as f:
        np.save(f, indices)
    with open(similarity_path, 'wb') as f:
        np.save(f, similarity)
    print('done knn')
    return None

def get_true_cosine_similarity(embeddings, indices):
    similarity = []
    for i in range(indices.shape[0]):
        similarity.append(np.dot(embeddings[indices[i]], embeddings[i].reshape(-1, 1)).reshape(-1))
    return np.array(similarity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="GanjinZero/coder_eng_pp",
        type=str,
        help="model"
    )
    parser.add_argument(
        "--save_dir",
        default="../use_data/",
        type=str,
        help="output dir"
    )
    parser.add_argument(
        "--phrase2idx_path",
        default="../use_data/phrase2idx.pkl",
        type=str,
        help="Path to ner phrase2idx file"
    )
    args = parser.parse_args()
    args.indices_path = args.save_dir + 'indices.npy'
    args.similarity_path = args.save_dir + 'similarity.npy'
    args.embedding_path = args.save_dir + 'embedding.npy'
    
    find_new_index(
        model_name_or_path=args.model_name_or_path,
        indices_path=args.indices_path,
        similarity_path=args.similarity_path,
        embedding_path=args.embedding_path,
        ner_phrase2idx_path=args.phrase2idx_path
    )
