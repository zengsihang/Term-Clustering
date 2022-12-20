import itertools
import pickle
import os
import numpy as np
from tqdm import tqdm, trange
from sklearn.cluster import spectral_clustering, KMeans
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import argparse
import ahocorasick
from random import sample
from multiprocessing import Pool



device = torch.device("cuda:0")
batch_size = 64
MAX_CLUSTER_COUNT = 10
mode = 'ratio'
# model = AutoModel.from_pretrained('GanjinZero/coder_eng_pp')
# tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp')

def read_en_ch_sim(path, sim_automaton):
    print('read_en_ch_sim')
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            sim_automaton.add_word('|'.join([line[0].lower(), line[1]]), float(line[3]))
            sim_automaton.add_word('|'.join([line[1], line[0].lower()]), float(line[3]))
    return sim_automaton

def read_en_en_sim(similarity, indices, idx2phrase, sim_automaton):
    print('read_en_en_sim')
    for idx in trange(indices.shape[0]):
        phrase1 = idx2phrase[idx]
        for pos, idx2 in enumerate(indices[idx]):
            phrase2 = idx2phrase[idx2]
            sim = similarity[idx][pos]
            sim_automaton.add_word('|'.join([phrase1, phrase2]), sim)
            sim_automaton.add_word('|'.join([phrase2, phrase1]), sim)
    return sim_automaton


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        df = pickle.load(f)
    return df

def get_bert_embed(phrase_list, normalize=True, summary_method="MEAN", tqdm_bar=False):
    global model, tokenizer
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, padding='max_length')['input_ids'])
        # print(len(input_ids))
    model.eval()
    model = model.to(device)
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
                embed = model(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
                    torch.cuda.empty_cache()
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
            torch.cuda.empty_cache()
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    torch.cuda.empty_cache()
    return np.mean(np.concatenate(output_list, axis=0), axis=0)

def get_mean_embed(terms_list, phrase2idx, embedding):
    idx_list = [phrase2idx[term] for term in terms_list]
    embedding_list = [embedding[idx].reshape(1, -1) for idx in idx_list]
    return np.mean(np.concatenate(embedding_list, axis=0), axis=0)


def re_cluster(terms_list, mode='ratio', ratio=1.5):
    terms_list = list(terms_list)
    ready = [terms_list]
    res = []
    while ready:
        now = ready.pop()
        clu0, clu1 = cut(now, mode, sim_automaton, ratio)
        # membed_0 = get_bert_embed(clu0)
        # membed_1 = get_bert_embed(clu1)
        # membed_0 = get_mean_embed(clu0, phrase2idx, embedding)
        # membed_1 = get_mean_embed(clu1, phrase2idx, embedding)
        # if np.dot(membed_0, membed_1) > threshold or len(clu0) <= 1 or len(clu1) <= 1:
        # res.append(clu0)
        # res.append(clu1)
        # return res
        if len(clu1)==0:
            res.append(clu0)
            # res.append(clu1)
        else:
            # ready.append(clu0)
            # ready.append(clu1)
            if len(clu0) <= MAX_CLUSTER_COUNT:
                res.append(clu0)
            else:
                ready.append(clu0)
            if len(clu1) <= MAX_CLUSTER_COUNT:
                res.append(clu1)
            else:
                ready.append(clu1)
        # for clu in [clu0, clu1]:
        #     if len(clu) <= MAX_CLUSTER_COUNT:
        #         res.append(clu)
        #     else:
        #         ready.append(clu)
    return res

def cut(terms_list, mode, sim_automaton, ratio):
    if mode == 'ratio':
        clu0, clu1 = ratio_cut(terms_list, sim_automaton, ratio)
    else:
        clu0, clu1 = normalize_cut(terms_list, similarity)
    return clu0, clu1

def get_sim(terms_list, sim_automaton):
    sim = np.zeros(shape=(len(terms_list), len(terms_list)))
    term2idx = {term: idx for idx, term in enumerate(terms_list)}
    for term1, term2 in itertools.combinations(terms_list, 2):
        if '|'.join([term1, term2]) in sim_automaton:
            sim[term2idx[term1]][term2idx[term2]] = sim_automaton.get('|'.join([term1, term2]))
            sim[term2idx[term2]][term2idx[term1]] = sim[term2idx[term1]][term2idx[term2]]
        else:
            if term1 in en_automaton and term2 in en_automaton:
                idx1 = phrase2id[term1]
                idx2 = phrase2id[term2]
                if idx1 in indices[idx2]:
                    sim[term2idx[term2]][term2idx[term1]] = similarity[idx2][np.argwhere(indices[idx2]==idx1)]
                elif idx2 in indices[idx1]:
                    sim[term2idx[term1]][term2idx[term2]] = similarity[idx1][np.argwhere(indices[idx1]==idx2)]
    # diagnal = 1.0
    for idx in range(len(terms_list)):
        sim[idx][idx] = 0.0
            
    return sim, np.sum(sim)/(len(terms_list)*(len(terms_list)-1)+1e-8)

def laplacian(matrix, normalize=False):
    d_val = matrix.sum(axis=0)
    d = np.diag(d_val)
    l = d - matrix
    if normalize:
        d_inverse_root_val = d_val ** (-1/2)
        d_inverse_root = np.diag(d_inverse_root_val)
        l = np.matmul(np.matmul(d_inverse_root, l), d_inverse_root)
    return l

def transform(sim):
    return np.exp(-(np.max(0.95-sim, 0))**2/0.15**2)

def ratio_cut(terms_list, sim_automaton, ratio):
    sim, _ = get_sim(terms_list, sim_automaton)
    # sim = transform(sim)
    l = laplacian(sim)
    u, v = np.linalg.eig(l)
    index = np.argsort(u.real)
    # print(u.real[index][:5])
    feat = v[:,index[0:2]].real
    feat_norm = np.linalg.norm(feat, ord=2, axis=1, keepdims=True)
    feat = feat / (feat_norm+1e-8)
    cluster = KMeans(n_clusters=2).fit_predict(feat)
    clu0 = np.array(terms_list)[cluster==0].tolist()
    clu1 = np.array(terms_list)[cluster==1].tolist()
    sim0, mean_sim0 = get_sim(clu0, sim_automaton)
    sim1, mean_sim1 = get_sim(clu1, sim_automaton)
    cut_mean_sim = \
    (np.sum(sim)-np.sum(sim0)-np.sum(sim1))/(2*len(clu0)*(len(clu1))+1e-8)
    # print(mean_sim0, mean_sim1, cut_mean_sim, min(mean_sim0, \
        # mean_sim1)/(cut_mean_sim+1e-8))
    if min(mean_sim0, mean_sim1)/(cut_mean_sim+1e-8)<ratio:
        clu0 = terms_list
        clu1 = []
        # print(min(sim0, sim1)/(cut_mean_sim+1e-8))
    return clu0, clu1


def normalize_cut(terms_list, similarity):
    sim = get_sim(terms_list, similarity)
    l = laplacian(sim, True)
    u, v = np.linalg.eig(l)
    index = np.argsort(u.real)
    feat = v[:,index[0:2]].real
    feat_norm = np.linalg.norm(feat, ord=2, axis=1, keepdims=True)
    feat = feat / feat_norm
    cluster = KMeans(n_clusters=2).fit_predict(feat)
    clu0 = np.array(terms_list)[cluster==0].tolist()
    clu1 = np.array(terms_list)[cluster==1].tolist()
    return clu0, clu1

def print_cluster_to_file(f, one_cluster_result):
    for idx, term in enumerate(one_cluster_result):
        f.write(term)
        if idx != len(one_cluster_result) - 1:
            f.write('|')
        else:
            f.write('\n')

def load_cluster(path):
    cluster = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cluster.append(line.strip().split('|'))
    return cluster

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_data_dir",
        default="../clustering_220613/use_data/",
        type=str,
        help="Directory to indices and similarity and idx2phrase"
    )
    parser.add_argument(
        "--result_dir",
        default="../result/",
        type=str,
        help="Directory to save clustering result"
    )
    parser.add_argument(
            "--ratio",
            type=float,
            default=5)
    args = parser.parse_args()
    args.indices_path = os.path.join(args.use_data_dir, 'indices.npy')
    args.similarity_path = os.path.join(args.use_data_dir, 'similarity.npy')
    args.idx2phrase_path = os.path.join(args.use_data_dir, 'idx2phrase.pkl')
    args.result_path = os.path.join(args.result_dir, 'en_ch_precut.txt')
    # args.result_path = './en_ch_precut.txt'
    args.phrase2idx_path = os.path.join(args.use_data_dir, 'phrase2idx.pkl')
    args.embedding_path = os.path.join(args.use_data_dir, 'embedding.npy')
    args.en_ch_sim_path = os.path.join(args.use_data_dir, \
            'bt_high_sim.txt')


    # cluster_res = load_pickle(args.result_path)
    cluster_res = load_cluster(args.result_path)
    id2phrase = load_pickle(args.idx2phrase_path)
    phrase2id = load_pickle(args.phrase2idx_path)
    similarity = np.load(args.similarity_path)
    indices = np.load(args.indices_path)
    sim_automaton = ahocorasick.Automaton()
    sim_automaton = read_en_ch_sim(args.en_ch_sim_path, sim_automaton)
    en_automaton = ahocorasick.Automaton()
    for phrase in tqdm(phrase2id.keys()):
        en_automaton.add_word(phrase, '')
    # sim_automaton = read_en_en_sim(similarity, indices, id2phrase, sim_automaton)
    # embedding = np.load(args.embedding_path)

    final_res = []
    ori = []
    ratio = args.ratio
    for cluster in tqdm(cluster_res):
        # if 'porous carbon matrix' not in cluster and 'incidental finding' not \
        # in cluster:
            # continue
        if len(cluster) >= 10:
            re_cluster_list = re_cluster(list(cluster), ratio=ratio)
            for clus in re_cluster_list:
                final_res.append(clus)
            ori.append(cluster)
            # final_res.append(['-'*100])
        else:
            # pass
            final_res.append(cluster)

    print(len(final_res))
    terms = []
    with open(os.path.join(args.result_dir, 'aftercut_en_ch.txt'), 'w') as f:
        for cluster in tqdm(final_res):
            print_cluster_to_file(f, cluster)
            terms += cluster
    print(len(terms))

    # with open('./precut_test.txt', 'w') as f:
        # for cluster in tqdm(ori):
            # print_cluster_to_file(f, cluster)

        
