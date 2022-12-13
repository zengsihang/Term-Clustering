import pickle

import ahocorasick
import numpy as np
from clustering import DisjointSet
from tqdm import tqdm
import argparse
import os

def read_npy(path):
    return np.load(path)


def read_cluster_result(path):
    # read cluster result from path
    # it is a txt file
    # each row is a cluster, terms split by |
    # return a disjointset of these clusters
    ds = DisjointSet()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            terms = line.replace('\n', '').split('|')
            for term in terms:
                ds.add(terms[0], term)
    return ds


def read_pkl(path):
    # read pkl file
    # return a dict
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_cluster_res(ds, path):
    # save cluster result to path
    # it is a txt file
    # each row is a cluster, terms split by |
    # print total term numbers
    total_num = 0
    cluster_num = 0
    with open(path, 'w') as f:
        for key in ds.group:
            cluster = ds.group[key]
            f.write('|'.join(list(cluster)) + '\n')
            total_num += len(cluster)
            cluster_num += 1
    print('total term numbers:', total_num)
    print('total cluster numbers:', cluster_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='cluster_result.txt')
    parser.add_argument('--use_data_dir', type=str, default='use_data')
    args = parser.parse_args()

    similarity_path = os.path.join(args.use_data_dir, 'similarity.npy')
    indices_path = os.path.join(args.use_data_dir, 'indices.npy')
    idx2phrase_path = os.path.join(args.use_data_dir, 'idx2phrase.pkl')
    result_path = os.path.join(args.result_dir, 'aftercut_en_ch.txt')
    abb_len2ori_path = os.path.join(args.use_data_dir, 'abb_lem2ori.pkl')
    save_path = os.path.join(args.result_dir, 'final_cluster_res_en_ch_include_short.txt')
    similarity = read_npy(similarity_path)
    indices = read_npy(indices_path)
    idx2phrase = read_pkl(idx2phrase_path)
    automaton = read_pkl(abb_len2ori_path)
    ds = read_cluster_result(result_path)
    all_phrases = ahocorasick.Automaton()
    for phrase in tqdm(idx2phrase.values()):
        # print(phrase)
        all_phrases.add_word(phrase, '')
    for idx, phrase in tqdm(idx2phrase.items()):
        if phrase in automaton:
            phrase1 = automaton.get(phrase)
            if phrase1 in all_phrases:
                ds.add(phrase, phrase1)
            else:
                ds.add(phrase, phrase)
        elif len(phrase) <= 3:
            # put it to the same cluster with highest similarity
            # highest!!!
            if np.max(similarity[idx]) > 0.8:
                # find argmax
                idx_max = np.argmax(similarity[idx])
                phrase2 = idx2phrase[indices[idx, idx_max]]
                if phrase2 == phrase:
                    # find arg second max
                    flat_sim = similarity[idx].flatten()
                    flat_sim.sort()
                    sim = flat_sim[-2]
                    if sim > .8:
                        idx2 = np.where(similarity[idx] == sim)[0][0]
                        phrase2 = idx2phrase[indices[idx, idx2]]
                        ds.add(phrase, phrase2)
                    else:
                        ds.add(phrase, phrase)
                else:
                    ds.add(phrase, phrase2)
            else:
                ds.add(phrase, phrase)
        else:
            continue
    save_cluster_res(ds, save_path)
