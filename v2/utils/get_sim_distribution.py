import numpy as np
import itertools
import argparse
from scipy import stats
from tqdm import tqdm
import random

def get_sim_list(file, file_type='cluster_eng', phrase2idx=None, embedding=None):
    sim_list = []
    if file_type == 'cluster_eng':
        # cluster file
        # each line is a cluster split by '|'
        with open(file, 'r') as f:
            cluster_list = [line.strip().split('|') for line in tqdm(f.readlines())]
        for cluster in tqdm(cluster_list):
            # for each term pair in cluster, calculate the similarity
            # use itertools.combinations to get all the combinations
            comb = list(itertools.combinations(cluster, 2))
            for term1, term2 in comb:
                idx1, idx2 = phrase2idx[term1], phrase2idx[term2]
                sim = np.dot(embedding[idx1], embedding[idx2])
                sim_list.append(sim)
    elif file_type == 'back_translation':
        # for each line, line[0] is the source phrase, line[1] is the target phrase, line[3] is the similarity
        with open(file, 'r') as f:
            sim_list = [float(line.strip().split('\t')[3]) for line in tqdm(f.readlines())]
    sim_list = np.array(sim_list)
    return sim_list

def read_pkl(file):
    import pickle
    with open(file, 'rb') as f:
        return pickle.load(f)

def convert_bt_to_cluster_sim_by_percentile(sim_list_bt, sim_list_cluster):
    print('convert back translation similarity to cluster similarity by percentile')
    sim_list_bt_percentile = stats.rankdata(sim_list_bt) / len(sim_list_bt) * 100
    sim_list_cluster = random.sample(list(sim_list_cluster), 1000)
    sim_list_converted = [np.percentile(sim_list_cluster, percentile) for percentile in tqdm(sim_list_bt_percentile)]
    return sim_list_converted

def read_back_translation_file(file):
    # return row[0], row[1], row[2]
    with open(file, 'r') as f:
        return [line.strip().split('\t') for line in tqdm(f.readlines())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_file', type=str, default=None)
    parser.add_argument('--back_translation_file', type=str, default=None)
    parser.add_argument('--phrase2idx_file', type=str, default=None)
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    if args.phrase2idx_file is not None:
        phrase2idx = read_pkl(args.phrase2idx_file)
    else:
        phrase2idx = None
    if args.embedding_file is not None:
        embedding = np.load(args.embedding_file)
    else:
        embedding = None
    sim_list_cluster = get_sim_list(args.cluster_file, file_type='cluster_eng', phrase2idx=phrase2idx, embedding=embedding)
    sim_list_back_translation = get_sim_list(args.back_translation_file, file_type='back_translation')
    # percent = stats.percentileofscore(sim_list_back_translation, 0.55)
    # print(np.percentile(sim_list_cluster, percent))
    sim_list_converted = convert_bt_to_cluster_sim_by_percentile(sim_list_back_translation, sim_list_cluster)
    bt = read_back_translation_file(args.back_translation_file)
    output_data = [[line[0], line[1], line[2], str(sim_list_converted[idx])] for idx, line in enumerate(bt) if float(line[3]) >= 0.55]
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(['\t'.join(line) for line in output_data]))

        