from clustering import DisjointSet
import numpy as np
import pickle
from tqdm import tqdm

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
    with open(path, 'w') as f:
        for key in ds.group:
            cluster = ds.group[key]
            f.write('|'.join(list(cluster)) + '\n')
            total_num += len(cluster)
    print('total term numbers:', total_num)

if __name__ == '__main__':
    similarity_path = '../use_data/similarity.npy'
    indices_path = '../use_data/indices.npy'
    idx2phrase_path = '../use_data/idx2phrase.pkl'
    result_path = '../result/final_cluster_res.txt'
    similarity = read_npy(similarity_path)
    indices = read_npy(indices_path)
    idx2phrase = read_pkl(idx2phrase_path)
    ds = read_cluster_result(result_path)
    for idx, phrase in tqdm(idx2phrase.items()):
        if len(phrase) <= 3:
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
                        idx2 = np.where(similarity[idx]==sim)[0][0]
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
    save_cluster_res(ds, '../result/final_cluster_res.txt')
