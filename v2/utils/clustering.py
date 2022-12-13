import numpy as np
from tqdm import tqdm
import pickle
import argparse
import ahocorasick
import os

# reference: https://stackoverflow.com/questions/3067529/a-set-union-find-algorithm
class DisjointSet(object):
    def __init__(self):
        self.leader = dict()     # maps a member to the group's leader
        self.group = dict()  # maps a group leader to the group (which is a set)
    
    def add(self, a, b):
        leadera = self.leader.get(a)   
        leaderb = self.leader.get(b)   
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_data_dir",
        default="../../clustering_220613/use_data/",
        type=str,
        help="Directory to indices and similarity and idx2phrase"
    )
    parser.add_argument(
        "--result_dir",
        default="../result/",
        type=str,
        help="Directory to save clustering result"
    )
    args = parser.parse_args()
    args.indices_path = os.path.join(args.use_data_dir, 'indices.npy')
    args.similarity_path = os.path.join(args.use_data_dir, 'similarity.npy')
    args.idx2phrase_path = os.path.join(args.use_data_dir, 'idx2phrase.pkl')
    args.result_path = os.path.join(args.result_dir, 'clustering_result.pkl')
    args.abb_lem2ori_path = os.path.join(args.use_data_dir, 'abb_lem2ori.pkl')
    # args.result_path1 = args.result_dir + 'clustering_result_noratiocut.txt'

    indices = np.load(args.indices_path)
    similarity = np.load(args.similarity_path)
    print(similarity[0])
    with open(args.idx2phrase_path, 'rb') as f:
        idx2phrase = pickle.load(f)
    with open(args.abb_lem2ori_path, 'rb') as f:
        automaton = pickle.load(f)
    print(len(automaton))
    # print('hiv' in automaton)
    in_automaton = 0
    ds = DisjointSet()
    for idxi in tqdm(range(similarity.shape[0])):
    # for idxi in tqdm(range(100000)):
        a = idx2phrase[idxi]
        if len(a) <= 3 or a in automaton:
            # print(a in automaton)
            in_automaton += 1
            continue
        else:
            # continue
            ds.add(a, a)
        for idxj in range(indices.shape[1]):
            if indices[idxi][idxj] < 0:
                continue
            if similarity[idxi, idxj] > 0.8:
                b = idx2phrase[indices[idxi][idxj]]
                if len(b) <= 3 or b in automaton:
                    continue
                ds.add(a, b)
#        break
    print(in_automaton)
    clusters = list(ds.group.values())
    terms = []
    for clu in clusters:
        terms += clu
    print(len(terms))
    with open(args.result_path, 'wb') as f:
        pickle.dump(ds.group, f)
    # with open(args.result_path1, 'w') as f:
        # f.write('\n'.join(['|'.join(list(group)) for group in list(ds.group.values())]))
    print('done')
