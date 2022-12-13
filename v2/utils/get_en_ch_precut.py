import ahocorasick
import argparse
from tqdm import tqdm

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


def read_en_clusters(path, ds):
    with open(path, 'r') as f:
        clusters = []
        for line in tqdm(f.readlines()):
            clusters.append(line.strip().split('|'))
    all_en_terms = ahocorasick.Automaton()
    for cluster in tqdm(clusters):
        for i in range(len(cluster)):
            ds.add(cluster[0], cluster[i])
            all_en_terms.add_word(cluster[i], cluster[i])
    return ds, all_en_terms

def read_en_ch(path, ds, all_en_terms):
    with open(path, 'r') as f:
        clusters = []
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            clusters.append([line[0].lower(), line[1]])
    for cluster in tqdm(clusters):
        if cluster[0] in all_en_terms:
            ds.add(cluster[0], cluster[1])
    return ds

def print_all_term_num(ds):
    # sum the number of ds.group.values
    print(sum([len(v) for v in ds.group.values()]))

def save_clusters(ds, path):
    # save ds.group.values to txt file
    with open(path, 'w') as f:
        f.write('\n'.join(['|'.join(list(v)) for v in ds.group.values()]))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_clusters_path', default='en_clusters.txt', type=str, help='en_clusters.txt')
    parser.add_argument('--en_ch_path', default='en_ch.txt', type=str, help='en_ch.txt')
    args = parser.parse_args()
    ds = DisjointSet()
    ds, all_en_terms = read_en_clusters(args.en_clusters_path, ds)
    print_all_term_num(ds)
    ds = read_en_ch(args.en_ch_path, ds, all_en_terms)
    print_all_term_num(ds)
    save_clusters(ds, '../result/en_ch_precut.txt')
    
