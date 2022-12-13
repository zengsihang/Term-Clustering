'''
This file generates idx2phrase, phrase2idx, and embeddings for English terms.
Generating from scratch is time consuming, so we use the previous results as our basics.
'''

import argparse
import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
import ahocorasick
from transformers import AutoTokenizer, AutoModel

def read_npy(path):
    return np.load(path)


def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_base_use_data(embedding_path, idx2phrase_path, phrase2idx_path):
    # return embeddings, idx2phrase, phrase2idx
    return read_npy(embedding_path), read_pkl(idx2phrase_path), read_pkl(
        phrase2idx_path)


def read_all_terms(path):
    with open(path, 'r') as f:
        all_terms = [line.replace('\n', '').lower() for line in f.readlines()]
    print(len(all_terms))
    print(all_terms[-5:])
    return list(set(all_terms))


def get_bert_embed(phrase_list,
                   m,
                   tok,
                   device='cuda:0',
                   normalize=True,
                   summary_method="CLS",
                   tqdm_bar=True,
                   batch_size=64):
    '''
    This function is used to generate embedding vectors for phrases in phrase_list
    
    param:
        phrase_list: list of phrases to be embeded
        m: model
        tok: tokenizer
        args: mainly args.device
        normalize: normalize the embeddings or not
        summary_method: method for generating embeddings from bert output, CLS for class token or MEAN for mean pooling
        tqdm_bar: progress bar
        batch_size: batch size for bert

    return:
        embeddings in numpy array with shape (phrase_list_length, embedding_dim)
    '''
    m = m.to(device)
    input_ids = []
    for phrase in tqdm(phrase_list):
        input_ids.append(
            tok.encode_plus(
                phrase,
                max_length=32,
                add_special_tokens=True,
                truncation=True,
                pad_to_max_length=True)['input_ids'])
        # print(len(input_ids))
    m.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if tqdm_bar:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(
                input_ids[now_count:min(now_count +
                                        batch_size, count)]).to(device)
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
                    # torch.cuda.empty_cache()
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if tqdm_bar:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
            # torch.cuda.empty_cache()
        if tqdm_bar:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    # torch.cuda.empty_cache()
    return np.concatenate(output_list, axis=0)

def save_npy(path, data):
    np.save(path, data)

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def generate_use_data(args):
    # read base data
    print('Reading base data...')
    base_embeddings, base_idx2phrase, base_phrase2idx = read_base_use_data(
        args.base_embedding_path, args.base_idx2phrase_path,
        args.base_phrase2idx_path)
    # read all terms
    print('Reading all terms...')
    all_terms = read_all_terms(args.all_terms_path)
    all_base_terms = ahocorasick.Automaton()
    for idx, term in tqdm(base_idx2phrase.items()):
        all_base_terms.add_word(term, idx)

    # diff
    diff_terms = list(set(all_terms) - set(base_idx2phrase.values()))
    print('diff terms:', len(diff_terms))
    diff_idx2phrase = {idx: phrase for idx, phrase in enumerate(diff_terms)}
    diff_phrase2idx = {phrase: idx for idx, phrase in diff_idx2phrase.items()}
    # generate embeddings for diff
    print('Generating embeddings for diff terms...')
    diff_embeddings = get_bert_embed(diff_terms, args.model, args.tokenizer, args.device)
    
    # get bert embeddings
    print('Generating embeddings...')
    idx2phrase = dict()
    phrase2idx = dict()
    embeddings = np.zeros((len(all_terms), 768))
    for idx, term in tqdm(enumerate(all_terms)):
        idx2phrase[idx] = term
        phrase2idx[term] = idx
        if term in all_base_terms:
            embeddings[idx] = base_embeddings[all_base_terms.get(term)]
        else:
            embeddings[idx] = diff_embeddings[diff_phrase2idx[term]]
    # save
    print('Saving...')
    save_npy(args.embedding_path, embeddings)
    save_pkl(args.idx2phrase_path, idx2phrase)
    save_pkl(args.phrase2idx_path, phrase2idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_data_dir', type=str, default='../use_data')
    parser.add_argument('--all_terms_path', type=str, default='../ner_data/all_en_terms.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    args.base_embedding_path = os.path.join(args.use_data_dir, 'base_embedding.npy')
    args.base_idx2phrase_path = os.path.join(args.use_data_dir, 'base_idx2phrase.pkl')
    args.base_phrase2idx_path = os.path.join(args.use_data_dir, 'base_phrase2idx.pkl')
    args.embedding_path = os.path.join(args.use_data_dir, 'embedding.npy')
    args.idx2phrase_path = os.path.join(args.use_data_dir, 'idx2phrase.pkl')
    args.phrase2idx_path = os.path.join(args.use_data_dir, 'phrase2idx.pkl')
    args.model = AutoModel.from_pretrained('GanjinZero/coder_eng_pp')
    args.tokenizer = AutoTokenizer.from_pretrained('GanjinZero/coder_eng_pp')
    generate_use_data(args)
