# Term-Clustering
*Clustering terms using CODER++, v2*

## Instruction for use
1. Put your NER vocabulary under `./ner_data/`. Each line of the file is one term.
2. Put your previous embedding, idx2phrase, phrase2idx files in `./use_data/` as `base_embedding.npy`, `base_idx2phrase.pkl`, and `base_phrase2idx.pkl`.
3. Put your "abbreviation/lemma to original words" mapping file in `./use_data/` as `abb_lem2ori.pkl`. It's an Automaton.
4. Put your backtranslation file in `./use_data/back_translation.txt`.
5. `cd ./utils`, `bash run.sh`.
6. The result is in `./result/final_cluster_res_en_ch_include_short.txt`.

## Steps
1. `generate_use_data.py`: Construct dicts (idx2phrase and phrase2idx) and embeddings on the basis of previous results to save time. You can also generate these data from scratch.
2. `generate_faiss_index.py`: Using Faiss to find the top k most similar terms for each term, including their indices and similarities. For large scale data (30M terms), we use IVFPQIndex, which is an inexact search. For small scale data (3M), we use FlatIndex, which is an exact search.
3. `clustering.py`: Use union-find algorithm to pre-cluster the terms. In this step, short terms (len(term)<=3), abbreviations, and lemmatization are excluded, which will be added back in the last step. Term pairs with similarity>0.8 are regarded as synonyms.
4. `ratio_cut_en.py`: We use ratio cut with elaborately defined early stopping rule to refine the clusters by cutting those large clusters into small ones. We use ratio cut with default stopping rule as the size of each cluster smaller than 5. We also design an early stopping rule. If `min(mean_sim0, mean_sim1)/(cut_mean_sim)<ratio=1.5`, which means the mean in-cluster similarity of two clusters after cut divided by the mean between-cluster similarity is smaller than a ratio, then the cut is stopped.
5. `get_sim_distribution.py`: Align the distribution of backtranslation similarities and the clustering similarities by quantile. Exclude translation pairs with backtranslation similarity < 0.55.
6. `get_en_ch_precut.py`: Merge the translation result and English clustering result, creating new bilingual clusters.
7. `ratio_cut_en_ch.py`: Use ratio cut to post-process the bilingual clusters. Similar to step 4, however the stopping rules are changed into size 10 and ratio 5.0.
8. `short_term_process.py`: We add back the short terms by assigning each of them to the cluster where one of the terms has the highest similarity with the short term.

## Figure

