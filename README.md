# Term-Clustering
*Clustering terms using CODER++*

## Instruction for use
1. Put your NER vocabulary under `./ner_data/`. Each line of the file is one term.
2. `cd ./utils`, `bash run.sh`.
3. The result is in `./result/final_cluster_res.txt`.

## Steps
1. `generate_use_data.py`: Remove duplication in the NER data and construct dicts (idx2phrase, phrase2idx).
2. `generate_faiss_index.py`: Using Faiss to find the top k most similar terms for each term, including their indices and similarities. For large scale data (30M terms), we use IVFPQIndex, which is an inexact search. For small scale data (3M), we use FlatIndex, which is an exact search.
3. `clustering.py`: Use union-find algorithm to pre-cluster the terms. In this step, short terms (len(term)<=3) are excluded, which will be added back in the last step. Term pairs with similarity>0.8 are regarded as synonyms.
4. `ratio_cut.py`: We use ratio cut with elaborately defined early stopping rule to refine the clusters by cutting those large clusters into small ones. First, similarities are transformed with $S_{ij}=e^{[0.95-S_{ij}, 0]_+^2/0.15^2}$. Then we use ratio cut with default stopping rule as the size of each cluster smaller than 5. We also design an early stopping rule. If `min(mean_sim0, mean_sim1)/(cut_mean_sim)<ratio=1.5`, which means the mean in-cluster similarity of two clusters after cut divided by the mean between-cluster similarity is smaller than a ratio, then the cut is stopped.
5. `short_term_process.py`: We add back the short terms by assigning each of them to the cluster where one of the terms has the highest similarity with the short term.

## Figure
![image](https://user-images.githubusercontent.com/34975104/185576827-4e980627-3611-4f8d-978b-6df5a39c18e3.png)
