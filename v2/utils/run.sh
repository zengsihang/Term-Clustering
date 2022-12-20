# Clustering English terms
python generate_use_data.py --use_data_dir ../use_data/ --all_terms_path ../ner_data/all_terms.txt --device cuda:0
python generate_faiss_index.py --use_data_dir ../use_data/ --device_idx 6
python clustering.py --use_data_dir ../use_data/ --result_dir ../result/
python ratio_cut_en.py --use_data_dir ../use_data/ --result_dir ../result/

# Bilingual clustering
python get_sim_distribution.py --cluster_file ../result/cluster_res_en.txt \
                               --back_translation_file ../use_data/back_translation.txt \
                               --phrase2idx_file ../use_data/phrase2idx.pkl \
                               --embedding_file ../use_data/embedding.npy \
                               --output_file ../use_data/bt_converted_hith_sim.txt
python get_en_ch_precut.py --en_clusters_path ../result/cluster_res_en.txt \
                           --en_ch_path ../use_data/bt_converted_hith_sim.txt
python ratio_cut_en_ch.py --use_data_dir ../use_data/ --result_dir ../result/ --ratio 5.0
python short_term_process.py --use_data_dir ../use_data/ --result_dir ../result/