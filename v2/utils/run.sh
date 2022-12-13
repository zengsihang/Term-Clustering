python generate_faiss_index.py --use_data_dir ../use_data/ --device_idx 6
python clustering.py --use_data_dir ../use_data/ --result_dir ../result/
python ratio_cut_en.py --use_data_dir ../use_data/ --result_dir ../result/

