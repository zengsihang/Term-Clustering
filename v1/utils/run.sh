mkdir -p ../use_data
mkdir -p ../result

python generate_use_data.py --ner_path ../ner_data/ner_result.txt
python generate_faiss_index.py --model_name_or_path GanjinZero/coder_eng_pp
# union-find and exclude terms that len(term)<=3
python clustering.py
# ratio cut
python ratio_cut.py
# process short terms
python short_term_process.py
