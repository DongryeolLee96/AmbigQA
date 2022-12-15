data_dir="/media/disk1/ksk5693/openQA/AmbigQA"
predict_file="${data_dir}/data/ambigqa/dev_light.json"
checkpoint="${data_dir}/checkpoint/ambignq-bart-large-12-0/best-model.pt"
psg_sel_dir="${data_dir}/psg_sel/reranking_results"
output_dir="out/ambignq-span-seq-gen"
python3 cli.py --do_predict --task qa --checkpoint ${checkpoint} \
    --output_dir ${output_dir} \
    --dpr_data_dir ${data_dir} \
    --predict_file ${predict_file} \
    --psg_sel_dir ${psg_sel_dir} \
    --bert_name bart-large \
    --discard_not_found_answers \
    --train_batch_size 20 --predict_batch_size 8 \
    --eval_period 500 --wait_step 10 --ambigqa --wiki_2020 --max_answer_length 25
#     --skip_db_load \

#     --train_batch_size 20 --predict_batch_size 40 \

# Guide
# python3 cli.py --task qa --checkpoint out/ambignq-span-seq-gen \
#     --dpr_data_dir ${data_dir} \
#     --train_file data/ambigqa/train_light.json \
#     --predict_file data/ambigqa/dev_light.json \
#     --psg_sel_dir out/nq-span-selection \
#     --bert_name bart-large \
#     --discard_not_found_answers \
#     --train_batch_size 20 --predict_batch_size 40 \
#     --eval_period 500 --wait_step 10 --ambigqa --wiki_2020 --max_answer_length 25


# for i in 0 1 2 3 4 5 6 7 8 9 ; 
# do
# 	python3 cli.py --do_predict --bert_name bert-base-uncased --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --task dpr --predict_file ${predict_file} --predict_batch_size 3200 --db_index $i --wiki_2020 --do_prepro_only
# done
# python3 cli.py --bert_name bert-base-uncased --output_dir out/dpr --dpr_data_dir ${data_dir} --do_predict --task dpr --predict_batch_size 3200 --predict_file ${predict_file} --wiki_2020

# python3 cli.py --do_predict --task qa --output_dir out/nq-span-selection \
#     --dpr_data_dir ${data_dir} \
#     --predict_file ${predict_file} \
#     --bert_name bert-base-uncased \
#     --predict_batch_size 32 --save_psg_sel_only --wiki_2020
