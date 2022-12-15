OUT="out2"
data_dir="/media/disk1/${USER}/openQA/AmbigQA"
root_dir="${data_dir}/data/ambigqa"
#train_file="${root_dir}/train_cq.json"
train_file="${root_dir}/train_cq_with_dqs_by_autoconversion.json"
predict_files=("${root_dir}/dev_cq.json")

psg_sel_dir="${data_dir}/psg_sel/reranking_results"
TASK="cqa"
MAX_QUESTION_LENGTH=128
gradient_accumulation_steps=5  # Orignally, 1
train_batch_size=4  # Orignally, 20
#eval_period=4000  # Originally, 500. FYI steps<2000 take for one epoch.
#num_train_epochs=5  # We set 100 epochs. (~ 108,000 steps)
eval_period=2000  # Originally, 500. FYI steps<2000 take for one epoch.
num_train_epochs=6  # We set 100 epochs. (~ 108,000 steps)
max_token_nums=512  # Originally, 1024 maximum model input tokens' nums

plm="nq-bart-large-24-0"
ablations=()
checkpoint="${data_dir}/released_ckpts/${plm}/best-model.pt"
#output_dir="${data_dir}/out/${TASK}-${plm}"
output_dir="${data_dir}/${OUT}/${TASK}-${plm}"
python3 cli.py --do_train --task ${TASK} \
    --train_file ${train_file} \
    --output_dir ${output_dir} \
    --dpr_data_dir ${data_dir} \
    --psg_sel_dir ${psg_sel_dir} \
    --bert_name bart-large \
    --discard_not_found_answers \
    --train_batch_size ${train_batch_size} \
    --num_train_epochs ${num_train_epochs} \
    --max_token_nums ${max_token_nums} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --predict_batch_size 8 \
    --eval_period ${eval_period} \
    --wait_step 10 --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
    --ambigqa \
    --skip_inference \
    --checkpoint ${checkpoint}

#plm="nq-bert-base-uncased-32-32-0"
#for predict_file in ${predict_files[@]}; do
#    checkpoint="${data_dir}/released_ckpts/${plm}/best-model.pt"
#    output_dir="${data_dir}/out/${TASK}-${plm}"
#    python3 cli.py --do_predict --task ${TASK} \
#        --output_dir ${output_dir} \
#        --dpr_data_dir ${data_dir} \
#        --predict_file ${predict_file} \
#        --psg_sel_dir ${psg_sel_dir} \
#        --bert_name bert-base-uncased \
#        --max_token_nums ${max_token_nums} \
#        --predict_batch_size 8 \
#        --wait_step 10 --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
#        --ambigqa
#        --checkpoint ${checkpoint}
#done

#plm="nq-bert-large-uncased-16-16-0"
#for predict_file in ${predict_files[@]}; do
#    checkpoint="${data_dir}/released_ckpts/${plm}/best-model.pt"
#    output_dir="${data_dir}/out/${TASK}-${plm}"
#    python3 cli.py --do_predict --task ${TASK} \
#        --output_dir ${output_dir} \
#        --dpr_data_dir ${data_dir} \
#        --predict_file ${predict_file} \
#        --psg_sel_dir ${psg_sel_dir} \
#        --bert_name bert-large-uncased \
#        --max_token_nums ${max_token_nums} \
#        --predict_batch_size 8 \
#        --wait_step 10 --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
#        --ambigqa
#        --checkpoint ${checkpoint}
#done










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
