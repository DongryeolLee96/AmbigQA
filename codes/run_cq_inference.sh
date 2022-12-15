out="out_re"
data_dir="/media/disk1/${USER}/openQA/AmbigQA"
train_file="${data_dir}/data/ambigqa/train_cq.json"
predict_file="${data_dir}/data/ambigqa/dev_cq.json"
psg_sel_dir="${data_dir}/psg_sel/reranking_results"
#output_dir="out/ambignq-span-seq-gen"
#output_dir="out/ambignq-cq-gen"

output_dir="${data_dir}/${out}/ambignq-cq-gen"
TASK="cqg"
MAX_QUESTION_LENGTH=128
max_token_nums=512  # Originally, 1024 maximum model input tokens' nums
#checkpoint="${output_dir}/best-model-056000.pt"
#checkpoints=("${output_dir}/best-model-008000.pt" "${output_dir}/best-model-020000.pt" "${output_dir}/best-model-060000.pt")
checkpoints=("${output_dir}/best-model-036000.pt")
ablations=("without_answers" "with_predicted_answers")
#ablation="without_answers"
#ablation="with_predicted_answers"
predicted_answers_path="${data_dir}/out/ambignq-span-seq-gen/dev_predictions.json"


for checkpoint in ${checkpoints[@]}; do
    for ablation in ${ablations[@]}; do
        python3 cli.py --do_predict --task ${TASK} \
            --output_dir ${output_dir} \
            --checkpoint ${checkpoint} \
            --dpr_data_dir ${data_dir} \
            --train_file ${train_file} \
            --predict_file ${predict_file} \
            --psg_sel_dir ${psg_sel_dir} \
            --bert_name bart-large \
            --discard_not_found_answers \
            --predict_batch_size 8 \
            --max_token_nums ${max_token_nums} \
            --skip_inference \
            --wait_step 10 --ambigqa --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
            --ablation ${ablation} \
            --predicted_answers_path ${predicted_answers_path}
    python3 cli.py --do_predict --task ${TASK} \
        --output_dir ${output_dir} \
        --checkpoint ${checkpoint} \
        --dpr_data_dir ${data_dir} \
        --train_file ${train_file} \
        --predict_file ${predict_file} \
        --psg_sel_dir ${psg_sel_dir} \
        --bert_name bart-large \
        --discard_not_found_answers \
        --predict_batch_size 8 \
        --max_token_nums ${max_token_nums} \
        --skip_inference \
        --wait_step 10 --ambigqa --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
        --predicted_answers_path ${predicted_answers_path}
    done
done
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
