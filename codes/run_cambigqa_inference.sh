postfix="-insertion_edit"
out="out"
data_dir="/media/disk1/${USER}/openQA/AmbigQA"
#root_dir="${data_dir}/data/ambigqa_re"
root_dir="${data_dir}/data/ambigqa"

#predict_files=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_without_answers_by_best-model-020000.json")
#predict_files=("${root_dir}/dev_cq.json")
#predict_files+=("${root_dir}/dev_cq_with_dqs_by_autoconversion.json")
#predict_files+=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_without_answers_by_best-model-020000.json")
#predict_files+=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_with_predicted_answers_by_best-model-020000.json")
#predict_files+=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_by_best-model-020000.json")

#predict_files=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_without_answers_by_best-model-020000.json")
#predict_files+=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_with_predicted_answers_by_best-model-020000.json")
#predict_files+=("${root_dir}/dev_cq_with_dqs_by_ambignq-cq-gen-dev_predictions_by_best-model-020000.json")

predict_files=("${root_dir}/dev_cq_with_dqs_by_aqs.json")

#predict_files=("${root_dir}/dev_cq_with_dqs_by_autoconversion.json")

steps=(10000 20000)

psg_sel_dir="${data_dir}/psg_sel/reranking_results"
TASK="cqa"
MAX_QUESTION_LENGTH=128
max_token_nums=512  # Originally, 1024 maximum model input tokens' nums

plm="nq-bart-large-24-0"
#for predict_file in ${predict_files[@]}; do
#    checkpoint="${data_dir}/released_ckpts/${plm}/best-model.pt"
#    output_dir="${data_dir}/${out}/${TASK}-${plm}"
#    echo ${predict_file}
#    python3 cli.py --do_predict --task ${TASK} \
#        --output_dir ${output_dir} \
#        --dpr_data_dir ${data_dir} \
#        --predict_file ${predict_file} \
#        --original_predict_file ${root_dir}/dev_cq.json \
#        --psg_sel_dir ${psg_sel_dir} \
#        --bert_name bart-large \
#        --max_token_nums ${max_token_nums} \
#        --predict_batch_size 8 \
#        --wait_step 10 --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
#        --ambigqa \
#        --checkpoint ${checkpoint}
#done

# FT-QA MODEL
for predict_file in ${predict_files[@]}; do
    for step in ${steps[@]}; do
        output_dir="${data_dir}/${out}/${TASK}-${plm}${postfix}"
        checkpoint="${output_dir}/best-model-0${step}.pt"
        python3 cli.py --do_predict --task ${TASK} \
            --output_dir ${output_dir} \
            --dpr_data_dir ${data_dir} \
            --predict_file ${predict_file} \
            --original_predict_file ${root_dir}/dev_cq.json \
            --psg_sel_dir ${psg_sel_dir} \
            --bert_name bart-large \
            --max_token_nums ${max_token_nums} \
            --predict_batch_size 8 \
            --wait_step 10 --wiki_2020 --max_question_length ${MAX_QUESTION_LENGTH} \
            --ambigqa \
            --checkpoint ${checkpoint}
    done
done

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
