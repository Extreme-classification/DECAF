#!/bin/bash
# $1 GPU DEIVCE ID
# $2 ABLATION TYPE
# $3 DATASET
# $4 VERSION
# eg. ./run_main.sh 0 DECAF LF-AmazonTitles-131K 0

export work_dir="$HOME/scratch/XC"
export PROGRAMS_DIR=$(pwd)/DECAF
export PYTHONPATH="${PYTHONPATH}:${PROGRAMS_DIR}"
export CUDA_VISIBLE_DEVICES=$1
# NUM_CORES=10
# export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES
model_type=$2
dataset=$3
version=$4

cd DECAF/configs
temp_model_data="df-xml_data"
data_dir="${work_dir}/data"
model_dir="${work_dir}/models/${model_type}/${dataset}/v_${version}"
result_dir="${work_dir}/results/${model_type}/${dataset}/v_${version}"
meta_data_folder="${data_dir}/$dataset/${temp_model_data}"
num_trees=$(python3 -c "import json; print(json.load(open('${model_type}/$dataset.json'))['DEFAULT']['num_trees'])")
mkdir -p "${meta_data_folder}"

train_file="${data_dir}/${dataset}/train.txt"
test_file="${data_dir}/${dataset}/test.txt"
trn_ft_file="${data_dir}/${dataset}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/${dataset}/trn_X_Y.txt"
tst_ft_file="${data_dir}/${dataset}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/${dataset}/tst_X_Y.txt"
num_labels=$(head -1 $train_file | awk -F ' ' '{print $3}')

run_eval() {
    log_eval_file="$result_dir/log_eval.txt"
    python -u ${PROGRAMS_DIR}/tools/evaluate.py "$data_dir/$dataset/trn_X_Y.txt" \
        "$data_dir/$dataset/tst_X_Y.txt" "$result_dir/${1}" \
        ${data_dir}/$dataset "${model_type}/$dataset.json" 2>&1 | tee -a $log_eval_file
}

merge_split_predictions() {
    echo "Merging predictions.."
    python3 ${PROGRAMS_DIR}/tools/merge_split_predictions.py $1 $2 $3 $4
}

convert() {
    perl ${PROGRAMS_DIR}/tools/convert_format.pl $1 $2 $3
    perl ${PROGRAMS_DIR}/tools/convert_format.pl $4 $5 $6
}

if [ ! -e "${trn_ft_file}" ]; then
    convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}
fi

if [ ! -e "${meta_data_folder}/v_lbs_fts_split.txt" ]; then
    echo "Fectching valid indices for training"
    python3 ${PROGRAMS_DIR}/tools/label_features_split.py $data_dir/$dataset \
        ${trn_ft_file} ${trn_lbl_file} Yf.txt ${meta_data_folder}
fi

module1() {
    mkdir -p $model_dir/surrogate
    mkdir -p $result_dir/surrogate

    PARAMS="--model_fname DECAF-S \
            --data_dir ${data_dir} \
            --model_dir ${model_dir}/surrogate \
            --result_dir ${result_dir}/surrogate \
            --config ${model_type}/$dataset.json \
            --tree_id -1 --dataset $dataset"

    log_tr_file="${result_dir}/surrogate/log_train.txt"
    log_extract_file="${result_dir}/log_extract.txt"
    python -u ${PROGRAMS_DIR}/main.py $PARAMS --mode train 2>&1 | tee $log_tr_file
    python -u ${PROGRAMS_DIR}/main.py $PARAMS --mode extract 2>&1 | tee -a $log_extract_file
}

train() {
    tree_idx=$1
    model_version=$2
    model_fname=$3
    mkdir -p $model_dir/$model_version
    mkdir -p $result_dir/$model_version

    PARAMS="--model_fname $model_fname \
        --data_dir ${data_dir} \
        --model_dir ${model_dir}/$model_version \
        --result_dir ${result_dir}/$model_version \
        --config ${model_fname}/$dataset.json \
        --tree_id $tree_idx --dataset $dataset \
        --emb_dir $model_dir/surrogate \
        --pred_fname test_predictions"

    log_tr_file="${result_dir}/$model_version/log_train.txt"
    log_pr_file="${result_dir}/$model_version/log_predict.txt"
    python -u ${PROGRAMS_DIR}/main.py $PARAMS --mode train 2>&1 | tee $log_tr_file
    python -u ${PROGRAMS_DIR}/main.py $PARAMS --mode predict 2>&1 | tee $log_pr_file
}

module1
for ((tree_idx = ${num_trees}; tree_idx > 0; tree_idx--)); do
    train $tree_idx "XC/${tree_idx}" ${model_type}
done

for ((tree_idx = ${num_trees}; tree_idx > 0; tree_idx--)); do
    echo "Taking ${tree_idx} instances in emsemble"
    python ${PROGRAMS_DIR}/tools/merge_outputs.py $result_dir/XC $tree_idx "test_predictions.npz"
    merge_split_predictions "${result_dir}" \
        "test_predictions_num_trees=${tree_idx}.npz" \
        "${meta_data_folder}" $num_labels
    run_eval "test_predictions_num_trees=${tree_idx}"
done

cd -
