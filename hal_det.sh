CUDA_VISIBLE_DEVICES=6 \
python hal_det_llama.py \
    --seed 41 \
    --dataset_name 2wikimultihopqa \
    --model_name llama3-1-8B-instruct \
    --use_rouge 0 \
    --most_likely 1 \
    --weighted_svd 1 \
    --feat_loc_svd 3 \
    --regenerate_emb \
    --thres_gt 0.73 \
    --fewshots 6
