CUDA_VISIBLE_DEVICES=2,3 \
python hal_det_llama_inference.py \
    --seed 41 \
    --dataset_name 2wikimultihopqa \
    --model_name llama3-1-8B-instruct \
    --most_likely 1 \
    --thres_gt 0.5 \
    --fewshots 6
