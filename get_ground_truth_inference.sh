CUDA_VISIBLE_DEVICES=6,7 \
python hal_det_llama_inference.py \
    --dataset_name 2wikimultihopqa \
    --model_name llama3-1-8B-instruct \
    --generate_gt 1 \
    --fewshots 6