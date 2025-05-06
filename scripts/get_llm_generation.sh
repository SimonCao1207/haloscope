CUDA_VISIBLE_DEVICES=2,3 \
python hal_det_llama.py \
    --dataset_name 2wikimultihopqa \
    --model_name llama3-1-8B-instruct \
    --most_likely 1 \
    --num_gene 1 \
    --gene 1 \
    --fewshots 6