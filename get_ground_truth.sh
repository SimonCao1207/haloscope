CUDA_VISIBLE_DEVICES=6 python hal_det_llama.py --dataset_name 2wikimultihopqa --model_name llama3-1-8B-instruct --most_likely 1 --use_rouge 0 --generate_gt 1 --fewshots 6

# CUDA_VISIBLE_DEVICES=6 python3 hal_det_llama.py --dataset_name tqa --model_name llama2_chat_7B --most_likely 0 --use_rouge 0 --generate_gt 1