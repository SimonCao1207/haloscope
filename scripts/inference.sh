echo "Starting LLM generation..."
bash scripts/get_llm_generation_inference.sh

echo "Fetching ground truth data..."
bash scripts/get_ground_truth_inference.sh

echo "Running haloscope detection..."
bash scripts/hal_det_inference.sh
