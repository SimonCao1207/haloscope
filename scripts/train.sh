echo "Starting LLM generation..."
bash scripts/get_llm_generation.sh

echo "Fetching ground truth data..."
bash scripts/get_ground_truth.sh

echo "Running haloscope detection..."
bash scripts/hal_det.sh
