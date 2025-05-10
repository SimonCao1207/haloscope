import json
import os

import numpy as np
from tqdm import tqdm

from data.wmqa import WikiMultiHopQA, get_top_sentence
from hal_det_llama import _get_index_conclusion

dataset_name = "2wikimultihopqa"
model_name = "llama3-1-8B-instruct"
inference_type = "test"

path = (
    "./data/2WMQA_cot_dev.jsonl"
    if inference_type == "test"
    else "./data/2WMQA_cot_train.jsonl"
)
data = WikiMultiHopQA(path)
data.format(fewshot=6)
dataset = data.dataset
length = len(dataset)
_log = []
scores = np.load(f"./save_for_{inference_type}/ml_{dataset_name}_bleurt_score.npy")
score_idx = 0
for i in tqdm(range(length), desc="Logging predictions"):
    prefix = "most_likely"
    file_path = f"./save_for_{inference_type}/{dataset_name}_hal_det/answers/{prefix}_hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy"
    prediction = list(np.load(file_path))
    for j in range(len(prediction)):
        prediction[j] = get_top_sentence(str(prediction[j]))
    answer = dataset[i]["cot"]
    k = _get_index_conclusion(prediction)
    num_samples = k if k > 0 else len(prediction[:k])

    _log.append(
        {
            "question": dataset[i]["question"],
            "answer": answer,
            "prediction": prediction,
            "bleurt_score": scores[score_idx : score_idx + num_samples].tolist(),
        }
    )
    score_idx += num_samples
assert score_idx == len(scores)
os.makedirs("./log", exist_ok=True)
json.dump(_log, open("./log/preds_beam_search_dev.json", "w"), indent=4)
