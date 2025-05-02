import json
import logging
import os

from datasets import Dataset
from tqdm import tqdm

from .base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiMultiHopQA(BaseDataset):
    def __init__(self, data_path: str):
        logger.info(f"Loading WikiMultiHopQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path), "r") as fin:
            for line in tqdm(fin):
                example = json.loads(line)
                question = example["question"]
                ans = example["answer"]
                dataset.append(
                    {
                        "question": question,
                        "answer": ans,
                    }
                )
        self.dataset = Dataset.from_list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
