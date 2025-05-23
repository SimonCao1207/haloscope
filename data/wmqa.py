import json
import logging
import os
from typing import Dict, List

import spacy
from datasets import Dataset
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

from .base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiMultiHopQA(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "When did the director of film Hypocrite (Film) die?",
            "cot": "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
            "answer": "19 June 2013",
        },
        {
            "question": "Are both Kurram Garhi and Trojkrsti located in the same country?",
            "cot": "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            "answer": "no",
        },
        {
            "question": "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            "cot": "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            "answer": "no",
        },
        {
            "question": "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            "cot": "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            "answer": "Genghis Khan",
        },
        {
            "question": "Who was born first out of Martin Hodge and Ivania Martinich?",
            "cot": "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            "answer": "Martin Hodge",
        },
        {
            "question": "When did the director of film Laughter In Hell die?",
            "cot": "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            "answer": "August 25, 1963",
        },
        {
            "question": "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            "cot": "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            "answer": "Twenty Plus Two",
        },
        {
            "question": "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            "cot": "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            "answer": "Prithvipati Shah",
        },
    ]
    demo_input_template = test_input_template = (
        lambda self, ques: f"Question: {ques}\nAnswer:"
    )
    output_template = lambda self, cot, ans: f"{cot} So the answer is {ans}."

    def __init__(self, data_path: str):
        logger.info(f"Loading WikiMultiHopQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path), "r") as fin:
            for line in tqdm(fin):
                example = json.loads(line)
                question = example["question"]
                if data_path.endswith("train.jsonl"):
                    cot = example["chain_of_thought"]
                    evidences = example["evidences"]
                    unsure_statements = example["missing_knowledge_statements"]
                    intermediate_questions = example["intermediate_questions"]
                    dataset.append(
                        {
                            "question": question,
                            "cot": cot,
                            "evdidences": evidences,
                            "unsure_statements": unsure_statements,
                            "intermediate_questions": intermediate_questions,
                        }
                    )
                else:
                    answer = example["answer"]
                    dataset.append(
                        {
                            "question": question,
                            "cot": split_sentences(answer),
                        }
                    )
        self.dataset = Dataset.from_list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def split_sentences(text):
    sentences = [sent.text.strip() for sent in nlp(text).sents]
    sentences = [sent for sent in sentences if len(sent) > 0]
    return sentences


def get_top_sentence(text):
    sentences = split_sentences(text)
    return sentences[0] if len(sentences) > 0 else ""


def get_last_sentence(self, text):
    sentences = split_sentences(text)
    return sentences[-1] if len(sentences) > 0 else ""
