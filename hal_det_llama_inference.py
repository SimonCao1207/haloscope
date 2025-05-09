import argparse
import logging
import os
import pickle
from math import exp

import numpy as np
import torch
from tqdm import tqdm

from data.wmqa import WikiMultiHopQA, get_top_sentence
from generator import BasicGenerator
from hal_det_llama import (
    HF_NAMES,
    _get_index_conclusion,
    compute_bleurt_scores,
    generate_embeddings,
    generate_prompts,
    get_correct_answers,
    load_bleurt_model,
    load_generated_answers,
    post_process,
    save_generated_answers,
    seed_everything,
)
from linear_probe import NonLinearClassifier
from metric_utils import get_measures

logging.basicConfig(level=logging.INFO, format="%(message)s")


def find_token_range_for_sentence(sentence, tokens, start_index, vocab_dict):
    """Helper function to find which tokens belong to a sentence."""
    position = 0
    token_range = start_index

    while token_range < len(tokens):
        # Find the current token in the remaining part of the sentence
        token = tokens[token_range].strip()
        token_position = sentence[position:].find(token)
        if token_position == -1 and token in vocab_dict.keys():
            break

        # Move past this token in the sentence
        position += token_position + len(token)
        token_range += 1

    return token_range


def cal_flare_score(text, tokens, logprobs, vocab_dict):
    """
    Return confidence score for text.
    """
    sentence = get_top_sentence(text)
    token_index = 0

    if (
        "the answer is" in sentence.lower()
        or "thus" in sentence.lower()
        or "therefore" in sentence.lower()
    ):
        return 1

    # Find which tokens belong to the current sentence
    sentence_token_range = find_token_range_for_sentence(
        sentence, tokens, token_index, vocab_dict
    )
    assert sentence_token_range != token_index

    # Calculate probabilities for all tokens in the sentence
    token_probabilities = np.array(
        [exp(v) for v in logprobs[token_index:sentence_token_range]]
    )

    return np.min(token_probabilities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model_name", type=str, default="llama3-1-8B-instruct")
    parser.add_argument("--local", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="2wikimultihopqa")
    parser.add_argument("--fewshots", type=int, default=6)
    parser.add_argument("--feat_loc_svd", type=int, default=3)
    parser.add_argument("--num_gene", type=int, default=1)
    parser.add_argument("--gene", type=int, default=0)
    parser.add_argument("--most_likely", type=int, default=1)
    parser.add_argument("--generate_gt", type=int, default=0)
    parser.add_argument("--thres_gt", type=float, default=0.5)
    parser.add_argument(
        "--regenerate_embed",
        action="store_true",
        help="Whether to regenerate embeddings or load pre-existing one from local",
    )

    parser.add_argument(
        "--model_dir", type=str, default=None, help="local directory with model data"
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    path = "./data/2WMQA_cot_dev.jsonl"
    data = WikiMultiHopQA(path)
    data.format(fewshot=args.fewshots)
    dataset = data.dataset

    model_name = HF_NAMES[args.model_name]
    generator = BasicGenerator(model_name)
    base_dir = f"./save_for_test/{args.dataset_name}_hal_det/"
    os.makedirs("./save_for_test", exist_ok=True)
    if args.gene:
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(
            f"./save_for_test/{args.dataset_name}_hal_det/answers", exist_ok=True
        )
        flare_scores = []
        for i in tqdm(range(0, len(dataset)), desc="Generating answers"):
            prompts = generate_prompts(
                dataset, i, args.dataset_name, None, add_fewshots=True
            )
            return_dict = generator.generate(
                prompts,
                max_length=64,
                beam_search=True,
                output_scores=True,
                return_logprobs=True,
                return_entropies=False,
                num_return_sequences=1,
            )
            predictions = return_dict["text"]
            tokens_batch = return_dict["tokens"]
            logprobs_batch = return_dict["logprobs"]
            trim_preds = post_process(predictions, args)
            k = _get_index_conclusion(trim_preds)
            exclude_conclusion_predictions = trim_preds[:k]

            for j in range(len(exclude_conclusion_predictions)):
                flare_score = cal_flare_score(
                    exclude_conclusion_predictions[j],
                    tokens_batch[j],
                    logprobs_batch[j],
                    vocab_dict=generator.tokenizer.get_vocab(),
                )
                flare_scores.append(flare_score)

            save_generated_answers(
                args.dataset_name,
                args.model_name,
                predictions,
                i,
                args.most_likely,
                inference_type="test",
            )

        with open(f"{base_dir}/{args.dataset_name}_flare_score.pkl", "wb") as f:
            pickle.dump(flare_scores, f)

        print("flare scores len: ", len(flare_scores))

    elif args.generate_gt:
        bleurt_model, bleurt_tokenizer = load_bleurt_model()
        bleurt_model.eval()
        gts = np.zeros(0)
        for i in tqdm(range(len(dataset)), desc="Generating ground truth"):
            all_answers = get_correct_answers(dataset, i, args.dataset_name, None)
            predictions = load_generated_answers(args, i, inference_type="test")
            if not isinstance(predictions, list):
                predictions = list(predictions)
            predictions = post_process(predictions, args)
            if args.dataset_name == "2wikimultihopqa":
                k = _get_index_conclusion(predictions)
                predictions = predictions[:k]
                all_answers = all_answers[:k]
            if len(predictions) == 0:
                continue
            all_results = compute_bleurt_scores(
                args, bleurt_model, bleurt_tokenizer, predictions, all_answers
            )
            gts = np.concatenate([gts, all_results], 0)
        file_path = f"./save_for_test/ml_{args.dataset_name}_bleurt_score.npy"
        print("gts shape: ", gts.shape)
        np.save(file_path, gts)
    else:
        # Get the embeddings of the generated question and answers.
        model, tokenizer = generator.model, generator.tokenizer
        if args.regenerate_embed:
            embed_generated = generate_embeddings(
                args,
                dataset,
                None,
                len(dataset),
                model,
                tokenizer,
                inference_type="test",
            )
        else:
            logging.info("Loading embeddings from local")
            embed_generated = np.load(
                f"save_for_test/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy",
                allow_pickle=True,
            ).astype(np.float32)

        scores = np.load(f"./save_for_test/ml_{args.dataset_name}_bleurt_score.npy")
        thres = args.thres_gt
        gt_label = np.asarray(scores > thres, dtype=np.int32)
        assert len(gt_label) == embed_generated.shape[0]

        f = f"{base_dir}/{args.dataset_name}_flare_score.pkl"
        with open(f, "rb") as file:
            flare_scores = np.array(pickle.load(file))
        assert len(flare_scores) == len(gt_label)

        logging.info(f"Num truthful samples: {np.sum(gt_label == 1)}")
        logging.info(f"Num hallucinated samples: {np.sum(gt_label == 0)}")

        layer = 15
        checkpoint_dir = "./checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, f"clf_layer_{layer}.pth")
        clf = NonLinearClassifier(embed_generated.shape[2], num_classes=2).cuda()
        clf.load_state_dict(torch.load(checkpoint_path))

        clf.eval()
        output = clf(torch.from_numpy(embed_generated[:, layer, :]).cuda())
        pca_wild_score_binary_cls = torch.sigmoid(output)
        pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()
        if np.isnan(pca_wild_score_binary_cls).sum() > 0:
            breakpoint()
        halo_measures = get_measures(
            pca_wild_score_binary_cls[gt_label == 1],
            pca_wild_score_binary_cls[gt_label == 0],
            plot=True,
        )
        flare_measures = get_measures(
            flare_scores[gt_label == 1],
            flare_scores[gt_label == 0],
            plot=True,
        )
        print("Haloscope AUROC: ", halo_measures[0])
        print("Flare AUROC: ", flare_measures[0])


if __name__ == "__main__":
    main()
