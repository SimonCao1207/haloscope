import argparse
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from data.wmqa import WikiMultiHopQA
from hal_det_llama import (
    _get_index_conclusion,
    compute_bleurt_scores,
    generate_answers,
    generate_embeddings,
    generate_prompts,
    get_correct_answers,
    load_generated_answers,
    load_model,
    post_process,
    save_generated_answers,
    seed_everything,
)
from linear_probe import NonLinearClassifier
from metric_utils import get_measures

logging.basicConfig(level=logging.INFO, format="%(message)s")


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

    model, tokenizer = load_model(args)
    os.makedirs("./save_for_test", exist_ok=True)
    if args.gene:
        os.makedirs(f"./save_for_test/{args.dataset_name}_hal_det/", exist_ok=True)
        os.makedirs(
            f"./save_for_test/{args.dataset_name}_hal_det/answers", exist_ok=True
        )

        for i in tqdm(range(0, len(dataset)), desc="Generating answers"):
            prompts = generate_prompts(
                dataset, i, args.dataset_name, None, add_fewshots=True
            )
            input_ids = tokenizer(
                prompts, return_tensors="pt", padding=True
            ).input_ids.cuda()
            predictions = generate_answers(
                model,
                tokenizer,
                args.dataset_name,
                input_ids,
                args.num_gene,
                args.most_likely,
            )
            save_generated_answers(
                args.dataset_name,
                args.model_name,
                predictions,
                i,
                args.most_likely,
                inference_type="test",
            )
    elif args.generate_gt:
        model.eval()
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
                args, model, tokenizer, predictions, all_answers
            )
            if args.dataset_name == "2wikimultihopqa":
                gts = np.concatenate([gts, all_results], 0)
            else:
                gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
        file_path = f"./save_for_test/ml_{args.dataset_name}_bleurt_score.npy"
        np.save(file_path, gts)
    else:
        # Get the embeddings of the generated question and answers.
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
            embed_generated = np.load(
                f"save_for_test/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy",
                allow_pickle=True,
            ).astype(np.float32)

        # Get the split and label (true or false) of the unlabeled data and the test data.
        score_file = f"./save_for_test/ml_{args.dataset_name}_bleurt_score.npy"

        scores = np.load(score_file)
        thres = args.thres_gt
        gt_label = np.asarray(scores > thres, dtype=np.int32)
        assert len(gt_label) == embed_generated.shape[0]

        logging.info(f"Num truthful samples: {np.sum(gt_label == 1)}")
        logging.info(f"Num hallucinated samples: {np.sum(gt_label == 0)}")

        logging.info("Inference the test set")
        checkpoint_dir = "./checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, "clf_model.pth")
        clf = NonLinearClassifier(embed_generated.shape[2], num_classes=2).cuda()
        clf.load_state_dict(torch.load(checkpoint_path))

        layer = 14

        clf.eval()
        output = clf(torch.from_numpy(embed_generated[:, layer, :]).cuda())
        pca_wild_score_binary_cls = torch.sigmoid(output)
        pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()
        if np.isnan(pca_wild_score_binary_cls).sum() > 0:
            breakpoint()
        measures = get_measures(
            pca_wild_score_binary_cls[gt_label == 1],
            pca_wild_score_binary_cls[gt_label == 0],
            plot=False,
        )
        print("test AUROC: ", measures[0])


if __name__ == "__main__":
    main()
