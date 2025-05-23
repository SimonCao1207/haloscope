import argparse
import logging
import os
from typing import List

import evaluate
import numpy as np
import torch
from baukit import TraceDict
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm

from data.wmqa import get_top_sentence
from generator import BasicGenerator
from linear_probe import get_linear_acc
from metric_utils import get_measures, print_measures
from prepare_data import load_dataset_by_name

logging.basicConfig(level=logging.INFO, format="%(message)s")


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


HF_NAMES = {
    "llama_7B": "baffo32/decapoda-research-llama-7B-hf",
    "honest_llama_7B": "validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15",
    "alpaca_7B": "circulus/alpaca-7b",
    "vicuna_7B": "AlekseyKorshuk/vicuna-7b",
    "llama2_chat_7B": "models/Llama-2-7b-chat-hf",
    "llama2_chat_13B": "models/Llama-2-13b-chat-hf",
    "llama2_chat_70B": "meta-llama/Llama-2-70b-chat-hf",
    "llama3-1-8B-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def split_indices_and_labels(length, wild_ratio, gt_label):
    permuted_index = np.random.permutation(length)
    wild_q_indices = permuted_index[: int(wild_ratio * length)]
    # Exclude validation samples.
    wild_q_indices1 = wild_q_indices[: len(wild_q_indices) - 100]
    wild_q_indices2 = wild_q_indices[len(wild_q_indices) - 100 :]
    gt_label_test = []
    gt_label_wild = []
    gt_label_val = []
    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i : i + 1])
        elif i in wild_q_indices1:
            gt_label_wild.extend(gt_label[i : i + 1])
        else:
            gt_label_val.extend(gt_label[i : i + 1])
    return (
        np.asarray(gt_label_test),
        np.asarray(gt_label_wild),
        np.asarray(gt_label_val),
        wild_q_indices,
        wild_q_indices1,
        wild_q_indices2,
    )


def svd_embed_score(
    embed_generated_wild, gt_label, begin_k, k_span, mean=1, svd=1, weight=0
):
    """
    Perform dimensionality reduction using PCA or SVD on embeddings and evaluate their separability.

    This function iterates over a range of dimensions (`k`) and layers of the embeddings to find the
    optimal projection that maximizes the AUROC for separating true and false labels.
    """

    embed_generated = embed_generated_wild
    best_auroc_over_k = 0
    best_layer_over_k = 0
    best_scores_over_k = None
    best_projection_over_k = None
    for k in tqdm(range(begin_k, k_span)):
        best_auroc = 0
        best_layer = 0
        best_scores = None
        mean_recorded = None
        best_projection = None
        for layer in range(len(embed_generated_wild[0])):
            if mean:
                mean_recorded = embed_generated[:, layer, :].mean(0)
                centered = embed_generated[:, layer, :] - mean_recorded
            else:
                centered = embed_generated[:, layer, :]

            if not svd:
                pca_model = PCA(n_components=k, whiten=False).fit(centered)
                projection = pca_model.components_.T
                mean_recorded = pca_model.mean_
                if weight:
                    projection = pca_model.singular_values_ * projection
            else:
                _, sin_value, V_p = torch.linalg.svd(torch.from_numpy(centered).cuda())
                projection = V_p[:k, :].T.cpu().data.numpy()
                if weight:
                    projection = sin_value[:k] * projection

            scores = np.mean(np.matmul(centered, projection), -1, keepdims=True)
            assert scores.shape[1] == 1
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            # not sure about whether true and false data the direction will point to,
            # so we test both. similar practices are in the representation engineering paper
            # https://arxiv.org/abs/2310.01405
            measures1 = get_measures(
                scores[gt_label == 1], scores[gt_label == 0], plot=False
            )
            measures2 = get_measures(
                -scores[gt_label == 1], -scores[gt_label == 0], plot=False
            )

            if measures1[0] > measures2[0]:
                measures = measures1
                sign_layer = 1
            else:
                measures = measures2
                sign_layer = -1

            if measures[0] > best_auroc:
                best_auroc = measures[0]
                best_result = [100 * measures[2], 100 * measures[0]]
                best_layer = layer
                best_scores = sign_layer * scores
                best_projection = projection
                best_mean = mean_recorded
                best_sign = sign_layer

        logging.info(
            f"k: {k} | Best Result: {best_result} | Layer: {best_layer} | Mean: {mean} | SVD: {svd}"
        )

        if best_auroc > best_auroc_over_k:
            best_auroc_over_k = best_auroc
            best_result_over_k = best_result
            best_layer_over_k = best_layer
            best_k = k
            best_sign_over_k = best_sign
            best_scores_over_k = best_scores
            best_projection_over_k = best_projection
            best_mean_over_k = best_mean

    return {
        "k": best_k,
        "best_layer": best_layer_over_k,
        "best_auroc": best_auroc_over_k,
        "best_result": best_result_over_k,
        "best_scores": best_scores_over_k,
        "best_mean": best_mean_over_k,
        "best_sign": best_sign_over_k,
        "best_projection": best_projection_over_k,
    }


def _generate_prompt(dataset, i, add_fewshots=True):
    demo, case = dataset[i]["demo"], dataset[i]["case"]
    exemplars = "".join([d["case"] + "\n" for d in demo])
    if add_fewshots:
        prompt_text = exemplars
        prompt_text += 'Answer in the same format as before. Please ensure that the final sentence of the answer starts with "So the answer is".\n'
    else:
        prompt_text = ""
    prompt_text += case
    return prompt_text


def generate_prompts(
    dataset, i, dataset_name, used_indices=None, add_fewshots=True
) -> List[str]:
    """Generate the appropriate prompt based on the dataset type."""
    if dataset_name == "tydiqa" and used_indices is not None:
        question = dataset[int(used_indices[i])]["question"]
        context = dataset[int(used_indices[i])]["context"]
        prompts = (
            f"Concisely answer the following question based on the information in the given passage: \n"
            f" Passage: {context} \n Q: {question} \n A:"
        )
    elif dataset_name == "coqa":
        prompts = dataset[i]["prompt"]
    elif dataset_name == "2wikimultihopqa":
        cot = dataset[i]["cot"]
        prompts = []
        prefix = _generate_prompt(dataset, i, add_fewshots=add_fewshots)
        prompts.append(prefix)
        for j in range(1, len(cot)):
            prompts.append(f"{prefix} {' '.join(cot[:j])}")
    else:
        question = dataset[i]["question"]
        prompts = f"Answer the question concisely. Q: {question} A:"

    if isinstance(prompts, str):
        prompts = [prompts]

    return prompts


def clean_decoded(decoded, dataset_name):
    """Clean the decoded output based on dataset-specific corner cases."""
    if (
        dataset_name in ["tqa", "triviaqa"]
        and "Answer the question concisely" in decoded
    ):
        decoded = decoded.split("Answer the question concisely")[0]
    elif dataset_name == "coqa" and "Q:" in decoded:
        decoded = decoded.split("Q:")[0]
    return decoded


def generate_answers(generator, prompts, args):
    """Generate answers using the generator."""
    return_dict = generator.generate(
        prompts,
        max_length=64,
        output_scores=False,
        beam_search=True,
        return_logprobs=False,
        return_entropies=False,
    )
    predictions = return_dict["text"]
    predictions = [clean_decoded(pred, args.dataset_name) for pred in predictions]
    return predictions


def load_bleurt_model():
    model = BleurtForSequenceClassification.from_pretrained("./models/BLEURT-20")
    model = model.cuda()
    tokenizer = BleurtTokenizer.from_pretrained("./models/BLEURT-20")
    return model, tokenizer


def compute_rouge_scores(predictions, all_answers):
    """Compute ROUGE scores for predictions against all answers."""
    rouge = evaluate.load("rouge")
    num_answers, num_predictions = len(all_answers), len(predictions)
    all_results = np.zeros((num_answers, num_predictions))
    all_results1 = np.zeros((num_answers, num_predictions))
    all_results2 = np.zeros((num_answers, num_predictions))

    for anw in range(num_answers):
        results = rouge.compute(
            predictions=predictions,
            references=[all_answers[anw]] * num_predictions,
            use_aggregator=False,
        )
        all_results[anw] = results["rougeL"]
        all_results1[anw] = results["rouge1"]
        all_results2[anw] = results["rouge2"]

    return all_results, all_results1, all_results2


def compute_bleurt_scores(args, model, tokenizer, predictions, all_answers):
    """Compute BLEURT scores for predictions against all answers."""
    if args.dataset_name == "2wikimultihopqa":
        with torch.no_grad():
            inputs = tokenizer(
                all_answers, predictions, padding="longest", return_tensors="pt"
            )
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            res = model(**inputs).logits.flatten().tolist()
        return res
    else:
        num_answers, num_predictions = len(all_answers), len(predictions)
        all_results = np.zeros((num_answers, num_predictions))

        with torch.no_grad():
            for anw in range(num_answers):
                inputs = tokenizer(
                    predictions,
                    [all_answers[anw]] * num_predictions,
                    padding="longest",
                    return_tensors="pt",
                )
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                res = np.asarray(model(**inputs).logits.flatten().tolist())
                all_results[anw] = res

    return all_results


def save_scores(args, gts):
    prefix = "ml" if args.most_likely else "bg"
    metric = "rouge" if args.use_rouge else "bleurt"
    file_path = f"./{prefix}_{args.dataset_name}_{metric}_score.npy"
    np.save(file_path, gts)


def get_correct_answers(dataset, i, dataset_name, used_indices=None):
    """Retrieve all correct answers based on the dataset type."""
    if dataset_name == "tqa":
        best_answer = dataset[i]["best_answer"]
        correct_answer = dataset[i]["correct_answers"]
        all_answers = [best_answer] + correct_answer
    elif dataset_name == "triviaqa":
        all_answers = dataset[i]["answer"]["aliases"]
    elif dataset_name == "coqa":
        all_answers = dataset[i]["answer"]
    elif dataset_name == "tydiqa":
        all_answers = dataset[int(used_indices[i])]["answers"]["text"]
    elif dataset_name == "2wikimultihopqa":
        all_answers = dataset[i]["cot"]
    return all_answers


def save_generated_answers(
    dataset_name, model_name, answers, i, most_likely, inference_type="eval"
):
    """Save generated answers to a file."""
    info = "most_likely_" if most_likely else "batch_generations_"
    file_path = f"./save_for_{inference_type}/{dataset_name}_hal_det/answers/{info}hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy"
    np.save(file_path, answers)


def load_generated_answers(args, i, inference_type="eval"):
    prefix = "most_likely" if args.most_likely else "batch_generations"
    file_path = f"./save_for_{inference_type}/{args.dataset_name}_hal_det/answers/{prefix}_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
    return np.load(file_path)


def post_process(preds, args):
    preds = list(preds) if isinstance(preds, str) else preds
    if args.dataset_name == "2wikimultihopqa":
        for i in range(len(preds)):
            preds[i] = preds[i].strip()
            preds[i] = get_top_sentence(str(preds[i]))
        return preds
    else:
        preds = [p.strip() for p in preds]
        return preds


def get_embed_generated_split(
    embed_generated, feat_loc, wild_q_indices, wild_q_indices_train, wild_q_indices_val
):
    feat_indices_wild = []
    feat_indices_eval = []
    feat_indices_test = []
    for i in range(len(embed_generated)):
        if i in wild_q_indices_train:
            feat_indices_wild.extend(np.arange(i, i + 1).tolist())
        elif i in wild_q_indices_val:
            feat_indices_eval.extend(np.arange(i, i + 1).tolist())
        elif i not in wild_q_indices:
            feat_indices_test.extend(np.arange(1 * i, 1 * i + 1).tolist())
    if feat_loc == 3:
        embed_generated_wild = embed_generated[feat_indices_wild][:, 1:, :]
        embed_generated_eval = embed_generated[feat_indices_eval][:, 1:, :]
        embed_generated_test = embed_generated[feat_indices_test][:, 1:, :]
    else:
        embed_generated_wild = embed_generated[feat_indices_wild]
        embed_generated_eval = embed_generated[feat_indices_eval]
        embed_generated_test = embed_generated[feat_indices_test]

    return embed_generated_wild, embed_generated_eval, embed_generated_test


def _get_index_conclusion(predictions):
    """Get the index of the conclusion sentence in the predictions."""
    for i in range(len(predictions)):
        if (
            "so the answer is" in predictions[i].lower()
            or "thus" in predictions[i].lower()
            or "therefore" in predictions[i].lower()
        ):
            return i
    return len(predictions)


def generate_embeddings(
    args, dataset, used_indices, length, model, tokenizer, inference_type="eval"
):
    logging.info("Generating embeddings...")
    last_token_hidden_state = []
    mlp_layer_embeddings = []
    attention_head_embeddings = []

    if args.model_name == "llama3-1-8B-instruct":
        HEADS = [
            f"model.layers.{i}.self_attn.o_proj"
            for i in range(model.config.num_hidden_layers)
        ]
    else:
        HEADS = [
            f"model.layers.{i}.self_attn.head_out"
            for i in range(model.config.num_hidden_layers)
        ]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    for i in tqdm(
        range(length),
        desc=f"Generating features at feat_loc_svd={args.feat_loc_svd}",
    ):
        predictions = load_generated_answers(args, i, inference_type=inference_type)
        predictions = post_process(predictions, args)
        k = len(predictions)
        if args.dataset_name == "2wikimultihopqa":
            k = _get_index_conclusion(predictions)
            predictions = predictions[:k]
        if len(predictions) == 0:
            continue
        prompts = generate_prompts(
            dataset, i, args.dataset_name, used_indices, add_fewshots=False
        )
        prompts = prompts[:k]
        for j, pred in enumerate(predictions):
            prompts[j] += f" {pred}"

        input_ids = tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.cuda()

        with torch.no_grad():
            if args.feat_loc_svd == 3:
                output = model(input_ids, output_hidden_states=True, device_map="auto")

                hidden_states = output.hidden_states
                hidden_states = torch.stack(hidden_states, dim=0)
                # get the hidden states of the last token
                hidden_states = (
                    hidden_states.detach().to(torch.float32).cpu().numpy()[..., -1, :]
                )
                last_token_hidden_state.append(hidden_states)
            else:
                with TraceDict(model, HEADS + MLPS) as ret:
                    output = model(
                        input_ids, output_hidden_states=True, device_map="auto"
                    )
                head_wise_hidden_states = [
                    ret[head].output.squeeze().detach().cpu() for head in HEADS
                ]
                head_wise_hidden_states = (
                    torch.stack(head_wise_hidden_states, dim=0)
                    .squeeze()
                    .to(torch.float32)
                    .numpy()
                )
                mlp_wise_hidden_states = [
                    ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS
                ]
                mlp_wise_hidden_states = (
                    torch.stack(mlp_wise_hidden_states, dim=0)
                    .squeeze()
                    .to(torch.float32)
                    .numpy()
                )

                mlp_layer_embeddings.append(mlp_wise_hidden_states[:, -1, :])
                attention_head_embeddings.append(head_wise_hidden_states[:, -1, :])

    if args.feat_loc_svd == 3:
        last_token_hidden_state = np.concatenate(
            last_token_hidden_state, axis=1
        ).astype(np.float32)  # [33, num_preds, 4096]
        last_token_hidden_state = np.transpose(
            last_token_hidden_state, (1, 0, 2)
        )  # [num_preds, 33, 4096]

        np.save(
            f"save_for_{inference_type}/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy",
            last_token_hidden_state,
        )
        return last_token_hidden_state
    elif args.feat_loc_svd == 2:
        mlp_layer_embeddings = np.asarray(
            np.stack(mlp_layer_embeddings), dtype=np.float32
        )
        np.save(
            f"save_for_{inference_type}/{args.dataset_name}_hal_det/most_likely_{args.model_name}_embeddings_mlp_wise.npy",
            mlp_layer_embeddings,
        )
        return mlp_layer_embeddings
    else:
        attention_head_embeddings = np.asarray(
            np.stack(attention_head_embeddings), dtype=np.float32
        )

        np.save(
            f"save_for_{inference_type}/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy",
            attention_head_embeddings,
        )
        return attention_head_embeddings


def load_embeddings(args):
    logging.info("Loading embeddings from local...")
    if args.most_likely:
        if args.feat_loc_svd == 3:
            embed_generated = np.load(
                f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy",
                allow_pickle=True,
            ).astype(np.float32)
        elif args.feat_loc_svd == 2:
            embed_generated = np.load(
                f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_mlp_wise.npy",
                allow_pickle=True,
            ).astype(np.float32)
        else:
            embed_generated = np.load(
                f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy",
                allow_pickle=True,
            ).astype(np.float32)
    else:
        raise NotImplementedError("Batch generation is not implemented yet.")
    return embed_generated


def train_classifier_for_threshold_and_layer(
    embed_generated, best_scores, thres_wild, layer
):
    """
    Given embeddings, scores, a threshold, and a layer, split the data and train a classifier.
    Returns: clf
    """
    thres_wild_score = np.sort(best_scores)[int(len(best_scores) * thres_wild)]
    true_wild = embed_generated[:, layer, :][best_scores > thres_wild_score]
    false_wild = embed_generated[:, layer, :][best_scores <= thres_wild_score]
    embed_train = np.concatenate([true_wild, false_wild], 0)
    label_train = np.concatenate(
        [np.ones(len(true_wild)), np.zeros(len(false_wild))], 0
    )
    (
        best_acc,
        final_acc,
        (clf, best_state, best_preds, preds, labels_val),
        losses_train,
    ) = get_linear_acc(
        embed_train,
        label_train,
        embed_train,
        label_train,
        2,
        epochs=50,
        batch_size=512,
        cosine=True,
        nonlinear=True,
        learning_rate=0.05,
        weight_decay=0.0003,
    )
    return clf


def eval_clf(clf, embed_generated, layer, gt_label):
    """
    Evaluate the classifier and return the AUROC and result info.
    """
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
    return measures


def save_trained_classifier(clf, best_layer, seed, checkpoint_dir="./checkpoints"):
    """
    Save the trained classifier to a checkpoint folder and log the path.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"clf_layer_{best_layer}_seed_{seed}.pth",
    )
    torch.save(clf.state_dict(), checkpoint_path)
    logging.info(f"Model saved to {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model_name", type=str, default="llama2_chat_7B")
    parser.add_argument("--local", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="tqa")
    parser.add_argument("--fewshots", type=int, default=0)
    parser.add_argument("--num_gene", type=int, default=1)
    parser.add_argument("--gene", type=int, default=0)
    parser.add_argument("--generate_gt", type=int, default=0)
    parser.add_argument("--use_rouge", type=int, default=0)
    parser.add_argument("--weighted_svd", type=int, default=0)
    parser.add_argument("--feat_loc_svd", type=int, default=0)
    parser.add_argument("--wild_ratio", type=float, default=0.75)
    parser.add_argument("--thres_gt", type=float, default=0.5)
    parser.add_argument("--most_likely", type=int, default=0)
    parser.add_argument(
        "--regenerate_embed",
        action="store_true",
        help="Whether to regenerate embeddings or load pre-existing one from local",
    )

    parser.add_argument(
        "--model_dir", type=str, default=None, help="local directory with model data"
    )
    args = parser.parse_args()

    assert args.most_likely == 1, "Only greedy generation is supported."
    assert args.num_gene == 1, "Only one answer is generated"

    seed_everything(args.seed)
    dataset, used_indices = load_dataset_by_name(args)
    model_name = HF_NAMES[args.model_name]
    generator = BasicGenerator(model_name)
    model, tokenizer = generator.model, generator.tokenizer
    length = len(used_indices) if used_indices is not None else len(dataset)

    if args.gene:
        os.makedirs("./save_for_eval", exist_ok=True)
        os.makedirs(f"./save_for_eval/{args.dataset_name}_hal_det", exist_ok=True)
        os.makedirs(
            f"./save_for_eval/{args.dataset_name}_hal_det/answers", exist_ok=True
        )

        for i in tqdm(range(0, length), desc="Generating answers"):
            prompts = generate_prompts(dataset, i, args.dataset_name, used_indices)
            predictions = generate_answers(generator, prompts, args)
            save_generated_answers(
                args.dataset_name, args.model_name, predictions, i, args.most_likely
            )
    elif args.generate_gt:
        bleurt_model, bleurt_tokenizer = load_bleurt_model()
        bleurt_model.eval()
        gts = np.zeros(0)
        for i in tqdm(range(length), desc="Generating ground truth"):
            all_answers = get_correct_answers(
                dataset, i, args.dataset_name, used_indices
            )
            predictions = load_generated_answers(args, i)
            if not isinstance(predictions, list):
                predictions = list(predictions)
            predictions = post_process(predictions, args)
            if args.dataset_name == "2wikimultihopqa":
                k = _get_index_conclusion(predictions)
                predictions = predictions[:k]
                all_answers = all_answers[:k]
            if len(predictions) == 0:
                continue
            if args.use_rouge:
                all_results, _, _ = compute_rouge_scores(predictions, all_answers)
            else:
                all_results = compute_bleurt_scores(
                    args, bleurt_model, bleurt_tokenizer, predictions, all_answers
                )
            if args.dataset_name == "2wikimultihopqa":
                gts = np.concatenate([gts, all_results], 0)
            else:
                gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
        save_scores(args, gts)
    else:
        # Get the embeddings of the generated question and answers.
        if args.regenerate_embed:
            embed_generated = generate_embeddings(
                args, dataset, used_indices, length, model, tokenizer
            )
        else:
            embed_generated = load_embeddings(args)

        # Get the split and label (true or false) of the unlabeled data and the test data.
        score_type = "rouge" if args.use_rouge else "bleurt"
        prefix = "ml" if args.most_likely else "bg"
        score_file = f"./{prefix}_{args.dataset_name}_{score_type}_score.npy"

        bleurt_scores = np.load(score_file)
        thres = args.thres_gt
        gt_label = np.asarray(bleurt_scores > thres, dtype=np.int32)
        assert len(gt_label) == embed_generated.shape[0]

        (
            gt_label_test,
            gt_label_wild,
            gt_label_val,
            wild_q_indices,
            wild_q_indices_train,
            wild_q_indices_val,
        ) = split_indices_and_labels(len(gt_label), args.wild_ratio, gt_label)

        logging.info(
            f"Num truthful samples: {np.sum(gt_label == 1)} (test: {np.sum(gt_label_test == 1)}, val: {np.sum(gt_label_val == 1)})"
        )
        logging.info(
            f"Num hallucinated samples: {np.sum(gt_label == 0)} (test: {np.sum(gt_label_test == 0)}, val: {np.sum(gt_label_val == 0)})"
        )

        embed_generated_wild, embed_generated_eval, embed_generated_test = (
            get_embed_generated_split(
                embed_generated,
                args.feat_loc_svd,
                wild_q_indices,
                wild_q_indices_train,
                wild_q_indices_val,
            )
        )

        # returned_results = svd_embed_score(
        #     embed_generated_wild,
        #     gt_label_wild,
        #     1,
        #     11,
        #     mean=0,
        #     svd=0,
        #     weight=args.weighted_svd,
        # )
        # breakpoint()

        logging.info("Get the best hyper-parameters (k, layer) on validation set")
        returned_results = svd_embed_score(
            embed_generated_eval,
            gt_label_val,
            begin_k=1,
            k_span=11,
            mean=0,
            svd=0,
            weight=args.weighted_svd,
        )
        best_k_on_val, best_layer_on_val, best_sign_on_val = (
            returned_results["k"],
            returned_results["best_layer"],
            returned_results["best_sign"],
        )

        pca_model = PCA(n_components=best_k_on_val, whiten=False).fit(
            embed_generated_wild[:, best_layer_on_val, :]
        )
        projection = pca_model.components_.T
        if args.weighted_svd:
            projection = pca_model.singular_values_ * projection
        train_scores = np.mean(
            np.matmul(embed_generated_wild[:, best_layer_on_val, :], projection),
            -1,
            keepdims=True,
        )
        assert train_scores.shape[1] == 1
        best_train_scores = (
            np.sqrt(np.sum(np.square(train_scores), axis=1)) * best_sign_on_val
        )

        # Direct projection
        test_scores = np.mean(
            np.matmul(embed_generated_test[:, best_layer_on_val, :], projection),
            -1,
            keepdims=True,
        )

        assert test_scores.shape[1] == 1
        test_scores = np.sqrt(np.sum(np.square(test_scores), axis=1))

        measures = get_measures(
            best_sign_on_val * test_scores[gt_label_test == 1],
            best_sign_on_val * test_scores[gt_label_test == 0],
            plot=False,
        )
        print_measures(measures[0], measures[1], measures[2], "direct-projection")

        logging.info("Get the best threshold on the eval set")
        thresholds = np.linspace(0, 1, num=40)[1:-1]
        auroc_over_thres = []
        best_layer_over_thres = []
        for thres_wild in thresholds:
            best_auroc = 0
            for layer in range(len(embed_generated_wild[0])):
                clf = train_classifier_for_threshold_and_layer(
                    embed_generated_wild, best_train_scores, thres_wild, layer
                )

                measures = eval_clf(clf, embed_generated_eval, layer, gt_label_val)

                if measures[0] > best_auroc:
                    best_auroc = measures[0]
                    best_result = [100 * measures[0]]
                    best_layer_on_val = layer

            auroc_over_thres.append(best_auroc)
            best_layer_over_thres.append(best_layer_on_val)
            logging.info(
                f"Threshold: {thres_wild:.3f} | Best AUROC: {best_result[0]:.2f} | Best Layer: {best_layer_on_val}"
            )
        argmax_index = max(
            range(len(auroc_over_thres)), key=auroc_over_thres.__getitem__
        )
        logging.info(
            "The best threshold calculated on the eval set is: %f, best layer is: %d",
            thresholds[argmax_index],
            best_layer_over_thres[argmax_index],
        )

        clf = train_classifier_for_threshold_and_layer(
            embed_generated_wild,
            best_train_scores,
            thresholds[argmax_index],
            best_layer_over_thres[argmax_index],
        )

        # Save the trained model to a checkpoint folder
        save_trained_classifier(clf, best_layer_over_thres[argmax_index], args.seed)

        test_measures = eval_clf(
            clf,
            embed_generated_test,
            best_layer_over_thres[argmax_index],
            gt_label_test,
        )
        print("test AUROC: ", test_measures[0])


if __name__ == "__main__":
    main()
