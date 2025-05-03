import argparse
import os

import evaluate
import numpy as np
import torch
from baukit import TraceDict
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import llama_iti
from linear_probe import get_linear_acc
from metric_utils import get_measures, print_measures
from prepare_data import load_dataset_by_name


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

        print(
            "k: ",
            k,
            "best result: ",
            best_result,
            "layer: ",
            best_layer,
            "mean: ",
            mean,
            "svd: ",
            svd,
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


def _generate_prompt(dataset, i):
    demo, case = dataset[i]["demo"], dataset[i]["case"]
    exemplars = "".join([d["case"] + "\n" for d in demo])
    prompt_text = exemplars
    prompt_text += 'Answer in the same format as before. Please ensure that the final sentence of the answer starts with "So the answer is".\n'
    prompt_text += case
    return prompt_text


def generate_prompt(dataset, i, dataset_name, used_indices=None):
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
        all_answers = dataset[i]["all_answers"]
        prompts = []
        prefix = _generate_prompt(dataset, i)
        prompts.append(prefix)
        for j in range(1, len(all_answers)):
            prompts.append(f"{prefix} {' '.join(all_answers[:j])}")
    else:
        question = dataset[i]["question"]
        prompts = f"Answer the question concisely. Q: {question} A:"

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


def generate_answers(model, tokenizer, dataset_name, input_ids, num_gene, most_likely):
    """Generate answers using the model."""
    answers = []
    if dataset_name == "2wikimultihopqa":
        assert num_gene == 1, "Only one answer is generated for 2wikimultihopqa."
        assert most_likely == 1, "Only most-likely generation is supported."
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        generation_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 64,
            "num_return_sequences": 1,
            "return_dict_in_generate": True,
            "num_beams": 5,
            "do_sample": False,
        }
        outputs = model.generate(**generation_args)
        input_length = input_ids.shape[-1]
        generated_tokens = outputs.sequences[:, input_length:]
        answers = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    else:
        for _ in range(num_gene):
            generation_args = {
                "input_ids": input_ids,
                "max_new_tokens": 64,
                "num_return_sequences": 1,
            }
            if most_likely:
                generation_args.update({"num_beams": 5, "do_sample": False})
            else:
                generation_args.update(
                    {
                        "do_sample": True,
                        "num_beams": 1,
                        "temperature": 0.5,
                        "top_p": 1.0,
                    }
                )

            generated = model.generate(**generation_args)
            decoded = tokenizer.decode(
                generated[0, input_ids.shape[-1] :], skip_special_tokens=True
            )
            decoded = clean_decoded(decoded, dataset_name)
            answers.append(decoded)

    return answers


def load_model(args):
    if args.generate_gt:
        model = BleurtForSequenceClassification.from_pretrained("./models/BLEURT-20")
        model = model.cuda()
        tokenizer = BleurtTokenizer.from_pretrained("./models/BLEURT-20")
    elif args.local:
        model_name = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir
        tokenizer = llama_iti.LlamaTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = llama_iti.LlamaForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = model.cuda()
    else:
        model_name = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code="falcon" in model_name,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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


def compute_bleurt_scores(model, tokenizer, predictions, all_answers):
    """Compute BLEURT scores for predictions against all answers."""
    num_answers, num_predictions = len(all_answers), len(predictions)
    all_results = np.zeros((num_answers, num_predictions))

    with torch.no_grad():
        for anw in range(num_answers):
            inputs = tokenizer(
                predictions.tolist(),
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
        all_answers = dataset[i]["all_answers"]
    return all_answers


def save_generated_answers(dataset_name, model_name, answers, i, most_likely):
    """Save generated answers to a file."""
    info = "most_likely_" if most_likely else "batch_generations_"
    file_path = f"./save_for_eval/{dataset_name}_hal_det/answers/{info}hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy"
    np.save(file_path, answers)


def load_generated_answers(args, i):
    prefix = "most_likely" if args.most_likely else "batch_generations"
    file_path = f"./save_for_eval/{args.dataset_name}_hal_det/answers/{prefix}_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
    return np.load(file_path)


def post_process(preds, args):
    preds = list(preds) if isinstance(preds, str) else preds
    if args.dataset_name == "2wikimultihopqa":
        for i in range(len(preds)):
            preds[i] = preds[i].strip()
            preds[i] = preds[i].split("\n")[0]
            return preds
    else:
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
        "--model_dir", type=str, default=None, help="local directory with model data"
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset, used_indices = load_dataset_by_name(args)
    model, tokenizer = load_model(args)
    length = len(used_indices) if used_indices is not None else len(dataset)

    if args.gene:
        if not os.path.exists(f"./save_for_eval/{args.dataset_name}_hal_det/"):
            os.mkdir(f"./save_for_eval/{args.dataset_name}_hal_det/")

        if not os.path.exists(f"./save_for_eval/{args.dataset_name}_hal_det/answers"):
            os.mkdir(f"./save_for_eval/{args.dataset_name}_hal_det/answers")

        for i in tqdm(range(0, length), desc="Generating answers"):
            prompt_text = generate_prompt(dataset, i, args.dataset_name, used_indices)
            input_ids = tokenizer(
                prompt_text, return_tensors="pt", padding=True
            ).input_ids.cuda()
            answers = generate_answers(
                model,
                tokenizer,
                args.dataset_name,
                input_ids,
                args.num_gene,
                args.most_likely,
            )
            save_generated_answers(
                args.dataset_name, args.model_name, answers, i, args.most_likely
            )
    elif args.generate_gt:
        model.eval()
        gts = np.zeros(0)
        for i in tqdm(range(length), desc="Generating ground truth"):
            all_answers = get_correct_answers(
                dataset, i, args.dataset_name, used_indices
            )
            predictions = load_generated_answers(args, i)
            predictions = post_process(predictions, args)
            if args.use_rouge:
                all_results, _, _ = compute_rouge_scores(predictions, all_answers)
            else:
                all_results = compute_bleurt_scores(
                    model, tokenizer, predictions, all_answers
                )
            gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)

        save_scores(args, gts)
    else:
        # Get the embeddings of the generated question and answers.
        embed_generated = []

        for i in tqdm(range(length), desc="Generating embeddings from block output"):
            answers = load_generated_answers(args, i)

            for anw in answers:
                prompt_text = generate_prompt(
                    dataset, i, args.dataset_name, used_indices
                )
                prompt_text += f" {anw}"
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()

                with torch.no_grad():
                    hidden_states = model(
                        input_ids, output_hidden_states=True
                    ).hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    hidden_states = hidden_states.detach().cpu().numpy()[:, -1, :]
                    embed_generated.append(hidden_states)
        embed_generated = np.asarray(np.stack(embed_generated), dtype=np.float32)
        np.save(
            f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy",
            embed_generated,
        )

        HEADS = [
            f"model.layers.{i}.self_attn.head_out"
            for i in range(model.config.num_hidden_layers)
        ]
        MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
        mlp_layer_embeddings = []
        attention_head_embeddings = []
        for i in tqdm(
            range(length),
            desc="Generating embeddings from attention head and mlp output",
        ):
            answers = load_generated_answers(args, i)
            for anw in answers:
                prompt_text = generate_prompt(
                    dataset, i, args.dataset_name, used_indices
                )
                prompt_text += f" {anw}"
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.cuda()

                with torch.no_grad():
                    with TraceDict(model, HEADS + MLPS) as ret:
                        output = model(input_ids, output_hidden_states=True)
                    head_wise_hidden_states = [
                        ret[head].output.squeeze().detach().cpu() for head in HEADS
                    ]
                    head_wise_hidden_states = (
                        torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                    )
                    mlp_wise_hidden_states = [
                        ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS
                    ]
                    mlp_wise_hidden_states = (
                        torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()
                    )

                    mlp_layer_embeddings.append(mlp_wise_hidden_states[:, -1, :])
                    attention_head_embeddings.append(head_wise_hidden_states[:, -1, :])
        mlp_layer_embeddings = np.asarray(
            np.stack(mlp_layer_embeddings), dtype=np.float32
        )
        attention_head_embeddings = np.asarray(
            np.stack(attention_head_embeddings), dtype=np.float32
        )

        np.save(
            f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy",
            attention_head_embeddings,
        )
        np.save(
            f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_embeddings_mlp_wise.npy",
            mlp_layer_embeddings,
        )

        # Get the split and label (true or false) of the unlabeled data and the test data.
        score_type = "rouge" if args.use_rouge else "bleurt"
        prefix = "ml" if args.most_likely else "bg"
        score_file = f"./{prefix}_{args.dataset_name}_{score_type}_score.npy"

        scores = np.load(score_file)
        thres = args.thres_gt
        gt_label = np.asarray(scores > thres, dtype=np.int32)

        (
            gt_label_test,
            gt_label_wild,
            gt_label_val,
            wild_q_indices,
            wild_q_indices_train,
            wild_q_indices_val,
        ) = split_indices_and_labels(length, args.wild_ratio, gt_label)

        feat_loc = args.feat_loc_svd

        if args.most_likely:
            if feat_loc == 3:
                embed_generated = np.load(
                    f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy",
                    allow_pickle=True,
                )
            elif feat_loc == 2:
                embed_generated = np.load(
                    f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_mlp_wise.npy",
                    allow_pickle=True,
                )
            else:
                embed_generated = np.load(
                    f"save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy",
                    allow_pickle=True,
                )
        else:
            raise NotImplementedError("Batch generation is not implemented yet.")

        embed_generated_wild, embed_generated_eval, embed_generated_test = (
            get_embed_generated_split(
                embed_generated,
                feat_loc,
                wild_q_indices,
                wild_q_indices_train,
                wild_q_indices_val,
            )
        )

        # returned_results = svd_embed_score(embed_generated_wild, gt_label_wild,
        #                                    1, 11, mean=0, svd=0, weight=args.weighted_svd)

        # Get the best hyper-parameters on validation set
        returned_results = svd_embed_score(
            embed_generated_eval,
            gt_label_val,
            1,
            11,
            mean=0,
            svd=0,
            weight=args.weighted_svd,
        )

        pca_model = PCA(n_components=returned_results["k"], whiten=False).fit(
            embed_generated_wild[:, returned_results["best_layer"], :]
        )
        projection = pca_model.components_.T
        if args.weighted_svd:
            projection = pca_model.singular_values_ * projection
        scores = np.mean(
            np.matmul(
                embed_generated_wild[:, returned_results["best_layer"], :], projection
            ),
            -1,
            keepdims=True,
        )
        assert scores.shape[1] == 1
        best_scores = (
            np.sqrt(np.sum(np.square(scores), axis=1)) * returned_results["best_sign"]
        )

        # Direct projection
        test_scores = np.mean(
            np.matmul(
                embed_generated_test[:, returned_results["best_layer"], :], projection
            ),
            -1,
            keepdims=True,
        )

        assert test_scores.shape[1] == 1
        test_scores = np.sqrt(np.sum(np.square(test_scores), axis=1))

        measures = get_measures(
            returned_results["best_sign"] * test_scores[gt_label_test == 1],
            returned_results["best_sign"] * test_scores[gt_label_test == 0],
            plot=False,
        )
        print_measures(measures[0], measures[1], measures[2], "direct-projection")

        # Train a linear classifier on the train set and get the best threshold when evaluating on the eval set
        thresholds = np.linspace(0, 1, num=40)[1:-1]
        normalizer = lambda x: x / (
            np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )
        auroc_over_thres = []
        best_layer_over_thres = []
        for thres_wild in thresholds:
            best_auroc = 0
            for layer in range(len(embed_generated_wild[0])):
                thres_wild_score = np.sort(best_scores)[
                    int(len(best_scores) * thres_wild)
                ]
                true_wild = embed_generated_wild[:, layer, :][
                    best_scores > thres_wild_score
                ]
                false_wild = embed_generated_wild[:, layer, :][
                    best_scores <= thres_wild_score
                ]

                embed_train = np.concatenate([true_wild, false_wild], 0)
                label_train = np.concatenate(
                    [np.ones(len(true_wild)), np.zeros(len(false_wild))], 0
                )

                ## gt training, saplma
                # embed_train = embed_generated_wild[:,layer,:]
                # label_train = gt_label_wild
                ## gt training, saplma

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
                    print_ret=True,
                    batch_size=512,
                    cosine=True,
                    nonlinear=True,
                    learning_rate=0.05,
                    weight_decay=0.0003,
                )

                clf.eval()
                output = clf(torch.from_numpy(embed_generated_eval[:, layer, :]).cuda())
                pca_wild_score_binary_cls = torch.sigmoid(output)
                pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()

                if np.isnan(pca_wild_score_binary_cls).sum() > 0:
                    breakpoint()
                measures = get_measures(
                    pca_wild_score_binary_cls[gt_label_val == 1],
                    pca_wild_score_binary_cls[gt_label_val == 0],
                    plot=False,
                )

                if measures[0] > best_auroc:
                    best_auroc = measures[0]
                    best_result = [100 * measures[0]]
                    best_layer = layer

            auroc_over_thres.append(best_auroc)
            best_layer_over_thres.append(best_layer)
            print(
                "thres: ",
                thres_wild,
                "best result: ",
                best_result,
                "best_layer: ",
                best_layer,
            )
        argmax_index = max(
            range(len(auroc_over_thres)), key=auroc_over_thres.__getitem__
        )
        print(
            "the best threshold calculated on the eval set is: ",
            thresholds[argmax_index],
            "best layer is: ",
            best_layer_over_thres[argmax_index],
        )

        # Get the result on the test set
        thres_wild_score = np.sort(best_scores)[
            int(len(best_scores) * thresholds[argmax_index])
        ]
        true_wild = embed_generated_wild[:, best_layer_over_thres[argmax_index], :][
            best_scores > thres_wild_score
        ]
        false_wild = embed_generated_wild[:, best_layer_over_thres[argmax_index], :][
            best_scores <= thres_wild_score
        ]

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
            print_ret=True,
            batch_size=512,
            cosine=True,
            nonlinear=True,
            learning_rate=0.05,
            weight_decay=0.0003,
        )

        clf.eval()
        output = clf(
            torch.from_numpy(
                embed_generated_test[:, best_layer_over_thres[argmax_index], :]
            ).cuda()
        )
        pca_wild_score_binary_cls = torch.sigmoid(output)
        pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()
        if np.isnan(pca_wild_score_binary_cls).sum() > 0:
            breakpoint()
        measures = get_measures(
            pca_wild_score_binary_cls[gt_label_test == 1],
            pca_wild_score_binary_cls[gt_label_test == 0],
            plot=False,
        )
        print("test AUROC: ", measures[0])


if __name__ == "__main__":
    main()
