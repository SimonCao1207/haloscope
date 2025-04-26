import argparse
import os

import evaluate
import numpy as np
import torch
from baukit import TraceDict
from sklearn.decomposition import PCA
from tqdm import tqdm

import llama_iti
from metric_utils import get_measures, print_measures
from prepapre_data import load_dataset_by_name


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'honest_llama_7B': 'validation/results_dump/llama_7B_seed_42_top_48_heads_alpha_15',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'models/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'models/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3-1-8B-instruct' : 'models/Llama-3.1-8B-Instruct',
}


def split_indices_and_labels(length, wild_ratio, gt_label):
    permuted_index = np.random.permutation(length)
    wild_q_indices = permuted_index[:int(wild_ratio * length)]
    # Exclude validation samples.
    wild_q_indices1 = wild_q_indices[:len(wild_q_indices) - 100]
    wild_q_indices2 = wild_q_indices[len(wild_q_indices) - 100:]
    gt_label_test = []
    gt_label_wild = []
    gt_label_val = []
    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i: i+1])
        elif i in wild_q_indices1:
            gt_label_wild.extend(gt_label[i: i+1])
        else:
            gt_label_val.extend(gt_label[i: i+1])
    return (
        np.asarray(gt_label_test),
        np.asarray(gt_label_wild),
        np.asarray(gt_label_val),
        wild_q_indices,
        wild_q_indices1,
        wild_q_indices2
    )

def svd_embed_score(
    embed_generated_wild, gt_label, begin_k, k_span, mean=1, svd=1, weight=0
):
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

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--num_gene', type=int, default=1)
    parser.add_argument('--gene', type=int, default=0)
    parser.add_argument('--generate_gt', type=int, default=0)
    parser.add_argument('--use_rouge', type=int, default=0)
    parser.add_argument('--weighted_svd', type=int, default=0)
    parser.add_argument('--feat_loc_svd', type=int, default=0)
    parser.add_argument('--wild_ratio', type=float, default=0.75)
    parser.add_argument('--thres_gt', type=float, default=0.5)
    parser.add_argument('--most_likely', type=int, default=0)

    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()

    MODEL = HF_NAMES[args.model_name] if not args.model_dir else args.model_dir
    dataset, used_indices = load_dataset_by_name(args)
    length = len(used_indices) if used_indices is not None else len(dataset)

    if args.gene:
        tokenizer = llama_iti.LlamaTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = llama_iti.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                           device_map="auto").cuda()

        begin_index = 0
        end_index = length

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}_hal_det/'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}_hal_det/')

        if not os.path.exists(f'./save_for_eval/{args.dataset_name}_hal_det/answers'):
            os.mkdir(f'./save_for_eval/{args.dataset_name}_hal_det/answers')

        period_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n']]
        period_token_id += [tokenizer.eos_token_id]

        for i in range(begin_index, end_index):
            answers = [None] * args.num_gene
            if args.dataset_name == 'tydiqa' and used_indices is not None:
                question = dataset[int(used_indices[i])]['question']
                prompt = tokenizer(
                    "Concisely answer the following question based on the information in the given passage: \n" + \
                    " Passage: " + dataset[int(used_indices[i])]['context'] + " \n Q: " + question + " \n A:",
                    return_tensors='pt').input_ids.cuda()
            elif args.dataset_name == 'coqa':
                prompt = tokenizer(
                    dataset[i]['prompt'], return_tensors='pt').input_ids.cuda()
            else:
                question = dataset[i]['question']
                prompt = tokenizer(f"Answer the question concisely. Q: {question}" + " A:", return_tensors='pt').input_ids.cuda()
            for gen_iter in range(args.num_gene):
                if args.most_likely:
                    generated = model.generate(prompt,
                                                num_beams=5,
                                                num_return_sequences=1,
                                                do_sample=False,
                                                max_new_tokens=64,
                                               )
                else:
                    generated = model.generate(prompt,
                                                do_sample=True,
                                                num_return_sequences=1,
                                                num_beams=1,
                                                max_new_tokens=64,
                                                temperature=0.5,
                                                top_p=1.0)


                decoded = tokenizer.decode(generated[0, prompt.shape[-1]:],
                                           skip_special_tokens=True)
                if args.dataset_name == 'tqa' or args.dataset_name == 'triviaqa':
                    # corner case.
                    if 'Answer the question concisely' in decoded:
                        print('#####error')
                        print(decoded.split('Answer the question concisely')[1])
                        print('#####error')
                        decoded = decoded.split('Answer the question concisely')[0]
                if args.dataset_name == 'coqa':
                    if 'Q:' in decoded:
                        print('#####error')
                        print(decoded.split('Q:')[1])
                        print('#####error')
                        decoded = decoded.split('Q:')[0]
                print(decoded)
                answers[gen_iter] = decoded


            print('sample: ', i)
            if args.most_likely:
                info = 'most_likely_'
            else:
                info = 'batch_generations_'
            print("Saving answers")
            np.save(f'./save_for_eval/{args.dataset_name}_hal_det/answers/' + info + f'hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy',
                    answers)
    elif args.generate_gt:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        model = BleurtForSequenceClassification.from_pretrained('./models/BLEURT-20').cuda()
        tokenizer = BleurtTokenizer.from_pretrained('./models/BLEURT-20')
        model.eval()

        rouge = evaluate.load('rouge')
        gts = np.zeros(0)
        if args.dataset_name == 'tydiqa':
            length = len(used_indices)
        else:
            length = len(dataset)
        for i in range(length):
            if args.dataset_name == 'tqa':
                best_answer = dataset[i]['best_answer']
                correct_answer = dataset[i]['correct_answers']
                all_answers = [best_answer] + correct_answer
            elif args.dataset_name == 'triviaqa':
                all_answers = dataset[i]['answer']['aliases']
            elif args.dataset_name == 'coqa':
                all_answers = dataset[i]['answer']
            elif args.dataset_name == 'tydiqa':
                all_answers = dataset[int(used_indices[i])]['answers']['text']

            if args.most_likely:
                answers = np.load(
                    f'./save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
            else:
                answers = np.load(
                    f'./save_for_eval/{args.dataset_name}_hal_det/answers/batch_generations_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
            # get the gt.
            if args.use_rouge:

                predictions = answers
                all_results = np.zeros((len(all_answers), len(predictions)))
                all_results1 = np.zeros((len(all_answers), len(predictions)))
                all_results2 = np.zeros((len(all_answers), len(predictions)))
                for anw in range(len(all_answers)):
                    results = rouge.compute(predictions=predictions,
                                            references=[all_answers[anw]] * len(predictions),
                                            use_aggregator=False)
                    all_results[anw] = results['rougeL']
                    all_results1[anw] = results['rouge1']
                    all_results2[anw] = results['rouge2']

                # breakpoint()
                gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)

                if i % 50 == 0:
                    print("samples passed: ", i)
            else:

                predictions = answers
                all_results = np.zeros((len(all_answers), len(predictions)))
                with torch.no_grad():
                    for anw in range(len(all_answers)):
                        inputs = tokenizer(predictions.tolist(), [all_answers[anw]] * len(predictions),
                                           padding='longest', return_tensors='pt')
                        for key in list(inputs.keys()):
                            inputs[key] = inputs[key].cuda()
                        res = np.asarray(model(**inputs).logits.flatten().tolist())
                        all_results[anw] = res
                gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
                if i % 10 == 0:
                    print("samples passed: ", i)
        # breakpoint()
        if args.most_likely:
            if args.use_rouge:
                np.save(f'./ml_{args.dataset_name}_rouge_score.npy', gts)
            else:
                np.save(f'./ml_{args.dataset_name}_bleurt_score.npy', gts)
        else:
            if args.use_rouge:
                np.save(f'./bg_{args.dataset_name}_rouge_score.npy', gts)
            else:
                np.save(f'./bg_{args.dataset_name}_bleurt_score.npy', gts)

    else:
        tokenizer = llama_iti.LlamaTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = llama_iti.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True,
                                                           torch_dtype=torch.float16,
                                                           device_map="auto").cuda()

        # firstly get the embeddings of the generated question and answers.
        embed_generated = []

        for i in tqdm(range(length)):
            if args.dataset_name == 'tydiqa':
                question = dataset[int(used_indices[i])]['question']
            else:
                question = dataset[i]['question']
            answers = np.load(
                f'save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')

            for anw in answers:

                if args.dataset_name == 'tydiqa':
                    prompt = tokenizer(
                        "Concisely answer the following question based on the information in the given passage: \n" + \
                        " Passage: " + dataset[int(used_indices[i])]['context'] + " \n Q: " + question + " \n A:",
                        return_tensors='pt').input_ids.cuda()
                elif args.dataset_name == 'coqa':
                    prompt = tokenizer(dataset[i]['prompt'] + anw, return_tensors='pt').input_ids.cuda()
                else:
                    prompt = tokenizer(
                        f"Answer the question concisely. Q: {question}" + " A:" + anw,
                        return_tensors='pt').input_ids.cuda()
                with torch.no_grad():
                    hidden_states = model(prompt, output_hidden_states=True).hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    hidden_states = hidden_states.detach().cpu().numpy()[:, -1, :]
                    embed_generated.append(hidden_states)
        embed_generated = np.asarray(np.stack(embed_generated), dtype=np.float32)
        np.save(f'save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy', embed_generated)

        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
        mlp_layer_embeddings = []
        attention_head_embeddings = []
        for i in tqdm(range(length)):
            if args.dataset_name == 'tydiqa':
                question = dataset[int(used_indices[i])]['question']
            else:
                question = dataset[i]['question']


            answers = np.load(
                f'save_for_eval/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy')
            for anw in answers:
                if args.dataset_name == 'tydiqa':
                    prompt = tokenizer(
                        "Concisely answer the following question based on the information in the given passage: \n" + \
                        " Passage: " + dataset[int(used_indices[i])]['context'] + " \n Q: " + question + " \n A:",
                        return_tensors='pt').input_ids.cuda()
                elif args.dataset_name == 'coqa':
                    prompt = tokenizer(dataset[i]['prompt'] + anw, return_tensors='pt').input_ids.cuda()
                else:
                    prompt = tokenizer(
                        f"Answer the question concisely. Q: {question}" + " A:" + anw,
                        return_tensors='pt').input_ids.cuda()

                with torch.no_grad():
                    with TraceDict(model, HEADS + MLPS) as ret:
                        output = model(prompt, output_hidden_states=True)
                    head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                    mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
                    mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

                    mlp_layer_embeddings.append(mlp_wise_hidden_states[:, -1, :])
                    attention_head_embeddings.append(head_wise_hidden_states[:, -1, :])
        mlp_layer_embeddings = np.asarray(np.stack(mlp_layer_embeddings), dtype=np.float32)
        attention_head_embeddings = np.asarray(np.stack(attention_head_embeddings), dtype=np.float32)

        np.save(f'save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy', attention_head_embeddings)
        np.save(f'save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_embeddings_mlp_wise.npy',  mlp_layer_embeddings)



        # get the split and label (true or false) of the unlabeled data and the test data.
        if args.use_rouge:
            if args.most_likely:
                gts = np.load(f'./ml_{args.dataset_name}_rouge_score.npy')
            else:
                gts_bg = np.load(f'./bg_{args.dataset_name}_rouge_score.npy')
        else:
            if args.most_likely:
                gts = np.load(f'./ml_{args.dataset_name}_bleurt_score.npy')
            else:
                gts_bg = np.load(f'./bg_{args.dataset_name}_bleurt_score.npy')

        thres = args.thres_gt

        if args.most_likely:
            gt_label = np.asarray(gts> thres, dtype=np.int32)
        else:
            gt_label_bg = np.asarray(gts_bg > thres, dtype=np.int32)


        if args.dataset_name == 'tydiqa':
            length = len(used_indices)
        else:
            length = len(dataset)

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
                embed_generated = np.load(f'save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy',
                                  allow_pickle=True)
            elif feat_loc == 2:
                embed_generated = np.load(
                    f'save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_mlp_wise.npy',
                    allow_pickle=True)
            else:
                embed_generated = np.load(
                    f'save_for_eval/{args.dataset_name}_hal_det/most_likely_{args.model_name}_gene_embeddings_head_wise.npy',
                    allow_pickle=True)
            feat_indices_wild = []
            feat_indices_eval = []

            if args.dataset_name == 'tydiqa':
                length = len(used_indices)
            else:
                length = len(dataset)

            for i in range(length):
                if i in wild_q_indices_train:
                    feat_indices_wild.extend(np.arange(i, i + 1).tolist())
                elif i in wild_q_indices_val:
                    feat_indices_eval.extend(np.arange(i, i + 1).tolist())
            if feat_loc == 3:
                embed_generated_wild = embed_generated[feat_indices_wild][:, 1:, :]
                embed_generated_eval = embed_generated[feat_indices_eval][:, 1:, :]
            else:
                embed_generated_wild = embed_generated[feat_indices_wild]
                embed_generated_eval = embed_generated[feat_indices_eval]

        # returned_results = svd_embed_score(embed_generated_wild, gt_label_wild,
        #                                    1, 11, mean=0, svd=0, weight=args.weighted_svd)

        # get the best hyper-parameters on validation set
        returned_results = svd_embed_score(embed_generated_eval, gt_label_val,
                                           1, 11, mean=0, svd=0, weight=args.weighted_svd)

        pca_model = PCA(n_components=returned_results['k'], whiten=False).fit(embed_generated_wild[:,returned_results['best_layer'],:])
        projection = pca_model.components_.T
        if args.weighted_svd:
            projection = pca_model.singular_values_ * projection
        scores = np.mean(np.matmul(embed_generated_wild[:,returned_results['best_layer'],:], projection), -1, keepdims=True)
        assert scores.shape[1] == 1
        best_scores = np.sqrt(np.sum(np.square(scores), axis=1)) * returned_results['best_sign']



        # direct projection
        feat_indices_test = []

        for i in range(length):
            if i not in wild_q_indices:
                feat_indices_test.extend(np.arange(1 * i, 1 * i + 1).tolist())
        if feat_loc == 3:
            embed_generated_test = embed_generated[feat_indices_test][:, 1:, :]
        else:
            embed_generated_test = embed_generated[feat_indices_test]

        test_scores = np.mean(np.matmul(embed_generated_test[:,returned_results['best_layer'],:],
                                   projection), -1, keepdims=True)

        assert test_scores.shape[1] == 1
        test_scores = np.sqrt(np.sum(np.square(test_scores), axis=1))

        measures = get_measures(returned_results['best_sign'] * test_scores[gt_label_test == 1],
                                 returned_results['best_sign'] *test_scores[gt_label_test == 0], plot=False)
        print_measures(measures[0], measures[1], measures[2], 'direct-projection')


        thresholds = np.linspace(0,1, num=40)[1:-1]
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        auroc_over_thres = []
        best_layer_over_thres = []
        from linear_probe import get_linear_acc
        for thres_wild in thresholds:
            best_auroc = 0
            for layer in range(len(embed_generated_wild[0])):
                thres_wild_score = np.sort(best_scores)[int(len(best_scores) * thres_wild)]
                true_wild = embed_generated_wild[:,layer,:][best_scores > thres_wild_score]
                false_wild = embed_generated_wild[:,layer,:][best_scores <= thres_wild_score]

                embed_train = np.concatenate([true_wild,false_wild],0)
                label_train = np.concatenate([np.ones(len(true_wild)),
                                              np.zeros(len(false_wild))], 0)


                ## gt training, saplma
                # embed_train = embed_generated_wild[:,layer,:]
                # label_train = gt_label_wild
                ## gt training, saplma



                best_acc, final_acc, (
                clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
                embed_train,
                label_train,
                embed_train,
                label_train,
                2, epochs = 50,
                print_ret = True,
                batch_size=512,
                cosine=True,
                nonlinear = True,
                learning_rate = 0.05,
                weight_decay = 0.0003)



                clf.eval()
                output = clf(torch.from_numpy(
                    embed_generated_eval[:, layer, :]).cuda())
                pca_wild_score_binary_cls = torch.sigmoid(output)


                pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()

                if np.isnan(pca_wild_score_binary_cls).sum() > 0:
                    breakpoint()
                measures = get_measures(pca_wild_score_binary_cls[gt_label_val == 1],
                                        pca_wild_score_binary_cls[gt_label_val == 0], plot=False)

                if measures[0] > best_auroc:
                    best_auroc = measures[0]
                    best_result = [100 * measures[0]]
                    best_layer = layer

            auroc_over_thres.append(best_auroc)
            best_layer_over_thres.append(best_layer)
            print('thres: ', thres_wild, 'best result: ', best_result, 'best_layer: ', best_layer)
        argmax_index = max(range(len(auroc_over_thres)), key=auroc_over_thres.__getitem__)
        print("the best threshold calculated on the eval set is: ", thresholds[argmax_index], "best layer is: ", best_layer_over_thres[argmax_index])

        # get the result on the test set
        thres_wild_score = np.sort(best_scores)[int(len(best_scores) * thresholds[argmax_index])]
        true_wild = embed_generated_wild[:, best_layer_over_thres[argmax_index], :][best_scores > thres_wild_score]
        false_wild = embed_generated_wild[:, best_layer_over_thres[argmax_index], :][best_scores <= thres_wild_score]

        embed_train = np.concatenate([true_wild, false_wild], 0)
        label_train = np.concatenate([np.ones(len(true_wild)),
                                      np.zeros(len(false_wild))], 0)



        best_acc, final_acc, (
            clf, best_state, best_preds, preds, labels_val), losses_train = get_linear_acc(
            embed_train,
            label_train,
            embed_train,
            label_train,
            2, epochs=50,
            print_ret=True,
            batch_size=512,
            cosine=True,
            nonlinear=True,
            learning_rate=0.05,
            weight_decay=0.0003)

        clf.eval()
        output = clf(torch.from_numpy(
            embed_generated_test[:, best_layer_over_thres[argmax_index], :]).cuda())
        pca_wild_score_binary_cls = torch.sigmoid(output)
        pca_wild_score_binary_cls = pca_wild_score_binary_cls.cpu().data.numpy()
        if np.isnan(pca_wild_score_binary_cls).sum() > 0:
            breakpoint()
        measures = get_measures(pca_wild_score_binary_cls[gt_label_test == 1],
                                pca_wild_score_binary_cls[gt_label_test == 0], plot=False)
        print('test AUROC: ', measures[0])



if __name__ == '__main__':
    main()
