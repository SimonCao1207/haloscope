# HaloScope


This is the source code accompanying the NeurIPS'24 spotlight [***HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection***](https://arxiv.org/abs/2409.17504) by Xuefeng Du, Chaowei Xiao, and Yixuan Li

## Additional notes

I recently got some emails on the inconsistent results on TydiQA dataset. Here is my notice to you all!

Thank you for the email! I did not know for sure what was happening. I have cleaned the code quite extensively from what I ran before the submission so there might be some pieces that are not proper in the current code. Since the code can produce very different results across different dataset splits and machines, so I suggest you tuning the following things in the code (from most important to least ones):

- Check the generated answers to see whether they contain repetitions or unnecessary stuff and filter them. 
- In line 572 (https://github.com/deeplearning-wisc/haloscope/blob/4045fc23c568c2c7598124c6dca1dd021c7a447e/hal_det_llama.py#L572), you can also choose what is the maximal principal components you want to use to test on validation set, it is quite sensitive tbh.
- The threshold and the metric for getting the gt labels (rather than 0.5).
- The location to extract the embeddings.
- Whether to use weighted scores.
- Finally, the number of validation samples in the code.

Thank you for the patience! Please also feel free to report the average results across multiple runs in your new experiments! We appreciate your additional efforts and interest in our work.

Authors


## Ads 

Checkout our ICML'23 work [SCONE](https://proceedings.mlr.press/v202/bai23a/bai23a.pdf), ICLR'24 work [SAL](https://openreview.net/forum?id=jlEjB8MVGa) and a recent [preprint](https://arxiv.org/abs/2410.00296v1) on leveraging unlabeled data for OOD detection and VLM harmful prompt detection and if you are interested!



## Requirements
```
conda install -f env.yml
```

## Models Preparation

Please download the LLaMA-2 7b / 13b  from [here](https://huggingface.co/meta-llama) and OPT [6.7b]((https://huggingface.co/facebook/opt-6.7b)) / [13b]((https://huggingface.co/facebook/opt-13b)) models. Setup a local directory for saving the models:
```angular2html
mkdir models
```
And put the model checkpoints inside the folder.
## Get LLM generations

Firstly, make a local directory for saving the LLM-generated answers, model-generated truthfulness ground truth, and features, etc.
```angular2html
mkdir save_for_eval
```

For TruthfulQA, please run:

```angular2html
CUDA_VISIBLE_DEVICES=0 python hal_det_llama.py --dataset_name tqa --model_name llama2_chat_7B --most_likely 1 --num_gene 1 --gene 1
```
- "most_likely" means whether you want to generate the most likely answers for testing (most_likely == 1) or generate multiple answers with sampling techniques for uncertainty estimation.
- "num_gene" is how many samples we generate for each question, for most_likely==1, num_gene should be 1 otherwise we set num_gene to 10.
- "dataset_name" can be chosen from tqa, coqa, triviaqa, tydiqa
- "model_name" can be chosen from llama2_chat_7B, and llama2_chat_13B

Please check section 4.1 implementation details in the paper for reference.

For OPT models, please run:
```angular2html
CUDA_VISIBLE_DEVICES=0 python hal_det_opt.py --dataset_name tqa --model_name opt-6.7b --most_likely 1 --num_gene 1 --gene 1
```

## Get the ground truth for the LLM generations
Since there is no ground truth for the generated answers, we leverage rouge and [BleuRT](https://arxiv.org/abs/2004.04696) for getting a sense of whether the answer is true or false.

To download the Bleurt models, please refer to [here](https://github.com/lucadiliello/bleurt-pytorch) and put the model to the ./models folder:

For TruthfulQA, please run:

```angular2html
CUDA_VISIBLE_DEVICES=0 python hal_det_llama.py --dataset_name tqa --model_name llama2_chat_7B --most_likely 1 --use_rouge 0 --generate_gt 1
```

- when "use_rouge" is 1, then we use rouge for determining the ground truth, otherwise we use BleuRT.

For OPT models, please run:
```angular2html
CUDA_VISIBLE_DEVICES=0 python hal_det_opt.py --dataset_name tqa --model_name opt-6.7b --most_likely 1 --use_rouge 0 --generate_gt 1
```

## Hallucination detection

For TruthfulQA, please run:
```angular2html
CUDA_VISIBLE_DEVICES=0 python hal_det_llama.py --dataset_name tqa --model_name llama2_chat_7B --use_rouge 0 --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
```
- "weighted_svd" denotes whether we need the weighting coeffcient by the singular values in the score.
- "feat_loc_svd" denotes which location in a transformer block do we extract the representations, 3 is block output, 2 is mlp output and 1 is attention head output.


For OPT models, please run:
```angular2html
CUDA_VISIBLE_DEVICES=0 python hal_det_opt.py --dataset_name tqa --model_name opt-6.7b --use_rouge 0 --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
```

## Citation ##
If you found any part of this code is useful in your research, please consider citing our paper:

```
 @inproceedings{du2024haloscope,
      title={ HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection}, 
      author={Xuefeng Du and Chaowei Xiao and Yixuan Li},
      booktitle={Advances in Neural Information Processing Systems},
      year = {2024}
}
```
