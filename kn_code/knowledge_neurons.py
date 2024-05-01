# main knowledge neurons class
import collections
import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .patch import *
from .blimp import BLiMPDataProcessing
from kn_code.rome import apply_rome_to_model
from kn_code.rome_hparams import ROMEHyperParams

class KnowledgeNeuronsBase:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif 'gptj' == model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.fc_in"
            self.output_ff_attr = "mlp.fc_out.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte.weight"
        elif model_type in ['gpt', 'gpt2']:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            # self.word_embeddings_attr = "transformer.wpe"
            self.word_embeddings_attr = "transformer.wte"
        elif 'llama' == model_type:
            self.transformer_layers_attr = "model.layers"
            self.input_ff_attr = "mlp.gate_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.word_embeddings_attr = "model.embed_tokens.weight"
        elif "t5" == model_type:
            self.transformer_layers_attr = "decoder.block"
            self.input_ff_attr = "layer.2.DenseReluDense.wi"
            self.output_ff_attr = "layer.2.DenseReluDense.wo.weight"
            self.word_embeddings_attr = "shared.weight"
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.model_type == 't5':
            target_input = self.tokenizer(target, return_tensors='pt').to(self.device)
            encoded_input['decoder_input_ids'] = target_input['input_ids']
            encoded_input['decoder_attention_mask'] = target_input['attention_mask']
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        elif self.model_type == 't5':
            mask_idx = list(range(encoded_input['decoder_input_ids'].size(1)))
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = -1
        if target is not None:
            if "gpt" in self.model_type or 't5' in self.model_type or 'llama' in self.model_type:
                target = self.tokenizer.encode(target)
            else:
                target = self.tokenizer.convert_tokens_to_ids(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label) if ("gpt" in self.model_type or 'llama' in self.model_type) else 1
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i] if n_sampling_steps > 1 else target_label
            # print(probs.shape)
            # gt_prob = probs[:, target_idx].item()
            # print(target_idx)
            if self.model_type == 't5':
                for q, target_idx_ in enumerate(target_idx):
                    gt_prob_= probs[:, q, target_idx_]
                    all_gt_probs.append(gt_prob_)

                    argmax_prob, argmax_id = [i.item() for i in probs[:,q,:].max(dim=-1)]
                    argmax_tokens.append(argmax_id)
                    argmax_str = self.tokenizer.decode([argmax_id])
                    all_argmax_probs.append(argmax_prob)

                    argmax_completion_str += argmax_str
            else:
                gt_prob = probs[:, target_idx]
                # print(gt_prob.shape)
                all_gt_probs.append(gt_prob)

                # get info about argmax completion
                argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
                argmax_tokens.append(argmax_id)
                argmax_str = self.tokenizer.decode([argmax_id])
                all_argmax_probs.append(argmax_prob)

                prompt += argmax_str
                argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def generate(self, prompt, n_sampling_steps=1):
        encoded_input, mask_idx, target_label = self._prepare_inputs(prompt, 'good')
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, _ = self._prepare_inputs(
                    prompt, 'good'
                )
            outputs = self.model(**encoded_input)
            probs = torch.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            return probs

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        if activations.dim() == 2:
            tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
            )
        elif activations.dim() == 3:
            tiled_activations = einops.repeat(activations, "b m d -> (r b) m d", r=steps)
            return (
                tiled_activations
                * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None, None]
            )
        else:
            raise Exception(f"Bad!! The dim of Activation is {activations.dim()}")

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
            )
            scores.append(layer_scores)
        scores = [score.to(self.device) for score in scores]
        return torch.stack(scores)

    def get_coarse_neurons(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
        
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), "Provide one and only one of threshold / adaptive_threshold / percentile"

        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            coarse_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
            if self.model_type == 't5' and len(coarse_neurons) > 0 and len(coarse_neurons[0]) == 3:
                coarse_neurons = list(set([(layer_idx, neuron_idx) for layer_idx, _, neuron_idx in coarse_neurons]))
            return coarse_neurons
        s = attribution_scores.flatten().detach().cpu().numpy()
        return (
            torch.nonzero(attribution_scores > np.percentile(s, percentile))
            .cpu()
            .tolist()
        )

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
        refine: bool = False,
    ) -> List[List[int]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = [
            self.get_coarse_neurons(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                adaptive_threshold=coarse_adaptive_threshold,
                threshold=coarse_threshold,
                percentile=coarse_percentile,
                pbar=False,
            )
            for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
            )
        ]
        if negative_examples is not None:
            negative_neurons = [
                self.get_coarse_neurons(
                    negative_example,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
                for negative_example in tqdm(
                    negative_examples,
                    desc="Getting coarse neurons for negative examples",
                    disable=quiet,
                )
            ]
        if not quiet:
            total_coarse_neurons = sum(len(i) for i in coarse_neurons)
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        if refine:
            refined_neurons = [list(neuron) for neuron, count in c.items() if count > t]
        else:
            refined_neurons = [list(neuron) for neuron, count in c.items()]
        # filter out neurons that are in the negative examples
        if negative_examples is not None and False:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        if not quiet:
            total_refined_neurons = len(refined_neurons)
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label) if ("gpt" in self.model_type or 'llama' in self.model_type) else 1
        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                if i > 0 and (self.model_type == "gpt" or self.model_type == 'llama'):
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                    if self.model_type == 't5':
                        inputs["decoder_input_ids"] = einops.repeat(
                            encoded_input["decoder_input_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )
                        inputs["decoder_attention_mask"] = einops.repeat(
                            encoded_input["decoder_attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    target_idx = (
                        target_label[i] if n_sampling_steps > 1 else target_label
                    )
                    if self.model_type == 't5':
                        assert probs.size(1) == len(target_idx)
                        target_probs = [probs[:, q, target_idx_] for q, target_idx_ in enumerate(target_idx)]

                        grad = torch.autograd.grad(
                            torch.unbind(torch.cat(target_probs, dim=0)), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)
                    else:
                        grad = torch.autograd.grad(
                            torch.unbind(probs[:, target_idx]), batch_weights
                        )[0]
                        grad = grad.sum(dim=0)
                        integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

                if n_sampling_steps > 1:
                    prompt += next_token_str
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )
            return integrated_grads
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and (self.model_type == "gpt" or self.model_type == 'llama'):
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def erase(self, prompt, neurons):
        _, mask_idx, _ = self._prepare_inputs(prompt, 'good')

        patch_ff_layer(
            self.model,
            mask_idx,
            mode='suppress',
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
        modified_layers = list(set([x[0] for x in neurons]))
        return modified_layers

class KnowledgeNeurons(KnowledgeNeuronsBase, BLiMPDataProcessing):

    def search_kns(self, paradigm, search_target, n_prompts=100, batch_size=20):

        st_pairs = self.get_blimp_source_target_pairs(paradigm, search_target)
        att_scores = []

        for source, target in tqdm(st_pairs[:n_prompts]):
            att_score = self.get_scores(source, target, batch_size=batch_size, pbar=False)
            att_scores.append(att_score.detach())
        att_scores = torch.stack(att_scores)

        print('TARGET', target)
        print(fix_p_neuron(att_scores))
        print(run_kn_alg(att_scores))
        print(cosine_sim_n(att_scores.view(att_scores.shape[0], -1)))
        print('=== END ===')

    def localising_paradigm(self, paradigm, batch_size=20, n_prompts=-1):
        combinations = self.get_blimp_combinations(paradigm)
        results = {}

        prompts = self.get_blimp_prompts(paradigm)[0]
        target_func = self.TARGET_FUNC[paradigm]

        for target, item in combinations:
            att_scores = []
            for prompt in tqdm(prompts[:n_prompts]):
                _, _, ori_item = target_func(prompt, target, True)
                assert prompt.count(ori_item) == 1
                source = prompt.replace(ori_item, item)
                att_score = self.get_scores(
                    source, target,
                    batch_size=batch_size, pbar=False)
                att_scores.append(att_score)
            att_scores = torch.stack(att_scores)
            kns, _ = run_kn_alg(att_scores)
            results[(target, item)] = kns

    def complete(self, prompts, max_out_len=20, n_gen_per_prompt=1, top_k=5):
        if isinstance(prompts, str):
            prompts = [prompts]
        input_prompts = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
        input_tokens = self.tokenizer(input_prompts, padding=True, return_tensors="pt").to(self.device)

        input_ids, attention_mask = input_tokens["input_ids"], input_tokens["attention_mask"]
        batch_size = input_ids.size(0)
        prompt_size = input_ids.size(1)
        past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

        with torch.no_grad():
            while input_ids.size(1) < max_out_len + prompt_size:
                model_out = self.model(
                    input_ids=input_ids[:, cur_context],
                    attention_mask=attention_mask[:, cur_context],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits, past_key_values = model_out.logits, model_out.past_key_values
                softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)


                # Top-k sampling
                tk = torch.topk(softmax_out, top_k, dim=1).indices
                softmax_out_top_k = torch.gather(softmax_out, 1, tk)
                softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
                new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                new_toks = torch.gather(tk, 1, new_tok_indices)

                # If we're currently generating the continuation for the last token in `input_ids`,
                # create a new index so we can insert the new token
                if cur_context.stop == input_ids.size(1):
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                    )
                    input_ids = torch.cat(
                        [
                            input_ids,
                            input_ids.new_ones(batch_size, 1) * self.tokenizer.pad_token_id,
                        ],
                        dim=1,
                    )

                last_non_masked = attention_mask.sum(1) - 1
                for i in range(batch_size):
                    new_idx = last_non_masked[i] + 1
                    if last_non_masked[i].item() + 1 != cur_context.stop:
                        continue

                    # Stop generating if we've already maxed out for this prompt
                    if new_idx < max_out_len:
                        input_ids[i][new_idx] = new_toks[i]
                        attention_mask[i][new_idx] = 1

                cur_context = slice(cur_context.stop, cur_context.stop + 1)

        text = [self.tokenizer.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
        text = [
            x
            .replace("\n\n", " ")
            .replace("<|endoftext|>", "")
            for x in text
        ]

        return text

    def rome_edit(self, prompt, target_new, ground_truth, subject):

        hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl')

        request = [{
            'prompt': prompt,
            'target_new': target_new,
            'ground_truth': ground_truth,
            'subject': subject,
        }]

        model_edited, orig_weights = apply_rome_to_model(
            self.model, self.tokenizer,
            request, hparams, return_orig_weights=True, keep_original_weight=True)

        self.model = model_edited
        return orig_weights


def get_refined_neurons(coarse_neurons, p):
    t = len(coarse_neurons) * p
    c = collections.Counter()
    for neurons in coarse_neurons:
        for n in neurons:
            c[tuple(n)] += 1
    refined_neurons = [list(neuron) for neuron, count in c.items() if count > t]
    return refined_neurons

def fix_p_neuron(scores, threshold=0.2, p=0.7):
    coarse_neurons = []
    for score in scores:
        threshold = score.max().item() * threshold
        coarse_neurons.append(torch.nonzero(score > threshold).tolist())
    refined_neurons = get_refined_neurons(coarse_neurons, p)
    return refined_neurons

def run_kn_alg(scores, kn_num_min=2, kn_num_max=5, adaptive_threshold=0.2, p=0.7):
    coarse_neurons = []

    for score in scores:
        threshold = score.max().item() * adaptive_threshold
        coarse_neurons.append(torch.nonzero(score > threshold).tolist())

    for i in range(20):
        refined_neurons = get_refined_neurons(coarse_neurons, p)
        if len(refined_neurons) < kn_num_min:
            p -= 0.05
        elif len(refined_neurons) > kn_num_max:
            p += 0.05
        else:
            break
    return refined_neurons, p

@torch.no_grad()
def cosine_sim_n(scores):
    # https://stats.stackexchange.com/questions/239059/similarity-metrics-for-more-than-two-vectors
    y = scores / scores.norm(dim=1).unsqueeze(1)
    y = torch.nan_to_num(y, nan=1e-15)
    u, s, v = torch.svd(y)
    return (s[0] ** 2 - 1)/(scores.shape[0] - 1)
