
import logging
import math
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import random
import numpy as np
import copy

from rouge_chinese import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import torch.nn as nn

from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional, List, Union, Tuple, Any

import datasets
import evaluate
import torch
from datasets import load_dataset, load_metric
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.tuners.lora import Linear4bit, LoraLayer

from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils.generic import PaddingStrategy
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast


from VLora.vlora import VLoraLayer
from VLora.WAE import gVar, gData
import bitsandbytes as bnb

EPSILON = 1e-4

def kl_divergence(data1, data2):
    mu1, sigma1 = torch.tensor(data1[0]).detach(), torch.tensor(data1[1]).detach()
    mu2, sigma2 = torch.tensor(data2[0]).detach(), torch.tensor(data2[1]).detach()
    # sigma1 = torch.exp(logsigma1)
    # sigma2 = torch.exp(logsigma2)

    kl_div = 0.5 * (torch.log(sigma2 ** 2 + EPSILON) - torch.log(sigma1 ** 2 + EPSILON) +
                    (sigma1 ** 2 + (mu1 - mu2) ** 2) / (EPSILON + sigma2) ** 2 - 1)
    return abs(kl_div)

beta = 1e-2


class CustomLlama(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.prior_net = nn.Sequential(
        #     self.model.embed_tokens,
        #     self.model.layers[0]
        # )
        #
        # self.prior_net[1].self_attn.o_proj = VLoraLayer(in_features=5120, out_features=5120, adapter_name='default')
        self.prior_net = None
        self.update_vae()


    def init_prior_net(self):
        self.prior_net = self.model
        # self.prior_net = copy.deepcopy(self.model)
        # why deepcopy net cannot update VAE_Z in VLoraLayer?????? direct is wrong.
        self.prior_net.layers[0].self_attn.o_proj = VLoraLayer(in_features=5120, out_features=5120,
                                                               adapter_name='default')

    def update_vae(self):
        self.model.layers[0].self_attn.o_proj = VLoraLayer(in_features=5120, out_features=5120, adapter_name='default')
        self.prior_net = copy.deepcopy(self.model)


    def forward(
        self,
        ###prior
        prior_ids: torch.LongTensor = None,
        ###posterior
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        # return super().forward()
        print('new forward')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        # torch.cuda.synchronize()

        # print('posterior net')
        # for name, param in self.model.layers[0].self_attn.v_proj.named_parameters():
        #     if param.requires_grad:
        #
        #     # param.annotation_data = param.annotation_data.to(torch.float32)
        #     # param.requires_grad = True
        #         print(name, param.data, param.data.dtype, param.grad)
        #     # if param.requires_grad:
        #     #     print(name, param.grad)
        #
        #
        # print('prior net all layers')
        # for name, param in self.prior_net.layers[0].self_attn.o_proj.named_parameters():
        #     print(name, param.data, param.data.dtype, param.grad)


        # param.requires_grad = True


        # exit()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # print('logits:', logits.size())

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            ### VAE_LOSS+
            posterior_z = self.model.layers[0].self_attn.o_proj.VAE_Z


            print('posterior_z:', posterior_z)


            prior_outputs = self.prior_net(
                input_ids=prior_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            prior_hidden_states = prior_outputs[0]
            prior_logits = self.lm_head(prior_hidden_states)
            print(prior_logits)
            print('prior net all layers')
            for name, param in self.prior_net.layers[0].self_attn.o_proj.named_parameters():
                print(name, param.data, param.data.dtype, param.grad)
            prior_z = self.prior_net.layers[0].self_attn.o_proj.VAE_Z

            print('prior_z:', prior_z)

            custom_loss = torch.sum(kl_divergence(posterior_z, prior_z))




            #
            #
            print('custom_loss:', custom_loss)
            print(custom_loss.size())

            print('loss: ', loss)
            print(loss.size())

            loss += beta * custom_loss



            ### new optimizer
            posterior_layer = self.model.layers[0].self_attn.o_proj
            prior_layer = self.prior_net.layers[0].self_attn.o_proj

            ### require true for vae layers
            print('pos layer type')
            for name, param in posterior_layer.named_parameters():
                print(param.dtype)
                param.data = param.data.to(torch.float32)
                param.requires_grad = True
            print('prior layer type')
            for name, param in prior_layer.named_parameters():
                print(param.dtype)
                param.data = param.data.to(torch.float32)
                param.requires_grad = True



            new_optimizer = torch.optim.Adam(list(posterior_layer) + list(prior_layer), lr=1e-4)
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()

            ### no grad for vae layers
            ### require true for vae layers
            print('pos layer type')
            for name, param in posterior_layer.named_parameters():
                print(param.dtype)
                param.data = param.data.to(torch.float32)
                param.requires_grad = False
            print('prior layer type')
            for name, param in prior_layer.named_parameters():
                print(param.dtype)
                param.data = param.data.to(torch.float32)
                param.requires_grad = False

            ### new optimizer-----------------------------------------------------------

            print('loss: ', loss)

        if not return_dict:
            output = (logits,) + outputs[1:]

            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        torch.cuda.synchronize()
        if self.label_smoother is not None and "labels" in inputs:
            print('labels is not None.....')
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'], labels=inputs['labels'], prior_ids=inputs['prior_ids'])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss





    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer.
    #     """
    #     # Extract labels
    #     pri_input = inputs['input']
    #     pos_input = inputs['second']
    #     labels = inputs['target']
    #     inputs = pos_input
    #     # labels = inputs.pop("labels")
    #
    #     # Original model outputs
    #     outputs = model(**inputs)
    #
    #     # Original loss
    #     logits = outputs[0]
    #     loss_fct = nn.CrossEntropyLoss()
    #     original_loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    #
    #     # Your custom loss computation
    #     pos_z = model.VAE_Z
    #
    #     output_ = model(**pri_input)
    #     pri_z = model.VAE_Z
    #
    #     custom_loss = kl_divergence(pos_z, pri_z)
    #
    #     # Combine original loss with custom loss
    #     total_loss = original_loss + beta * custom_loss
    #
    #     return (total_loss, outputs) if return_outputs else total_loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.annotation_data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # print('train_dataset[0]:')
        # print(train_dataset.__getitem__(0))
        # print(len(train_dataset.__getitem__(0)))

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:

                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            # worker_init_fn=seed_worker,
        )



@dataclass
class CustomDataCollator(transformers.DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    #
    def __call__(self, features, return_tensors=None):

        print(len(features[0]))


        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                remainder_p = [32000] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "labels" else remainder + feature["labels"]
                    )
                    feature["prior_ids"] = (
                        feature["prior_ids"] + remainder_p if padding_side == "prior_ids" else remainder_p + feature["prior_ids"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    feature["prior_ids"] = (
                        feature["prior_ids"] + remainder_p if padding_side == "prior_ids" else remainder_p + feature[
                            "prior_ids"]
                    )
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    feature["prior_ids"] = (
                        feature["prior_ids"] + remainder if padding_side == "prior_ids" else remainder + feature[
                            "prior_ids"]
                    )

        prior_features = [feature.pop('prior_ids', None) for feature in features]
        print(len(prior_features[0]))
        print(len(features[0]))
        print(features[0].keys())
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # print(features)

        features.data['prior_ids'] = torch.tensor(prior_features)


        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

    # def __call__(self, features, return_tensors=None):
    #     print('pad before')
    #     print(len(features[1]['input_ids']))
    #     if return_tensors is None:
    #         return_tensors = self.return_tensors
    #     labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
    #     print('labels:', labels==None)
    #     # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    #     # same length to return tensors.
    #     if labels is not None:
    #         print('padding....')
    #         max_label_length = max(len(l) for l in labels)
    #         if self.pad_to_multiple_of is not None:
    #             max_label_length = (
    #                 (max_label_length + self.pad_to_multiple_of - 1)
    #                 // self.pad_to_multiple_of
    #                 * self.pad_to_multiple_of
    #             )
    #
    #         padding_side = self.tokenizer.padding_side
    #         print(padding_side)
    #         for feature in features:
    #             remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
    #             remainder_att = [-1] * (max_label_length - len(feature["labels"]))
    #             if isinstance(feature["labels"], list):
    #                 for k, v in feature.items():
    #                     if k == 'attention_mask':
    #                         feature[k] = (
    #                             v + remainder_att if padding_side == "right" else remainder_att + v
    #                         )
    #                     else:
    #                         feature[k] = (
    #                             v + remainder if padding_side == "right" else remainder + v
    #                         )
    #                 print('labels: list')
    #             elif padding_side == "right":
    #                 feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
    #             else:
    #                 feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
    #
    #     print('after padding...')
    #     # print(len(features[0]['labels']))
    #     # print(len(features[1]['labels']))
    #     print(len(features[0]['input_ids']))
    #     print(len(features[1]['input_ids']))
    #
    #     features = self.tokenizer.pad(
    #         features,
    #         padding=self.padding,
    #         max_length=self.max_length,
    #         pad_to_multiple_of=self.pad_to_multiple_of,
    #         return_tensors=return_tensors,
    #     )
    #
    #     print('after tokenizer.pad....')
    #
    #
    #     # prepare decoder_input_ids
    #     if (
    #         labels is not None
    #         and self.model is not None
    #         and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
    #     ):
    #         decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
    #         features["decoder_input_ids"] = decoder_input_ids
    #
    #
    #
    #     return features
