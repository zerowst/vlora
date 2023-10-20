from transformers import AutoTokenizer, AutoModelForCausalLM

import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft import (  # noqa: E402
    LoraModel,
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from peft.tuners.lora import Linear4bit, LoraLayer
import bitsandbytes as bnb

from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig
)
import torch
import time
import evaluate

from VLora.WAE import gData, gVar

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


class Variation(nn.Module):
    def __init__(self, input_size, output_size, z_size=200):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.output_size = output_size
        self.fc = nn.Sequential(
            bnb.nn.Linear4bit(input_size, z_size),
            nn.LayerNorm(z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            bnb.nn.Linear4bit(z_size, z_size),
            nn.LayerNorm(z_size),
            # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.Tanh(),
        )
        self.context_to_mu = bnb.nn.Linear4bit(z_size, z_size)  # activation???
        self.context_to_logsigma = bnb.nn.Linear4bit(z_size, z_size)
        self.z_to_w = bnb.nn.Linear4bit(self.output_size + self.z_size, self.output_size)



        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, z_size),
        #     nn.LayerNorm(z_size),
        #     # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        #     nn.Linear(z_size, z_size),
        #     nn.LayerNorm(z_size),
        #     # nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
        #     nn.Tanh(),
        # )
        # self.context_to_mu = nn.Linear(z_size, z_size)  # activation???
        # self.context_to_logsigma = nn.Linear(z_size, z_size)
        #
        # self.z_to_w = nn.Linear(self.output_size + self.z_size, self.output_size)

        # torch.nn.init.zeros_(self.z_to_w.weight)
        # self.fc.apply(self.init_weights)
        # self.init_weights(self.context_to_mu)
        # self.init_weights(self.context_to_logsigma)
        # self.init_weights(self.z_to_w)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        print('input x type:', x.dtype)
        batch_size, sequence_size, embed = x.size()
        lora_x = x.clone()
        x = self.fc(x)
        mu = self.context_to_mu(x)
        # logsigma**2
        logsigma = self.context_to_logsigma(x)
        std = torch.exp(0.5 * logsigma)

        epsilon = torch.randn([batch_size,sequence_size, self.z_size]).cuda()
        # print('epsilon type:', epsilon.dtype)
        # size: batch_size * sequence_len * embed_size


        z = epsilon * std + mu
        z = torch.cat([z, lora_x], dim=2)
        z = self.z_to_w(F.tanh(z))
        # print('mu, std:', mu, std)
        return z, mu, std



class VLoraLayer(Linear4bit):
    def __init__(self, in_features, out_features, adapter_name: str = "default",
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 **kwargs,
                 ):

        Linear4bit.__init__(self, in_features=in_features, out_features=out_features, adapter_name=adapter_name)

        # super().__init__(in_features, out_features, *args,**kwargs)

        # self.beta = beta

        self.active_adapter = adapter_name

        self.VAE_model = Variation(input_size=out_features, output_size=out_features)
        self.VAE_Z = None

    def forward(self, x: torch.Tensor):
        print('vlora x type', x.dtype)
        result = super().forward(x)

        result = result.clone()
        print('result', result.dtype)


        result = result.clone()
        if not torch.is_autocast_enabled():
            expected_dtype = result.dtype
            print('result type: ', expected_dtype)
            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            print('x type:', self.lora_A[self.active_adapter].weight.dtype)
            output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    ).to(expected_dtype)
                    * self.scaling[self.active_adapter]
            )
            # print(2)
        else:
            output = (
                    self.lora_B[self.active_adapter](
                        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
            )
            # print(3)

        # print('output:', output)
        # if output.dim() == 1: output.unsqueeze(0)
        # print(output.size())
        v_result, mu, std = self.VAE_model(output)
        print('v_result type:', v_result.dtype)


        self.VAE_Z = [mu, std]


        result = result + v_result
        return result


    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer.
            """
            # Extract labels
            pos_input, pri_input = inputs
            inputs = pos_input
            labels = inputs.pop("labels")

            # Original model outputs
            outputs = model(**inputs)

            # Original loss
            logits = outputs[0]
            loss_fct = CrossEntropyLoss()
            original_loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

            # Your custom loss computation
            pos_z = model.VAE_Z

            output_ = model(pri_input)
            pri_z = model.VAE_Z

            custom_loss = kl_divergence(pos_z, pri_z)

            # Combine original loss with custom loss
            total_loss = original_loss + beta * custom_loss

            return (total_loss, outputs) if return_outputs else total_loss



if __name__ == '__main__':

    def kl_divergence(data1, data2):
        mu1, sigma1 = data1
        mu2, sigma2 = data2
        # sigma1 = torch.exp(logsigma1)
        # sigma2 = torch.exp(logsigma2)

        kl_div = 0.5 * (logsigma2 - logsigma1 + (sigma1 ** 2 + (mu1 - mu2) ** 2) / sigma2 ** 2 - 1)

        return kl_div


    # Example
    mu1 = torch.tensor([0.5])
    logsigma1 = torch.tensor([0.1])

    mu2 = torch.tensor([0.4])
    logsigma2 = torch.tensor([0.2])

    loss = kl_divergence([mu1, logsigma1], [mu2, logsigma2])
    # print(loss)

    beta = 1e-4
    x = torch.rand([10, 100]).cuda()
    # print(x)
    model = VLoraLayer(in_features=100, out_features=100).to('cuda')
    print(model(x))


    # def custom_loss(output, label, prior):
    #     loss_ori








