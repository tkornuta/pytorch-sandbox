# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Encoder-Decoder architectures"""

import warnings
import copy
from typing import Optional
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.utils import logging
from transformers import AutoConfig
from transformers import AutoModel, AutoModelForCausalLM
from transformers import EncoderDecoderConfig

class MultimodalEncoderDecoderConfig(PretrainedConfig):

    model_type = "multimodal-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "image_encoder" in kwargs and "command_encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with image encoder, command encoder and decoder configs"
        image_encoder_config = kwargs.pop("image_encoder")
        image_encoder_model_type = image_encoder_config.pop("model_type")

        command_encoder_config = kwargs.pop("command_encoder")
        command_encoder_model_type = command_encoder_config.pop("model_type")

        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.image_encoder = AutoConfig.for_model(image_encoder_model_type, **image_encoder_config)
        self.command_encoder = AutoConfig.for_model(command_encoder_model_type, **command_encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, image_encoder_config: PretrainedConfig, command_encoder_config: PretrainedConfig,decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`MultimodalEncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.
        Returns:
            [`MultimodalEncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(image_encoder=image_encoder_config.to_dict(),
            command_encoder=command_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["image_encoder"] = self.image_encoder.to_dict()
        output["command_encoder"] = self.command_encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output



logger = logging.get_logger(__name__)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MultimodalEncoderDecoderModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = MultimodalEncoderDecoderConfig
    base_model_prefix = "multimodal_encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        image_encoder: Optional[PreTrainedModel] = None,
        command_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (image_encoder is None or command_encoder is None or decoder is None):
            raise ValueError("Either a configuration or both encoders and a decoder have to be provided.")
        if config is None:
            config = MultimodalEncoderDecoderConfig.from_encoder_decoder_configs(image_encoder.config, command_encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # TODO: probably this is wrong! both encoders should have the same hidden size!
        #import pdb;pdb.set_trace()
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != (config.image_encoder.hidden_size + config.command_encoder.hidden_size):
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, "
                    "it has to be equal to the sum of encoder's `hidden_size`s. "
                    f"Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` "
                    f"and {config.image_encoder.hidden_size} for `config.image_encoder.hidden_size`"
                    f"and {config.command_encoder.hidden_size} for `config.command_encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if image_encoder is None:
            from transformers import AutoModel

            image_encoder = AutoModel.from_config(config.image_encoder)

        if command_encoder is None:
            from transformers import AutoModel

            command_encoder = AutoModel.from_config(config.command_encoder)

        if decoder is None:
            from transformers import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.image_encoder = image_encoder
        self.command_encoder = command_encoder
        self.decoder = decoder

        #if self.encoder.config.to_dict() != self.config.encoder.to_dict():
        #    logger.warning(
        #        f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
        #    )
        #if self.decoder.config.to_dict() != self.config.decoder.to_dict():
        #    logger.warning(
        #        f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}"
        #    )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.image_encoder.config = self.config.image_encoder
        self.command_encoder.config = self.config.command_encoder
        self.decoder.config = self.config.decoder

        # encoders outputs might need to be projected to different dimension for decoder cross-attention.
        # TODO: probably this is wrong! both encoders should have the same hidden size!
        #if (
        #    self.decoder.config.hidden_size != (self.image_encoder.config.hidden_size + self.command_encoder.config.hidden_size)
        #    and self.decoder.config.cross_attention_hidden_size is None
        #):
        #    self.enc_to_dec_proj = nn.Linear(self.image_encoder.config.hidden_size + self.command_encoder.config.hidden_size, self.decoder.config.hidden_size)

        #if self.encoder.get_output_embeddings() is not None:
        #    raise ValueError(
        #        f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
        #    )

        # tie encoder, decoder weights if config set accordingly
        #self.tie_weights()

    #def tie_weights(self):
    #    # tie encoder & decoder if needed
    #    if self.config.tie_encoder_decoder:
    #        # tie encoder and decoder base model
    #        decoder_base_model_prefix = self.decoder.base_model_prefix
    #        self._tie_encoder_decoder_weights(
    #            self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
    #        )

    def get_image_encoder(self):
        return self.image_encoder

    def get_command_encoder(self):
        return self.command_encoder

    def get_decoder(self):
        return self.decoder

    def get_input_image_embeddings(self):
        # TODO: check: does it makes even sense?
        return self.image_encoder.get_input_embeddings()

    def get_input_command_embeddings(self):
        return self.command_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_image=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        print("model forward !!!")
        import pdb;pdb.set_trace()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            # Get image embeddings.
            image_encoder_outputs = self.image_encoder(pixel_values=input_image)

            # Get command embeddings.
            command_encoder_outputs = self.command_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

            # Concatenate outputs of both encoders along "sequence" dimension.
            encoder_outputs = torch.cat((image_encoder_outputs[0], command_encoder_outputs[0]), 1)
        
        # Back to regular track ;)
        encoder_hidden_states = encoder_outputs

        # optionally project encoder_hidden_states
        # TODO: probably this is wrong! both encoders should have the same hidden size!
        #if (
        #    self.decoder.config.hidden_size != (self.image_encoder.config.hidden_size + self.command_encoder.config.hidden_size)
        #    and self.decoder.config.cross_attention_hidden_size is None
        #):
        #    encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        #import pdb;pdb.set_trace()

        # Decode
        print("decoder decode !!!")
        import pdb;pdb.set_trace()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            #warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        import pdb;pdb.set_trace()
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values, # tuple
            decoder_hidden_states=decoder_outputs.hidden_states, # None!
            decoder_attentions=decoder_outputs.attentions, # None!
            cross_attentions=decoder_outputs.cross_attentions, # None!
            # TODO: Not sure about the following ones! what should be there?
            encoder_last_hidden_state=command_encoder_outputs.last_hidden_state,
            encoder_hidden_states=command_encoder_outputs.hidden_states,
            encoder_attentions=command_encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        import pdb;pdb.set_trace()
        print("prepare_inputs_for_generation !!!")
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. "
            "Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        import pdb;pdb.set_trace()
        print("_prepare_decoder_input_ids_for_generation !!!")

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * decoder_start_token_id


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        import pdb;pdb.set_trace()
        print("_prepare_encoder_decoder_kwargs_for_generation !!!")
        # 1. get encoder
        #encoder = self.get_encoder() - got two encoders.
        #input_image = model_kwargs.pop("input_image")
        input_image = model_kwargs["input_image"]

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "input_image"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # Prepare "encoders output".

        # Get image embeddings.
        image_encoder_outputs = self.image_encoder(pixel_values=input_image)
        

        # Get command embeddings.
        command_encoder_outputs = self.command_encoder(
            **encoder_kwargs
            )

        # Concatenate outputs of both encoders along "sequence" dimension.
        encoder_hidden_states = torch.cat((image_encoder_outputs[0], command_encoder_outputs[0]), 1)

        # To model output?
        # ModelOutput = ... ?

        model_kwargs["encoder_outputs"]: ModelOutput = encoder_hidden_states

        return model_kwargs