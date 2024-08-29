import argparse
import functools
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict,PeftModel
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import io
import matplotlib.pyplot as plt
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config, ae_norm, ae_denorm, CausalVAEModelWrapper
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights
from transformers import AutoConfig
from huggingface_hub import create_repo
from packaging import version
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from transformers import T5Tokenizer, T5EncoderModel, MT5EncoderModel
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers import PixArtAlphaPipeline

MAX_SEQ_LENGTH = 77
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
logger = get_logger(__name__)



import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
from diffusers.image_processor import PixArtImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

def _clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    # urls:
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r"@[\w\d]+\b", "", caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
        "-",
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)

    # &quot;
    caption = re.sub(r"&quot;?", "", caption)
    # &amp
    caption = re.sub(r"&amp", "", caption)

    # ip adresses:
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

    # article ids:
    caption = re.sub(r"\d:\d\d\s+$", "", caption)

    # \n
    caption = re.sub(r"\\n", " ", caption)

    # "#123"
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    # "#12345.."
    caption = re.sub(r"#\d{5,}\b", "", caption)
    # "123456.."
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()

def _text_preprocessing(text, clean_caption=False):
    if clean_caption and not is_bs4_available():
        logger.warning(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
        logger.warning("Setting `clean_caption` to False...")
        clean_caption = False

    if clean_caption and not is_ftfy_available():
        logger.warning(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
        logger.warning("Setting `clean_caption` to False...")
        clean_caption = False

    if not isinstance(text, (tuple, list)):
        text = [text]

    def process(text: str):
        if clean_caption:
            text = _clean_caption(text)
            text = _clean_caption(text)
        else:
            text = text.lower().strip()
        return text

    return [process(t) for t in text]


def encode_prompt(
    text_encoder,
    prompt: Union[str, List[str]],
    do_classifier_free_guidance: bool = True,
    negative_prompt: str = "",
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    clean_caption: bool = True,
    max_sequence_length: int = 512,
    **kwargs,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
            instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
            PixArt-Alpha, this should be "".
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            whether to use classifier free guidance or not
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            number of images that should be generated per prompt
        device: (`torch.device`, *optional*):
            torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
            string.
        clean_caption (`bool`, defaults to `False`):
            If `True`, the function will preprocess and clean the provided caption before encoding.
        max_sequence_length (`int`, defaults to 120): Maximum sequence length to use for the prompt.
    """

    if "mask_feature" in kwargs:
        deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)


    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # See Section 3.1. of the paper.
    max_length = max_sequence_length

    if prompt_embeds is None:
        prompt = _text_preprocessing(prompt, clean_caption=clean_caption)
        # print("new_prompt", prompt)
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # print("text_inputs", text_inputs)
        text_input_ids = text_inputs.input_ids
        # print("text_input_ids", text_input_ids)
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        # print("untruncated_ids", untruncated_ids)
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because T5 can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)
        # print("text_encoder",text_encoder)
        # print("text_input_ids", text_input_ids.to(device).dtype, text_input_ids.to(device).shape,text_input_ids.to(device))
        # print("prompt_attention_mask", prompt_attention_mask.dtype, prompt_attention_mask.shape,prompt_attention_mask)
        text_encoder=text_encoder.to(device)
        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0]
        # print("prompt_embeds", prompt_embeds)

    if text_encoder is not None:
        dtype = text_encoder.dtype
    # elif transformer is not None:
    #     dtype = transformer.dtype
    else:
        dtype = None

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
        uncond_tokens = _text_preprocessing(uncond_tokens, clean_caption=clean_caption)
        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative_prompt_attention_mask = uncond_input.attention_mask
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask



class Text2ImageDataset:
    def __init__(
        self,
        jsonl_path: str,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        def transform(example):
            return {"text": example["caption"]}

        self._train_dataset = load_dataset(
            'json',
            data_files=jsonl_path,
            split='train',
            streaming=False,
        )
        self._train_dataset = self._train_dataset.map(transform, batched=False, remove_columns=self._train_dataset.column_names)
        self._train_dataset = self._train_dataset.filter(lambda example: len(example) > 0)

        self.num_train_examples = num_train_examples
        self.per_gpu_batch_size = per_gpu_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def create_dataloader(self):
        # 创建 RandomSampler 以实现随机打乱
        sampler = torch.utils.data.RandomSampler(self._train_dataset)
        
        # 创建DataLoader
        dataloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.per_gpu_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            sampler=sampler  # 使用 RandomSampler
        )

        num_batches = math.ceil(self.num_train_examples / self.global_batch_size)
        dataloader.num_batches = num_batches
        dataloader.num_samples = self.num_train_examples

        return dataloader



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Model Checkpoint Loading Arguments----------
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_student_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM student model or model identifier from huggingface.co/models.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    # parser.add_argument(
    #     "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    # )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--unet_time_cond_proj_dim", type=int, default=512, help="The time embedding projection dimension for the student U-Net.")
    parser.add_argument("--train_type", type=str, default="distillation", help="The type of training to perform.")

    parser.add_argument("--lora_rank", type=int, default=64, help="Rank for LoRA adaptation.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA adaptation.")



    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--num_train_inferences", type=int, default=8, help="Number of inferences to run during training.")

    parser.add_argument("--image_dir", type=str, default="/home/chenkai/data/image/ckimage", help="The directory to save the generated images.")
    parser.add_argument("--text_encoder_name", type=str, default="google/mt5-xxl", help="The name of the text encoder model.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def time2tensor(t, zt2):
    time=t
    if not torch.is_tensor(t):
        dtype = zt2.dtype
        time = torch.tensor([time], dtype=dtype, device=zt2.device)
    elif len(t.shape) == 0:
        time = time[None].to(zt2.device)
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    time = time.expand(zt2.shape[0])
    return time


def ddim_solver(teacher_model,zt,t,s,alpha_schedule,sigma_schedule,latents,prompt_embeds,prompt_attention_mask,uncond_prompt_embeds,uncond_prompt_attention_mask,added_cond_kwargs,weight_dtype,latent_channels,guidance_scale):
    zt=zt.to(alpha_schedule.device)
    t=t.to(alpha_schedule.device)
    s=s.to(alpha_schedule.device)
    with torch.no_grad():
        with torch.autocast("cuda"):
            teacher_model.to(weight_dtype)
            prompt_embeds=prompt_embeds.to(weight_dtype)
            prompt_attention_mask=prompt_attention_mask.to(weight_dtype)
            # print("zt",zt.shape)
            # zt2 = torch.cat([zt] * 2)
            zt2=zt.to(weight_dtype)
            # zt2=scheduler.scale_model_input(zt2, t)
            time=time2tensor(t,zt2)
            # print("zt",zt.shape)
            # print("input",zt2.dtype,zt2.shape,zt2)
            # print("prompt_embeds",prompt_embeds.dtype,prompt_embeds.shape,prompt_embeds)
            # print("prompt_attention_mask",prompt_attention_mask.dtype,prompt_attention_mask.shape,prompt_attention_mask)
            # print("current_timestep",time.dtype,time.shape,time)
            # print("added_cond_kwargs",added_cond_kwargs)
            teacher_output = teacher_model(
                                    zt2.to(teacher_model.device),
                                    encoder_hidden_states=prompt_embeds.to(teacher_model.device),
                                    encoder_attention_mask=prompt_attention_mask.to(teacher_model.device),
                                    timestep=time.to(teacher_model.device),
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]
            # print("teacher_output",teacher_output)

            # noise_pred_uncond, noise_pred_text = teacher_output.chunk(2)
            uncond_teacher_output=teacher_model(
                                    zt2.to(teacher_model.device),
                                    encoder_hidden_states=uncond_prompt_embeds.to(teacher_model.device),
                                    encoder_attention_mask=uncond_prompt_attention_mask.to(teacher_model.device),
                                    timestep=time.to(teacher_model.device),
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )[0]
            noise_pred_text=teacher_output
            noise_pred_uncond=uncond_teacher_output
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # print("noise_pred",noise_pred.shape)
            # print("teacher_model.config.out_channels",teacher_model.config.out_channels)
            # print("latent_channels",latent_channels)
            if teacher_model.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # print("t",t)
            # print("noise_pred",noise_pred.dtype,noise_pred.shape,noise_pred)
            # print("zt",zt.dtype,zt.shape,zt)
            # print("extra_step_kwargs",extra_step_kwargs)
            # result = scheduler.step(noise_pred, t, zt, **extra_step_kwargs, return_dict=False)[0]
            # print("result",result.dtype,result.shape,result)

            # print(alpha_schedule.device,sigma_schedule.device,t.device,s.device)
            # print("result",result)
            # print("noise_pred",noise_pred)
            alpha_t=alpha_schedule[t].view(-1,1,1,1,1)
            sigma_t=sigma_schedule[t].view(-1,1,1,1,1)
            alpha_s=alpha_schedule[s].view(-1,1,1,1,1)
            sigma_s=sigma_schedule[s].view(-1,1,1,1,1)
            # print("alpha_t",alpha_t,"sigma_t",sigma_t)
            # print("zt",zt)
            pred_x=(zt-sigma_t*noise_pred)/alpha_t
            # print("pred_x",pred_x)
            # print("pred_x",pred_x.shape,"alpha_s",alpha_s.shape)
            result=alpha_s*pred_x+sigma_s*noise_pred
            # print("ddim",result.dtype)
            # print("result",result)
    return result


def denoise(model,zt,t,s,alpha_schedule,sigma_schedule,latents,prompt_embeds,prompt_attention_mask,added_cond_kwargs,latent_channels):
    device=alpha_schedule.device
    zt=zt.to(alpha_schedule.device)
    t=t.to(alpha_schedule.device)
    s=s.to(alpha_schedule.device)
    prompt_embeds=prompt_embeds.to(zt.dtype)
    c_skip, c_out = scalings_for_boundary_conditions(t)
    c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
    c_skip=c_skip.to(zt.dtype)
    c_out=c_out.to(zt.dtype)
    time=time2tensor(t,zt)
    # print("zt",zt.shape)
    # print("time",time.shape)
    # print("prompt_embeds",prompt_embeds.shape)
    # print("prompt_attention_mask",prompt_attention_mask.shape)
    # print("added_cond_kwargs",added_cond_kwargs)
    noise = model(
            zt.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            encoder_attention_mask=prompt_attention_mask.to(device),
            timestep=time.to(device),
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    
    if model.module.config.out_channels // 2 == latent_channels:
        noise = noise.chunk(2, dim=1)[0]
    else:
        noise = noise
    
    alpha_t = alpha_schedule[t].view(-1, 1, 1, 1,1)
    sigma_t = sigma_schedule[t].view(-1, 1, 1, 1,1)
    alpha_s=alpha_schedule[s].view(-1,1,1,1,1)
    sigma_s=sigma_schedule[s].view(-1,1,1,1,1)
    pred_x = (zt - sigma_t * noise) / alpha_t
    f_x = c_skip * zt + c_out * pred_x
    result=alpha_s*f_x+sigma_s*noise
    # print("denoise",result.dtype)
    return result

def generate_intermediate_t_vectors(steps,step,step_value,t,bsz,device):
    
    # Create a tensor to hold all intermediate t vectors for each batch element
    intermediate_ts = torch.zeros(steps - step_value.item()+1, bsz, device=device)

    # Calculate intermediate values for t for each batch element
    for i in range(bsz):
        end_t = t[i].item()
        start_t = 999
        num_intervals = steps - step[i].item()+1
        if num_intervals > 0:
            # Create evenly spaced values between start_t and end_t
            intermediate_ts[:num_intervals, i] = torch.linspace(start_t, end_t, num_intervals, device=device)

    return intermediate_ts

args = parse_args()

logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
)
tokenizer= T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

def prepare_latents(scheduler, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
    vae_scale_factor = ae_stride_config["CausalVAEModel_D4_4x8x8"]
    shape = (
        batch_size,
        num_channels_latents,
        (math.ceil((int(num_frames) - 1) / vae_scale_factor[0]) + 1) if int(num_frames) % 2 == 1 else math.ceil(int(num_frames) / vae_scale_factor[0]), 
        math.ceil(int(height) / vae_scale_factor[1]),
        math.ceil(int(width) / vae_scale_factor[2]),
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    # latents = latents * scheduler.init_noise_sigma


    return latents

def main(args):
    device=accelerator.device
    weight_dtype = torch.float16
    num_frames = 29
    height = 720
    width = 1280
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.image_dir is not None:
            os.makedirs(args.image_dir, exist_ok=True)
            
        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
                private=True,
            ).repo_id
    
    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir,
                                                  low_cpu_mem_usage=True, torch_dtype=weight_dtype).to(device)

    teacher_transformer = OpenSoraT2V.from_pretrained(args.pretrained_teacher_model)

    scheduler = DPMSolverMultistepScheduler()
    alpha_schedule = torch.sqrt(scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1-scheduler.alphas_cumprod)
    alpha_schedule=alpha_schedule.to(accelerator.device).to(weight_dtype)
    sigma_schedule=sigma_schedule.to(accelerator.device).to(weight_dtype)
    
    # vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_transformer.requires_grad_(False)

    student_transformer = OpenSoraT2V.from_pretrained(args.pretrained_student_model)
    transformer = OpenSoraT2V.from_config(student_transformer.config)
    transformer.register_to_config(**student_transformer.config)
    transformer.load_state_dict(student_transformer.state_dict())
    del student_transformer


    transformer.train()


    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(transformer).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(transformer).dtype}. {low_precision_error_string}"
        )
    
    # print(vae.device)
    transformer =transformer.to(weight_dtype).to(device)
    
    # os.system("gpustat")
    teacher_transformer=teacher_transformer.to(weight_dtype).to(accelerator.device)
    # os.system("gpustat")
    
    # vae=vae.to(weight_dtype).to(accelerator.device)
    # text_encoder.to(accelerator.device)
    # print(transformer.device,text_encoder.device,vae.device)
    os.system("gpustat")
    for param in transformer.parameters():
        # only upcast trainable parameters (LoRA) into fp32
        if param.requires_grad:
            param.data = param.to(torch.float32)


    # os.system("gpustat")
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            input_dir = os.path.join(input_dir, "transformer")
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                
                print("input_dir",input_dir)
                # load diffusers style into model
                load_model = OpenSoraT2V.from_pretrained(input_dir)
                model.register_to_config(**load_model.config)
 
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    
    transformer.gradient_checkpointing = True
    # if args.gradient_checkpointing:
    #     transformer.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    optimizer = optimizer_class(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = Text2ImageDataset(
        jsonl_path=args.train_shards_path_or_url,
        num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size,
        global_batch_size=args.train_batch_size * accelerator.num_processes,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader =dataset.create_dataloader()

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    
    transformer ,optimizer, lr_scheduler = accelerator.prepare(transformer ,optimizer, lr_scheduler)
   
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {train_dataloader.num_batches}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = args.learning_rate
                
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # print('steps in one epoch',expect_len)
    steps=args.num_train_inferences
    Tstep=len(alpha_schedule)/steps
    Tstep=round(Tstep)
    latent_channels = transformer.module.config.in_channels

    vae_scale_factor = ae_stride_config["CausalVAEModel_D4_4x8x8"]
    length_alpha_schedule = len(alpha_schedule)

    print('Tstep',Tstep)
    print('first_epoch',first_epoch)
    torch.cuda.empty_cache()

    print(len(scheduler)) 
    batch_size=args.train_batch_size
    num_images_per_prompt=1
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    # print("bbbbbb",teacher_transformer.config.sample_size)
    # if teacher_transformer.config.sample_size == 128:
    #     resolution = torch.tensor([args.resolution, args.resolution]).repeat(batch_size * num_images_per_prompt, 1)
    #     aspect_ratio = torch.tensor([float(args.resolution / args.resolution)]).repeat(batch_size * num_images_per_prompt, 1)
    #     resolution = resolution.to(dtype=weight_dtype, device=accelerator.device)
    #     aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=accelerator.device)

    # # if do_classifier_free_guidance:
    # #     resolution = torch.cat([resolution, resolution], dim=0)
    # #     aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

    # added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    generator = None
    for epoch in range(first_epoch, args.num_train_epochs):
        # print("Epoch: ", epoch,"/num", args.num_train_epochs)
        # N_i=exponential_schedule(epoch+1, args.num_train_epochs, N_start, N_max)
        # print('N_i',N_i)
        for index, batch in enumerate(train_dataloader):
            # if(index<1000):
            #     continue
            with accelerator.accumulate(transformer):
                # if args.resume_from_checkpoint and global_step == initial_global_step:  # Only update once
                #     for param_group in optimizer.param_groups:
                #         param_group["lr"] = args.learning_rate
                #     lr_scheduler.base_lrs = [args.learning_rate]
                #     print(f"Updated learning rate to: {args.learning_rate}")
                text =batch['text']
                # print('text',text)
                # latents = randn_tensor(
                #     (args.train_batch_size, latent_channels, int(args.resolution) // vae_scale_factor, int(args.resolution) // vae_scale_factor),
                #     dtype=weight_dtype,layout=torch.strided,device=accelerator.device)
                latents = prepare_latents(
                    scheduler,
                    batch_size * num_images_per_prompt,
                    latent_channels,
                    num_frames, 
                    height,
                    width,
                    weight_dtype,
                    accelerator.device,
                    generator,
                    None,
                )
                latents = latents.to(accelerator.device)
                # print('bsz',latents.shape[0])
                # print('latents',latents.shape)
                # Sample noise that we'll add to the latents
                # print(latents.shape)
                device=latents.device
                bsz = latents.shape[0]
                step_value = torch.randint(0, steps, (1,), device=device).long()
                step = step_value.expand(bsz)

                # Randomly select a relative step
                nrel = torch.randint(20, Tstep + 20, (bsz,), device=device).long()

                # Compute initial tstep and t
                tstep = step.float() / steps 

                t = tstep + nrel.float() / length_alpha_schedule
                # Ensure t does not exceed 1 - 1/len(alpha_schedule)
                t = torch.clamp(t, max=1 - 1.0 / length_alpha_schedule)
                # Compute s
                s = t - 20.0 / length_alpha_schedule

                # Round t, s, and tstep to nearest indices in alpha_schedule
                t = torch.round(t * length_alpha_schedule).long()
                s = torch.round(s * length_alpha_schedule).long()
                tstep = torch.round(tstep * length_alpha_schedule).long()

                intermediate_t_vectors = generate_intermediate_t_vectors(steps,step,step_value,t,bsz,device)
                intermediate_t_vectors = intermediate_t_vectors.to(torch.int64)
                guidance_scale=4.5

                (
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                ) = encode_prompt(
                    text_encoder=text_encoder,
                    prompt=text,
                    do_classifier_free_guidance=True,
                    negative_prompt="",
                    num_images_per_prompt=1,
                    device=latents.device,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    prompt_attention_mask=None,
                    negative_prompt_attention_mask=None,
                    clean_caption=True,
                    max_sequence_length=512,
                )

                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                if prompt_attention_mask.ndim == 2:
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l

                if negative_prompt_embeds.ndim == 3:
                    negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)
                if negative_prompt_attention_mask.ndim == 2:
                    negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(1)

                # print("index",index)
                for id_t,t_now in enumerate(intermediate_t_vectors):
                    if id_t==len(intermediate_t_vectors)-1:
                        break
                    t_next=intermediate_t_vectors[id_t+1]
                    # print("t_now",t_now)
                    # print("t_next",t_next)
                    latents=ddim_solver(teacher_transformer,latents,t_now.to(alpha_schedule.device),t_next.to(alpha_schedule.device),
                                        alpha_schedule,sigma_schedule,latents,prompt_embeds,prompt_attention_mask,
                                        negative_prompt_embeds,negative_prompt_attention_mask,added_cond_kwargs,
                                weight_dtype,latent_channels,guidance_scale)
                    
                zt=latents
                # print(accelerator.device)
                # print(transformer.device)
                # print("t",t)
                # print("s",s)
                # print("tstep",tstep)
                zs=ddim_solver(teacher_transformer,zt.to(alpha_schedule.device),t.to(alpha_schedule.device),s.to(alpha_schedule.device),
                               alpha_schedule,sigma_schedule,latents,prompt_embeds,prompt_attention_mask,
                                negative_prompt_embeds,negative_prompt_attention_mask,added_cond_kwargs,
                                weight_dtype,latent_channels,guidance_scale)
                with torch.no_grad():
                    z_ref_s=denoise(transformer,zs,s,tstep,alpha_schedule,sigma_schedule,latents,prompt_embeds,prompt_attention_mask,added_cond_kwargs,latent_channels)
                z_ref_t=denoise(transformer,zt,t,tstep,alpha_schedule,sigma_schedule,latents,prompt_embeds,prompt_attention_mask,added_cond_kwargs,latent_channels)
                ref_diff=z_ref_s-z_ref_t
                # print("zt",zt[0])
                # print("zs",zs[0])
                # print("z_ref_s",z_ref_s[0])
                # print("z_ref_t",z_ref_t[0])
                # print("ref_diff",ref_diff[0])
                print('ref_diff',ref_diff.shape)
                print('ref_diff and view',ref_diff.view(bsz,-1).shape)
                loss=torch.norm(ref_diff.view(bsz,-1),dim=1)
                # print('loss',loss.shape)
                loss=loss.mean()
                print('loss',loss)
                # print('loss',loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # print(transformer.module.dtype)
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 20.4.15. Make EMA update to target student model parameters
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # if global_step % args.validation_steps == 0:

                    #     log_validation(vae, unet, args, accelerator, weight_dtype, global_step, "online")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(os.path.join(args.output_dir, "transformer"))

    print('training finished')
    accelerator.end_training()


if __name__ == "__main__":
    main(args)