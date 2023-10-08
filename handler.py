import os, glob
import logging
import json
from typing import Generator, Union
import runpod
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    pipeline,
)
import logging
from peft import LoraConfig, PeftModel
from copy import copy

import re
import codecs


ESCAPE_SEQUENCE_RE = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )''', re.UNICODE | re.VERBOSE)

def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

def load_model(max_new_tokens):
    global generator, default_settings

    load = True
    if load:
        model_directory = snapshot_download(repo_id=os.environ["MODEL_REPO"], revision=os.getenv("MODEL_REVISION", "main"))
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        st_files = glob.glob(st_pattern)
        if not st_files:
            raise ValueError(f"No safetensors files found in {model_directory}")
        model_path = st_files[0]

        use_4bit = True
        use_nested_quant = False
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        config  = AutoConfig.from_pretrained(model_config_path)
        model = AutoModelForCausalLM.from_pretrained(model_directory,config=config, quantization_config = bnb_config)

        tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
        
        pipe = pipeline(task="text-generation", model = model, tokenizer = tokenizer, max_new_tokens = max_new_tokens)

        return pipe

        # EXLLAMA IMPLEMENTAION
        # Create config, model, tokenizer and generator
    #     config = ExLlamaConfig(model_config_path)               # create config from config.json
    #     config.model_path = model_path                          # supply path to model weights file

    #     gpu_split = os.getenv("GPU_SPLIT", "")
    #     if gpu_split:
    #         config.set_auto_map(gpu_split)
    #         config.gpu_peer_fix = True
    #     alpha_value = int(os.getenv("ALPHA_VALUE", "1"))
    #     config.max_seq_len = int(os.getenv("MAX_SEQ_LEN", "2048"))
    #     if alpha_value != 1:
    #         config.alpha_value = alpha_value
    #         config.calculate_rotary_embedding_base()

    #     model = ExLlama(config)                                 # create ExLlama instance and load the weights
    #     tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

    #     cache = ExLlamaCache(model)                             # create cache for inference
    #     generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
    #     default_settings = {
    #         k: getattr(generator.settings, k) for k in dir(generator.settings) if k[:2] != '__'
    #     }
    # return generator, default_settings

generator = None
default_settings = None
prompt_prefix = decode_escapes(os.getenv("PROMPT_PREFIX", ""))
prompt_suffix = decode_escapes(os.getenv("PROMPT_SUFFIX", ""))

def generate_with_streaming(prompt, max_new_tokens):
    global generator
    generator.end_beam_search()

    # Tokenizing the input
    ids = generator.tokenizer.encode(prompt)
    ids = ids[:, -generator.model.config.max_seq_len:]

    generator.gen_begin_reuse(ids)
    initial_len = generator.sequence[0].shape[0]
    has_leading_space = False
    for i in range(max_new_tokens):
        token = generator.gen_single_token()
        if i == 0 and generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
            has_leading_space = True

        decoded_text = generator.tokenizer.decode(generator.sequence[0][initial_len:])
        if has_leading_space:
            decoded_text = ' ' + decoded_text

        yield decoded_text
        if token.item() == generator.tokenizer.eos_token_id:
            break

def inference(event) -> Union[str, Generator[str, None, None]]:
    logging.info(event)
    job_input = event["input"]
    if not job_input:
        raise ValueError("No input provided")

    prompt: str = job_input.pop("prompt_prefix", prompt_prefix) + job_input.pop("prompt") + job_input.pop("prompt_suffix", prompt_suffix)
    max_new_tokens = job_input.pop("max_new_tokens", 100)
    stream: bool = job_input.pop("stream", False)
    
    pipe = load_model(max_new_tokens)
    
    ####
    # generator, default_settings = load_model(max_new_tokens)

    # settings = copy(default_settings)
    # settings.update(job_input)
    # for key, value in settings.items():
    #     setattr(generator.settings, key, value)

    if stream:
        print("Streaming not supported now bye bye")
        # output: Union[str, Generator[str, None, None]] = generate_with_streaming(prompt, max_new_tokens)
        # for res in output:
        #     yield res
    else:
        result = pipe(prompt)
        output_text = result[0]["generated_text"]
        output_text = generator.generate_simple(prompt, max_new_tokens = max_new_tokens)
        yield output_text[len(prompt):]

runpod.serverless.start({"handler": inference})
