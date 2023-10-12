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
    LlamaForCausalLM,
    LlamaConfig,
    LlamaTokenizerFast,
    GenerationConfig,
    pipeline,
)
from peft import PeftModel
import logging
from peft import LoraConfig, PeftModel
from copy import copy
import time
import re
import codecs


ESCAPE_SEQUENCE_RE = re.compile(
    r"""
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )""",
    re.UNICODE | re.VERBOSE,
)


def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), "unicode-escape")

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def load_model(max_new_tokens):
    global generator, default_settings

    # model_name = "meta-llama/Llama-2-13b-hf"
    # gokul_token='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'

    load = True
    if load:
        model_directory = snapshot_download(
            repo_id=os.environ["MODEL_REPO"],
            revision=os.getenv("MODEL_REVISION", "main"),
            token=os.environ["AUTH_TOKEN"],
        )
        # model_directory = snapshot_download(
        # repo_id=model_name,
        # token = gokul_token,
        # cache_dir = '/workspace/hub'
        # )
        # model_directory = "/workspace/hub/models--meta-llama--Llama-2-13b-hf/snapshots/db6b8eb1feabb38985fdf785a89895959e944936/"
        print("Model Directory: ", model_directory)
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

        config = LlamaConfig.from_pretrained(model_config_path)
        model = LlamaForCausalLM.from_pretrained(
            model_directory, config=config,  quantization_config=bnb_config
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_directory
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        model = PeftModel.from_pretrained(
            model, "gokul8967/Loki-lora", adapter_name="Loki", quantization_config=bnb_config
        )
 
        model.load_adapter("gokul8967/Stark-lora", adapter_name="Tony Stark", quantization_config=bnb_config)
        
        return model, tokenizer

generator = None
default_settings = None
prompt_prefix = decode_escapes(os.getenv("PROMPT_PREFIX", ""))
prompt_suffix = decode_escapes(os.getenv("PROMPT_SUFFIX", ""))
model, tokenizer = load_model(100)


def generate_with_streaming(prompt, max_new_tokens):
    global generator
    generator.end_beam_search()

    # Tokenizing the input
    ids = generator.tokenizer.encode(prompt)
    ids = ids[:, -generator.model.config.max_seq_len :]

    generator.gen_begin_reuse(ids)
    initial_len = generator.sequence[0].shape[0]
    has_leading_space = False
    for i in range(max_new_tokens):
        token = generator.gen_single_token()
        if i == 0 and generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith(
            "▁"
        ):
            has_leading_space = True

        decoded_text = generator.tokenizer.decode(generator.sequence[0][initial_len:])
        if has_leading_space:
            decoded_text = " " + decoded_text

        yield decoded_text
        if token.item() == generator.tokenizer.eos_token_id:
            break


def evaluate(
    prompt,
    input=None,
    temperature=0.7,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=100,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        do_sample=True,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output


def inference_test():
    job_input = {"prompt":"<<SYS>>\nYou are Loki Laufeyson, the God of Mischief from Asgard. You always look down on mortals. You are charismatic, witty, and always speak with a hint of sarcasm. You are talking to User, a mortal from Midgard\n<<SYS>>.\n\nUser:Hey loki boss</s>\nLoki:",
    "character":"Loki"}

    prompt: str = (
        job_input.pop("prompt_prefix", prompt_prefix)
        + job_input.pop("prompt")
        + job_input.pop("prompt_suffix", prompt_suffix)
    )
    character = job_input.pop("character", "Loki")
    max_new_tokens = job_input.pop("max_new_tokens", 100)
    stream: bool = job_input.pop("stream", False)

    st = time.time()
    model.set_adapter(character)
    et = time.time()

    print("Time to set adapter: ", (et - st))
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

    if stream:
        print("Streaming not supported now bye bye")
        # output: Union[str, Generator[str, None, None]] = generate_with_streaming(prompt, max_new_tokens)
        # for res in output:
        #     yield res
    else:
        result = evaluate(prompt)
        pipe_result = pipe(prompt)
        print("Evaluate output: ",result)
        print("Pipeline output: ", pipe_result)
        print("Parse evaluate output: ", result[len(prompt)+5 :])
    


def inference(event) -> Union[str, Generator[str, None, None]]:
    logging.info(event)
    job_input = event["input"]
    if not job_input:
        raise ValueError("No input provided")

    prompt: str = (
        job_input.pop("prompt_prefix", prompt_prefix)
        + job_input.pop("prompt")
        + job_input.pop("prompt_suffix", prompt_suffix)
    )
    character = job_input.pop("character", "Loki")
    max_new_tokens = job_input.pop("max_new_tokens", 100)
    stream: bool = job_input.pop("stream", False)

    st = time.time()
    model.set_adapter(character)
    et = time.time()

    print("Time to set adapter: ", (et - st))

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
        result = evaluate(prompt,max_new_tokens = max_new_tokens)
        yield result[len(prompt)+5 :]

# while True:
#     inp = input("Enter  ")
#     inference_test()

runpod.serverless.start({"handler": inference})
