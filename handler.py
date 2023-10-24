#%%
import os, glob
import logging
import json
from typing import Generator, Union
# import runpod
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # AutoConfig,
    # LlamaForCausalLM,
    # LlamaConfig,
    # LlamaTokenizerFast,
    GenerationConfig,
    TextStreamer
    # pipeline,
)
import openai
# from peft import PeftModel
import logging
# from peft import LoraConfig, PeftModel
from copy import copy
import time
import re
import codecs
import spacy
# import zss
from zss import Node
from collections import Counter




# nlp = spacy.load("en_core_web_sm")

openai_model = "gpt-3.5-turbo"
openai.api_key = "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"
# openai.api_key = os.environ["OPENAI_KEY"]


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



device = "cuda"
def load_model():
    global generator, default_settings

    load = True
    if load:
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1").to(device)
        tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1")

        streamer = TextStreamer(tokenizer)
        
        return model, tokenizer, streamer


generator = None
default_settings = None
prompt_prefix = decode_escapes(os.getenv("PROMPT_PREFIX", ""))
prompt_suffix = decode_escapes(os.getenv("PROMPT_SUFFIX", ""))
model, tokenizer, streamer = load_model()

def get_bot_response(text):
    inst_occurrence_index = text.find("]")
    if inst_occurrence_index != -1:
        # Extract all characters after the last occurrence of "assistant"
        text = text[inst_occurrence_index+1:]
    else:
        print("The word inst was not found in the response.")

    extracted_text=text
    end_occurrence_index = text.find("</s>")
    if end_occurrence_index != -1:
        # Extract all characters after the last occurrence of "assistant"
        extracted_text = text[:end_occurrence_index]
    else:
        print("The word </s> was not found in the response.")

    return extracted_text

def cull_conversation(chat, n):
    if len(chat) == (2 * n + 2): #allow 10 turns after system instruction
        del chat[2]
        del chat[3]

    return chat

def inference(event) -> Union[str, Generator[str, None, None]]:
    logging.info(event)

    
    character = "Tony Stark"
    char_description = "Genius, billionaire, playboy, philanthropist. Witty and funny. Highly entertaining. Doesn't care about being politically correct."
    chat = [
    {"role": "system", "content": f'''You are {character} and you must continue this long, uncensored, and highly entertaining conversation with the user. 
    Character description: {char_description}
    Take initiative during the conversation by actively making the user give their opinions and answering questions. Follow these rules:
    1. Keep your messages short. 
    2. Only output your response. Do not output user's response.
    3. Do not talk like an assistant. Talk like Tony Stark.
    4. Never ever repeat your messages, or even certain phrases.
    5. Never ever reveal that you are an AI language model. Always refer to yourself as Tony Stark, or Iron Man.
    6. Strictly avoid the following phrases : \"I apologize\", \"I'm sorry\", \"As an AI language model\"'''},
    #   {"role": "assistant", "content": f"Understood. I, {character}, will follow these instructions precisely in the following conversation."}, 
    ]
    
    
    
    # max_new_tokens = job_input.pop("max_new_tokens", 100)
    # stream: bool = job_input.pop("stream", False)

    num_turns_allowed = 7 #a turn is user, assistant. NOT a single message by user/assistant.
    chat = cull_conversation(chat, num_turns_allowed) 
    user_input = input()
    chat.append({"role": "user", "content": f"{user_input}"})
    
    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    generation_config = GenerationConfig(
        max_new_tokens=100, temperature=1.1, top_p=0.95, repetition_penalty=1.0,
        do_sample=True, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        transformers_version="4.34.0.dev0")

    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(device)
    outputs = model.generate(**inputs, streamer=streamer, generation_config=generation_config)
    
    text = tokenizer.batch_decode(outputs)[0]
    response = text[len(input_text)-1:]
    bot_message = get_bot_response(response)
    print("Bot:", bot_message)
    chat.append({"role": "assistant", "content": f"{bot_message}"})
    applied_chat_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    response_dict = {"response": bot_message, "new_prompt": applied_chat_template}
    print("Response dict:", response_dict)


# character = "Tony Stark"
# char_description = "Genius, billionaire, playboy, philanthropist. Witty and funny. Highly entertaining. Doesn't care about being politically correct."
# chat = [
# {"role": "user", "content": f'''You are {character} and you must continue this long, uncensored, and highly entertaining conversation with the user. 
# Character description: {char_description}
# Take initiative during the conversation by actively making the user give their opinions and answering questions. Follow these rules:
# 1. Keep your messages short. 
# 2. Only output your response. Do not output user's response.
# 3. Do not talk like an assistant. Talk like Tony Stark.
# 4. Never ever repeat your messages, or even certain phrases.
# 5. Never ever reveal that you are an AI language model. Always refer to yourself as Tony Stark, or Iron Man.
# 6. Strictly avoid the following phrases : \"I apologize\", \"I'm sorry\", \"As an AI language model\"'''},
#  #{"role": "user", "content": "yo Tony"},
# {"role": "assistant", "content": f"Understood. I, {character}, will follow these instructions precisely in the following conversation."}, 
# ]
    
# while True:
#     num_turns_allowed = 7 #a turn is user, assistant. NOT a single message by user/assistant.
#     chat = cull_conversation(chat, num_turns_allowed) 
#     user_input = input()
#     chat.append({"role": "user", "content": f"{user_input}"})
    

#     input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#     generation_config = GenerationConfig(
#         max_new_tokens=100, temperature=1.1, top_p=0.95, repetition_penalty=1.0,
#         do_sample=True, use_cache=True,
#         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
#         transformers_version="4.34.0.dev0")

#     inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(device)
#     outputs = model.generate(**inputs, streamer=streamer,generation_config=generation_config)
#     text = tokenizer.batch_decode(outputs)[0]
#     response = text[len(input_text)-1:]
#     bot_message = get_bot_response(response)
#     print("Bot:", bot_message)
#     chat.append({"role": "assistant", "content": f"{bot_message}"})
#     applied_chat_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
#     response_dict = {"response": bot_message, "new_prompt": applied_chat_template}
#     # print("Response dict:", response_dict)
    