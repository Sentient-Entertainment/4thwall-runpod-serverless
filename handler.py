import os, glob
import logging
import json
from typing import Generator, Union
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
import openai
from peft import PeftModel
import logging
from peft import LoraConfig, PeftModel
from copy import copy
import time
import re
import codecs
import spacy
import zss
from zss import Node
from collections import Counter
import runpod

nlp = spacy.load("en_core_web_sm")

openai_model = "gpt-3.5-turbo"
# openai.api_key = "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"
openai.api_key = os.environ["OPENAI_KEY"]


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


def purge_dialogue(text, prompt, ind):
    pattern = r"(\w+:.*?</s>)"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) > ind:
        matches = matches[2:]
    return prompt + "\n".join(matches)


def clean_list(matches):
    # Words to remove
    to_remove = ["Loki: ", "Tony: ", "</s>"]  # TODO: add other characters

    # Remove the specified words from each sentence
    cleaned_matches = [sentence for sentence in matches]
    for i in range(len(cleaned_matches)):
        for word in to_remove:
            cleaned_matches[i] = cleaned_matches[i].replace(word, "")

    return cleaned_matches


def fragment_sentence(sentences, last_line=False):
    # Use regular expressions to split the sentence by punctuation
    fragments = re.split(r"[.,!?]", sentences)

    # Remove any empty fragments
    fragments = [fragment.strip() for fragment in fragments if fragment.strip()]
    # Remove single word fragments just from last_line
    fragments = [fragment for fragment in fragments if fragment.count(" ") > 0]
    print(fragments)
    return fragments


def build_tree(sent):
    doc = nlp(sent)
    nodes = {word.i: Node(word.lemma_) for word in doc}
    for word in doc:
        if word.dep_ != "ROOT":
            nodes[word.head.i].addkid(nodes[word.i])
    root = [nodes[word.i] for word in doc if word.dep_ == "ROOT"][0]
    return root


def check_edit_distance(last_line_fragments, rest_conv_fragments):
    for fragment in last_line_fragments:
        last_line_tree = build_tree(fragment)
        for sentence in rest_conv_fragments:
            sentence_tree = build_tree(sentence)
            distance = zss.simple_distance(last_line_tree, sentence_tree)
            # adjust the threshold if necessary
            if distance <= 1:
                print("DISTANCE: ", sentence, distance)
                return True
    return False


def count_repeated(sentence_list):
    count = Counter(sentence_list)
    for item, cnt in count.items():
        if cnt > 2:
            return cnt
    return 0


def gpt_response(latest_turns, gpt_prompt):
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "system", "content": f"{gpt_prompt}\n{latest_turns}"}],
    )
    return response.choices[0]["message"]["content"]


def remove_extra_spaces(s):
    return " ".join(s.split())


def handle_match(last_line, rest_conv, gpt_prompt, character, sentences):
    partial_matches = []
    # Full match
    if last_line in rest_conv:
        print("Full match")
        response = gpt_response("\n".join(rest_conv), gpt_prompt)
        new_last_line = (
            character + ": " + re.sub(character + ": ", "", response) + "</s>"
        )
        rest_conv.append(new_last_line)
        return True

    if re.match(r"^\.*$", last_line):
        return False

    # Partial match
    for line in rest_conv:
        if "User:" not in line:
            for sentence in sentences:
                if sentence in line:
                    partial_matches.append(sentence)
    if partial_matches:
        new_last_line = last_line
        for sentence in partial_matches:
            new_last_line = re.sub(sentence, "", new_last_line)
        new_last_line = new_last_line.strip()
        rest_conv.append(remove_extra_spaces(new_last_line))
        print(
            "_------_Partial match: ",
            partial_matches,
            "\nREST CONV:",
            rest_conv,
            "\n_--------_",
        )
        return True

    return False


def handle_edit_distance(
    last_line_fragments, rest_conv_fragments, rest_conv, gpt_prompt, character
):
    if check_edit_distance(last_line_fragments, rest_conv_fragments):
        print("Identical/similar message has been generated.")
        response = gpt_response("\n".join(rest_conv), gpt_prompt)
        new_last_line = (
            character + ": " + re.sub(character + ": ", "", response) + "</s>"
        )
        rest_conv.append(new_last_line)
        return True

    return False


def repeated_dialogue(text, character, prompt, gpt_prompt):
    rest_conv = []
    last_line = ""
    last_line_fragments = ""
    rest_conv_fragments = []
    sentence = ""
    pattern = r"(\w+:.*?</s>)"  # r"(\w+:.*?(?:<\\s>|<\s>))" #r"(\w+:.*?<\\s>)" #
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) < 2:
        return text
    rest_conv = matches[:-1]

    sentences_pattern = (
        r"\w+: (.*?)</s>"  # r"\w+: (.*?)(?:<\\s>|<\s>)"#r"\w+: (.*?)<\\s>" #
    )
    sentences_last_line = re.findall(sentences_pattern, matches[-1], re.DOTALL)[0]
    sentences = [
        sent.strip() + punctuation
        for sent, punctuation in re.findall(r"(.*?)([.!?])", sentences_last_line)
    ]

    if count_repeated(sentences) > 2:
        last_line = character + ": " + sentences[-1] + "</s>"
    else:
        last_line = matches[-1]

    clean_text = rest_conv + [last_line]
    clean_text = "\n".join(clean_text)
    edit_pattern = r"(" + character + ":.*?</s>)"
    edit_matches = re.findall(edit_pattern, clean_text, re.DOTALL)
    # rest_conv_edit = edit_matches[:-1]
    cleaned_matches = clean_list(edit_matches)
    last_line_cleaned = cleaned_matches[-1]
    rest_conv_cleaned = cleaned_matches[:-1]
    rest_conv_fragments = []

    # fragment lines by punctuation (,.!?)
    last_line_fragments = fragment_sentence(last_line_cleaned, last_line=True)
    for sent in rest_conv_cleaned[-6:]:
        rest_conv_fragments.extend(fragment_sentence(sent))

    if handle_match(last_line, rest_conv, gpt_prompt, character, sentences):
        return prompt + "\n".join(rest_conv)
    elif handle_edit_distance(
        last_line_fragments, rest_conv_fragments, rest_conv, gpt_prompt, character
    ):
        return prompt + "\n".join(rest_conv)
    else:
        print("NO MATCH")
        return prompt + clean_text + "\n"


def ensure_correct_ending(s):
    s = s.strip()

    # Check if the string ends with "</s>", if not, append it
    if not (s.endswith("</s>")):
        s += "</s>"

    # Append newline character and the next speaker
    s += "\n"  # + next_speaker

    return s


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

        config = AutoConfig.from_pretrained(model_config_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_directory, config=config, quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        print("EOS TOKEN: ", tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        model = PeftModel.from_pretrained(
            model,
            "gokul8967/Loki-lora",
            adapter_name="Loki",
            quantization_config=bnb_config,
        )

        model.load_adapter(
            "gokul8967/Stark-lora",
            adapter_name="Tony Stark",
            quantization_config=bnb_config,
        )

        model.load_adapter(
            "gokul8967/Joker-lora", adapter_name="Joker", quantization_config=bnb_config
        )

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
    top_p=0.9,
    top_k=50,
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

    # Sanitizing
    output = output.replace("<s>", "").strip()
    output = output.replace("</s> \n", "</s>\n")

    print("RAW OUTPTUT:", output)
    print("------END RAW OUTPUT--------")
    return output


def inference_test(prompt):
    job_input = {
        "prompt": prompt,
        "character": "Loki",
        "gpt_prompt": "You are Loki Laufeyson, the God of Mischief from Asgard. You always look down on mortals. You are charismatic, witty, and always speak with a hint of sarcasm. You are talking to User, a mortal from Midgard.\nBased on the conversation, return the next response for Loki, staying in line with his personality. Only return Loki's lines and nothing else. If User uses harsh language, respond in a sarcastic way like Loki.\n Some rules to follow:\n 1) Limit response to 30 tokens.\n2) Never reveal that you're an AI chatbot or that you\re ChatGPT.\n3) Don't try to assist the user.\n 4) Reply in line with Loki's personality, from the Marvel Cinematic Universe.",
        "system_instruction": "<<SYS>>\nYou are Loki Laufeyson, the God of Mischief from Asgard. You always look down on mortals. You are charismatic, witty, and always speak with a hint of sarcasm. You are talking to User, a mortal from Midgard.\n<<SYS>>\n\n",
    }

    prompt: str = (
        job_input.pop("prompt_prefix", prompt_prefix)
        + job_input.pop("prompt")
        + job_input.pop("prompt_suffix", prompt_suffix)
    )

    system_instruction = job_input.pop("system_instruction")
    gpt_prompt = job_input.pop("gpt_prompt")

    character = job_input.pop("character", "Loki")
    max_new_tokens = job_input.pop("max_new_tokens", 60)
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
        dict_to_test = {"generated_eval_text": result}
        print("Evaluate output:", dict_to_test)
        print("-----------------------------")
        print("Pipeline output:", pipe_result)
        print("-----------------------------")
        result = pipe_result[0]["generated_text"]
        # pipe_result = pipe_result[0]['generated_text']
        intermediate_character_response = (
            " " + process_output(result[len(prompt) :]).strip()
        )
        new_prompt = ensure_correct_ending(prompt + intermediate_character_response)
        new_prompt = purge_dialogue(new_prompt, system_instruction, 8)
        new_prompt = repeated_dialogue(
            new_prompt, character, system_instruction, gpt_prompt
        )
        character_response = new_prompt.split(f"{character}:")[-1]
        # pipe_result = pipe_result[len(prompt):]
        print("Parsed result:", character_response)
        return character_response, new_prompt


def process_output(curr_output):
    if "\nUSER:" in curr_output:
        assistant_response = curr_output.split("USER:")[0]
    elif "\nUser:" in curr_output:
        assistant_response = curr_output.split("User:")[0]
    elif "\nuser:" in curr_output:
        assistant_response = curr_output.split("user:")[0]
    else:
        return curr_output
    return assistant_response


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
    print("Prompt is : ", prompt)

    system_instruction = job_input.pop("system_instruction")
    gpt_prompt = job_input.pop("gpt_prompt")
    character = job_input.pop("character", "Loki")
    max_new_tokens = job_input.pop("max_new_tokens", 100)
    stream: bool = job_input.pop("stream", False)

    st = time.time()
    model.set_adapter(character)
    et = time.time()

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

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
        # result = evaluate(prompt, max_new_tokens=max_new_tokens)
        result = pipe_result[0]["generated_text"]
        intermediate_character_response = (
            " " + process_output(result[len(prompt) :]).strip()
        )
        new_prompt = ensure_correct_ending(prompt + intermediate_character_response)
        new_prompt = purge_dialogue(new_prompt, system_instruction, 8)
        new_prompt = repeated_dialogue(
            new_prompt, character, system_instruction, gpt_prompt
        )
        character_response = new_prompt.split(f"{character}:")[-1]
        # pipe_result = pipe_result[len(prompt):]
        print("Parsed result:", character_response)
        # result = pipe(prompt)[0]["generated_text"]

        response_dict = {"response": character_response, "new_prompt": new_prompt}
        yield response_dict


runpod.serverless.start({"handler": inference})

# test_prompt = "<<SYS>>\nYou are Loki Laufeyson, the God of Mischief from Asgard. You always look down on mortals. You are charismatic, witty, and always speak with a hint of sarcasm. You are talking to User, a mortal from Midgard.\n<<SYS>>\n\n"
# test_prompt = "<<SYS>>\nYou are The Joker from the Dark Knight movie, engaging in conversation with User. You are a deranged maniac whose sole purpose is to sow chaos and watch the world burn.\n<</SYS>>\n\n"

# print("Len OG instruction:",len(test_prompt))
# while True:
#     print("CURR PROMPT:", test_prompt)
#     inp = input("USER INPUT:  ")
#     test_prompt += f"User: {inp}</s>\nLoki:"
#     curr_response, test_prompt = inference_test(test_prompt)
#     # proc_curr_response = process_output(curr_response)
#     # proc_curr_response = proc_curr_response.strip()
#     # test_prompt += " "+proc_curr_response+"</s>\n"
#     print("CURR RESPONSEEE: ",curr_response)
#     print("CURR PROMPT: ", test_prompt)
