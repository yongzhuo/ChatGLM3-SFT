# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: fastapi-post接口


import traceback
import logging
import random
import time
import json
import sys
import os
import re

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from chatglm3_sft.ft_chatglm3.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
# from rouge import Rouge  # pip install rouge
# from tqdm import tqdm
import torch

from pydantic import BaseModel
from fastapi import FastAPI
import time

# from transformers import ChatGLMForConditionalGeneration, ChatGLMConfig
# from transformers import ChatGLMTokenizer
from chatglm3_sft.models.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatglm3_sft.models.tokenization_chatglm import ChatGLMTokenizer
from chatglm3_sft.ft_chatglm3.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from chatglm3_sft.ft_chatglm3.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from chatglm3_sft.ft_chatglm3.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from chatglm3_sft.ft_chatglm3.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from chatglm3_sft.ft_chatglm3.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from chatglm3_sft.ft_chatglm3.config import USE_CUDA


app = FastAPI()  # 日志文件名,为启动时的日期, 全局日志格式
logger_level = logging.INFO
logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] "
                           "- %(levelname)s: %(message)s",
                    level=logger_level)
logger = logging.getLogger("ft-chatglm3")
console = logging.StreamHandler()
console.setLevel(logger_level)
logger.addHandler(console)


def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.bin"):
    """  仅保存 有梯度 的 模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def load_model_state(model, model_save_dir="./", model_name="adapter_model.bin", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)
        state_dict = torch.load(path_model, map_location=torch.device(device))
        state_dict = {k.replace("_orig_mod.", "")
                      .replace(".lora_A.weight", ".lora_A.default.weight")
                      .replace(".lora_B.weight", ".lora_B.default.weight")
                      : v for k, v in state_dict.items()}
        print(state_dict.keys())
        print("11111#"*128)
        ### 排查不存在model.keys的 state_dict.key
        name_dict = {name: 0 for name, param in model.named_parameters()}
        print(name_dict.keys())
        print("22222#"*128)
        for state_dict_key in state_dict.keys():
            if state_dict_key not in name_dict:
                print("{} is not exist!".format(state_dict_key))
        model.load_state_dict(state_dict, strict=False)
        # model.to(device)
        print("******model loaded success******")
        print("self.device: {}".format(device))
    except Exception as e:
        print(str(e))
        raise Exception("******load model error******")
    return model
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json
def generate_prompt(data_point, is_logger=False):
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"
    instruction = data_point.get("instruction", "").strip()
    text_input = data_point.get("input", "").strip()
    text_output = data_point.get("output", "")
    history = data_point.get("history", [])
    string_system = instruction
    string_print = TOKEN_SYSTEM + "\n" + string_system + "\n"
    ids_prompt_system = [ID_gMASK, ID_SOP, ID_system] + tokenizer.encode(
                            "\n" + string_system + "\n", is_split_into_words=True)[2:]
    # ids_text_input = [ID_assistant] + tokenizer.encode("\n", is_split_into_words=True)[2:]
    # ids_text_input = [ID_assistant] + tokenizer.encode("\n", is_split_into_words=True)[2:]
    ids_history = []
    history.append(text_input)
    for idx, item in enumerate(history):
        content = item
        if idx % 2 == 0:
            id_role = ID_user
            token_role = TOKEN_USER
        else:
            id_role = ID_assistant
            token_role = TOKEN_ASSISTANT
        string_print += token_role + "\n" + content + "\n"
        id_history = [id_role] + tokenizer.encode("\n" + content + "\n",
                                                  is_split_into_words=True)[2:]
        ids_history.extend(id_history)

    ids_prompt_input = ids_prompt_system + ids_history
    if len(ids_prompt_input) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        # 如果超出文本长度了
        ids_prompt_input = ids_prompt_input[:MAX_LENGTH_Q+MAX_LENGTH_A]
    ids_prompt_input += [ID_assistant] + tokenizer.encode("\n", is_split_into_words=True)[2:]
    out = {"input_ids": ids_prompt_input, "labels": []}
    if is_logger:
        print(string_print)
        print(out)
    return out


tokenizer = ChatGLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Allow batched inference
ID_MASK = 64789
ID_gMASK = 64790
ID_sMASK = 64791
ID_SOP = 64792
ID_EOP = 64793
ID_PAD = 64793
ID_system = 64794
ID_user = 64795
ID_assistant = 64796
ID_observation = 64797
IDS_END = [ID_user, ID_system, ID_EOP]   # start, <|user|>
# "<|system|>": 64794,
# "<|user|>": 64795,
# "<|assistant|>": 64796,
# "<|observation|>": 64797
TOKEN_ASSISTANT = "<|assistant|>"
TOKEN_SYSTEM = "<|system|>"
TOKEN_USER = "<|user|>"
TOKEN_START = "sop"
TOKEN_END = "eop"


model = ChatGLMForConditionalGeneration.from_pretrained(PATH_MODEL_PRETRAIN)
model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# model.is_parallelizable = False
# model.config.use_cache = False
# model.model_parallel = False

print_named_parameters(model, True)
model = load_model_state(model=model, model_save_dir=MODEL_SAVE_DIR)
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=False,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["post_attention_layernorm",
                          "final_layernorm",
                          "input_layernorm",
                          ],
        )
if USE_CUDA:
    model = model.half().cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, True)


def txt_read(path, encode_type="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encode_type, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines
def predict(data_point, generation_config):
    """  推理  """
    prompt_dict = generate_prompt(data_point)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    # input_dict = data_collator([prompt_dict])
    # if USE_CUDA:
    #     input_dict = {k:v.cuda() for k,v in input_dict.items()}
    # print(input_dict)
    generation_config = GenerationConfig(**generation_config)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            # max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(input_ids)
    print(s)
    print(output)
    return output
# prompt_system_std = "You are a helpful, safety and harmless assistant."
# prompt_system_std = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."

class Item(BaseModel):
    instruction: str = "You are a helpful, safety and harmless assistant."
    text: str = "What's the weather like today?"
    num_return_sequences = 1
    penalty_alpha: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.8  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.8  # 0.75
    top_k: int = 62
    history: list = []

@app.post("/nlg/text_generate/chatglm3")
def text_generate(request_data: Item):
    instruction = request_data.instruction
    text = request_data.text
    penalty_alpha = request_data.penalty_alpha
    max_new_tokens = request_data.max_new_tokens
    temperature = request_data.temperature
    do_sample = request_data.do_sample
    num_beams = request_data.num_beams
    top_p = request_data.top_p
    top_k = request_data.top_k
    history = request_data.history

    generation_dict = vars(request_data)
    print(generation_dict)
    generation_dict.pop("max_new_tokens")
    generation_dict.pop("instruction")
    generation_dict.pop("history")
    generation_dict.pop("text")
    data_point = {"instruction": instruction, "input": text,
                  "output": "", "history": history}
    generation_config = {"temperature": temperature,
                         "top_p": top_p,
                         "top_k": top_k,
                         "num_beams": num_beams,
                         "do_sample": do_sample,
                         "penalty_alpha": penalty_alpha,
                         "max_new_tokens": max_new_tokens,
                         "eos_token_id": IDS_END,
                         }
    response_answer = ""
    response = {}
    try:  # 数据预处理, 模型预测
        response = predict(data_point, generation_config)
        # response_answer = response.split(TOKEN_ASSISTANT)[-1].strip()
        response_zh = response.replace(TOKEN_ASSISTANT, "QA机器人").replace(TOKEN_USER, "QA用户")
        response_zh_sp = re.split(r"QA机器人|QA用户", response_zh)
        if len(response_zh_sp) > 1:
            response_zh_sp_qa = response_zh_sp[1:]
            if len(response_zh_sp_qa) > len(history):
                response_answer = response_zh_sp_qa[len(history)].strip() .replace(TOKEN_ASSISTANT, "")\
                                        .replace(TOKEN_USER, "").replace("\n", "\t").strip()
    except Exception as e:
        logger.info(traceback.print_exc())
        response_answer = TOKEN_END
    return {"response_answer": response_answer,
            "history": history + [response_answer],
            "response": response
            }



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8062,
                workers=1)


"""
# nohup python post_api.py > tc.post_api.py.log 2>&1 &
# tail -n 1000  -f tc.post_api_std.py.log
# |myz|

可以在浏览器生成界面直接访问: http://localhost:8032/docs

"""