# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/7 21:16
# @author  : Mo
# @function: ui-llm-sft
# @code    : most code from: https://huggingface.co/spaces/multimodalart/ChatGLM-6B/tree/main


import traceback
import requests
import logging
import json
import time
import sys
import os

num_limit = sys.getrecursionlimit()
sys.setrecursionlimit(num_limit * 100)
path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
print(path_sys)

import gradio as gr

import logging as logger


URL = "http://localhost:8062/nlg/text_generate/chatglm3"
TIMEOUT = 360


def post_sft(data_json, timeout=120, url=URL):
    """   请求微调后的模型   """
    no_repeat_ngram_size = data_json.get("no_repeat_ngram_size", 4)
    early_stopping = data_json.get("early_stopping", True)
    max_new_tokens = data_json.get("max_new_tokens", 512)
    instruction = data_json.get("instruction", "解答下面的问题。")
    penalty_alpha = data_json.get("penalty_alpha", 1.0)
    temperature = data_json.get("temperature", 0.8)
    do_sample = data_json.get("do_sample", True)
    num_beams = data_json.get("num_beams", 1)
    top_p = data_json.get("top_p", 1.0)
    top_k = data_json.get("top_k", 50)
    text = data_json.get("text", "")

    data = {"no_repeat_ngram_size": no_repeat_ngram_size,
            "early_stopping": early_stopping,
            "max_new_tokens": max_new_tokens,
            "penalty_alpha": penalty_alpha,
            "temperature": temperature,
            "instruction": instruction,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "top_p": top_p,
            "top_k": top_k,
            "text": text,
            }
    res = requests.post(url, json=data, timeout=timeout)
    res_json = json.loads(res.text)
    res.close()
    return res_json
def predict(max_new_tokens, penalty_alpha, instruction,
    temperature, top_p, top_k, query, history_chat=[]):
    print("history_chat:")
    print(history_chat)
    try:
        data_json = {"max_new_tokens": max_new_tokens,
                     "penalty_alpha": penalty_alpha,
                     "instruction": instruction,
                     "temperature": temperature,
                     "top_p": top_p,
                     "top_k": top_k,
                     "text": query,
                     # "history_chat": history_chat,
                     }
        logger.info("data_json is {}".format(data_json))
        res = post_sft(data_json=data_json,
                                     timeout=TIMEOUT,
                                     url=URL)
        logger.info(query)
        logger.info("res_json is {}".format(res))
        # "response": response, "history"
        response_answer = res.get("response_answer", "")
        history_chat.append((query, response_answer))
        return history_chat
    except Exception as e:
        return [(query, traceback.print_exc())]
def chatbot_clear(instruction):
    pass

with gr.Blocks() as demo:
    gr.Markdown("""## ChatGLM3""")
    state_top_1 = [("", "hello")]
    state = gr.State(state_top_1)
    chatbot = gr.Chatbot(state_top_1, elem_id="chatbot",
                label="chatbot").style(height=400)
    # chatbot.value
    with gr.Row():
        with gr.Column(scale=4):
            query = gr.Textbox(show_label=True,
                               label="query",
                               value="hello",
                               # placeholder="Enter text and press enter"
                               ).style(container=False)

        with gr.Column(scale=1):
            generate_button = gr.Button("Generate")
            clear_button = gr.Button("Clear")

    with gr.Accordion(label="Advanced options", open=False):
        # topic = gr.Radio(show_label=True,
        #                  label="topic",
        #                  value="天气",
        #                  choices=["天气", "时间", "宠物"],
        #                  ).style(container=False)
        instruction = gr.Textbox(label='Instruction',
                                   value="解答下面的问题。",
                                   lines=2)
        max_new_tokens = gr.Slider(
            label='Max new tokens',
            minimum=1,
            maximum=1024,
            step=1,
            value=512,
        )
        penalty_alpha = gr.Slider(
            label='Penalty Alpha',
            minimum=0.0,
            maximum=4.0,
            step=0.1,
            value=1.0,
        )
        temperature = gr.Slider(
            label='Temperature',
            minimum=0.0,
            maximum=4.0,
            step=0.1,
            value=0.1,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.01,
            maximum=1.0,
            step=0.05,
            value=0.8,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=32,
        )

    saved_input = gr.State(state_top_1)
    generate_button.click(fn=predict,
                          inputs=[max_new_tokens, penalty_alpha, instruction,
                                  temperature, top_p, top_k, query, chatbot],
                          outputs=chatbot)
    clear_button.click(fn=chatbot_clear,
                       inputs=[instruction],
                       outputs=[chatbot, saved_input],
                       queue=False,
                       api_name=False,
                       )

demo.queue().launch(server_name="0.0.0.0", server_port=8063, share=False, debug=True)


"""
必须是server_name="0.0.0.0", 默认的127.0.0.1不一定
访问IP地址:     http://localhost:8063/
"""

# nohup python ui_chatglm3_sft.py > tc.ui_chatglm3_sft.py.log 2>&1 &
# tail -n 1000  -f tc.ui_chatglm3_sft.py.log
# |myz|
