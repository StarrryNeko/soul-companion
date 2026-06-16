"""Gradio interface."""

from __future__ import annotations

import gradio as gr

from config.settings import UI_CONFIG
from src.tools.emotion_logger import EmotionLoggerTool


def create_app(pipeline) -> gr.Blocks:
    """Create Gradio Blocks app."""

    emotion_tool = EmotionLoggerTool()

    def respond(message: str, history: list) -> tuple[list, str, str]:
        result = pipeline.process(message, history)
        history = history + [(message, result["response"])]
        status = f"意图：{result['intent']} | 工具：{result['tool_used'] or '无'} | 危机：{result['is_crisis']}"
        return history, "", status

    def load_history() -> list[list]:
        rows = emotion_tool.get_history()
        return [[item["timestamp"], item["emotion"], item["intensity"], item["note"]] for item in rows]

    with gr.Blocks(title=UI_CONFIG["title"]) as demo:
        gr.Markdown(f"# {UI_CONFIG['title']}")
        gr.Markdown(f"> {UI_CONFIG['disclaimer']}")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=520)
                user_input = gr.Textbox(label="输入", placeholder="说说你现在的困扰，或输入一个心理健康相关问题")
                with gr.Row():
                    send = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空")
                status = gr.Textbox(label="系统状态", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### 情绪记录")
                emotion_table = gr.Dataframe(
                    headers=["时间", "情绪", "强度", "备注"],
                    datatype=["str", "str", "number", "str"],
                    interactive=False,
                )
                refresh = gr.Button("刷新记录")
                gr.Markdown("### 快捷练习")
                breathing = gr.Button("推荐一个放松练习")

        send.click(respond, [user_input, chatbot], [chatbot, user_input, status])
        user_input.submit(respond, [user_input, chatbot], [chatbot, user_input, status])
        clear.click(lambda: ([], ""), outputs=[chatbot, status])
        refresh.click(load_history, outputs=emotion_table)
        breathing.click(lambda: "我现在很紧张，能不能推荐一个呼吸练习？", outputs=user_input)

    return demo

