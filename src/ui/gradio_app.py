"""Gradio interface."""

from __future__ import annotations

import gradio as gr

from config.settings import UI_CONFIG
from src.tools.emotion_logger import EmotionLoggerTool


APP_CSS = """
:root {
    --leaf-50: #f3fbf1;
    --leaf-100: #e2f4dc;
    --leaf-200: #c6e8bd;
    --leaf-500: #55a667;
    --leaf-700: #2f6f43;
    --mint: #d9f2e2;
    --sage: #edf7e8;
    --sun: #fff0bf;
    --coral: #ffd8c7;
    --sky: #dff1ff;
    --ink: #1e3528;
}

body,
.gradio-container {
    background:
        radial-gradient(circle at 18% 8%, rgba(255, 240, 191, 0.55), transparent 28%),
        radial-gradient(circle at 82% 16%, rgba(217, 242, 226, 0.85), transparent 32%),
        linear-gradient(135deg, #f2faef 0%, #e4f4df 42%, #f8fff5 100%) !important;
    color: var(--ink);
}

.gradio-container {
    min-height: 100vh;
    font-family: "Inter", "Microsoft YaHei", "PingFang SC", sans-serif;
}

#main-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 18px 34px;
}

#app-title {
    padding: 20px 24px 10px;
}

#app-title h1 {
    margin-bottom: 8px;
    color: #214b31;
}

#app-title blockquote {
    border-left: 4px solid var(--leaf-500);
    background: rgba(255, 255, 255, 0.58);
    border-radius: 8px;
    color: #42624c;
    padding: 12px 16px;
}

.panel {
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(85, 166, 103, 0.22);
    border-radius: 8px;
    box-shadow: 0 18px 46px rgba(47, 111, 67, 0.12);
    padding: 16px;
}

.chat-panel {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(243, 251, 241, 0.82));
}

.side-panel {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(237, 247, 232, 0.9)),
        linear-gradient(90deg, rgba(255, 216, 199, 0.25), rgba(223, 241, 255, 0.24));
}

.side-panel h3 {
    color: #2f6f43;
}

.gradio-container .chatbot {
    border: 1px solid rgba(85, 166, 103, 0.22);
    border-radius: 8px;
    background: linear-gradient(180deg, #fbfffa, #eef8ec);
}

.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap {
    border-color: rgba(85, 166, 103, 0.28) !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: var(--leaf-500) !important;
    box-shadow: 0 0 0 3px rgba(85, 166, 103, 0.18) !important;
}

.gradio-container button.primary {
    background: linear-gradient(135deg, #4f9d5d, #2f7b4b) !important;
    border: 0 !important;
    color: #ffffff !important;
}

.gradio-container button.secondary {
    background: #f9fff7 !important;
    border-color: rgba(85, 166, 103, 0.32) !important;
    color: #2f6f43 !important;
}

.gradio-container button:hover {
    filter: brightness(0.98);
    transform: translateY(-1px);
    transition: all 140ms ease;
}

.accent-refresh button {
    background: var(--sky) !important;
    color: #24536c !important;
    border-color: rgba(83, 142, 177, 0.26) !important;
}

.accent-breathing button {
    background: var(--sun) !important;
    color: #735c10 !important;
    border-color: rgba(192, 151, 39, 0.26) !important;
}
"""


def create_app(pipeline) -> gr.Blocks:
    """Create Gradio Blocks app."""

    emotion_tool = EmotionLoggerTool()

    def respond(message: str, history: list) -> tuple[list, str, str]:
        result = pipeline.process(message, history)
        history = history + [(message, result["response"])]
        status = (
            f"意图：{result['intent']} | "
            f"工具：{result['tool_used'] or '无'} | "
            f"危机：{result['is_crisis']}"
        )
        return history, "", status

    def load_history() -> list[list]:
        rows = emotion_tool.get_history()
        return [[item["timestamp"], item["emotion"], item["intensity"], item["note"]] for item in rows]

    theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
        radius_size="md",
    )

    with gr.Blocks(title=UI_CONFIG["title"], theme=theme, css=APP_CSS) as demo:
        with gr.Column(elem_id="main-shell"):
            gr.Markdown(
                f"# {UI_CONFIG['title']}\n\n> {UI_CONFIG['disclaimer']}",
                elem_id="app-title",
            )
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["panel", "chat-panel"]):
                    chatbot = gr.Chatbot(height=520)
                    user_input = gr.Textbox(
                        label="输入",
                        placeholder="说说你现在的困扰，或输入一个心理健康相关问题",
                    )
                    with gr.Row():
                        send = gr.Button("发送", variant="primary")
                        clear = gr.Button("清空")
                    status = gr.Textbox(label="系统状态", interactive=False)
                with gr.Column(scale=2, elem_classes=["panel", "side-panel"]):
                    gr.Markdown("### 情绪记录")
                    emotion_table = gr.Dataframe(
                        headers=["时间", "情绪", "强度", "备注"],
                        datatype=["str", "str", "number", "str"],
                        interactive=False,
                    )
                    refresh = gr.Button("刷新记录", elem_classes=["accent-refresh"])
                    gr.Markdown("### 快捷练习")
                    breathing = gr.Button("推荐一个放松练习", elem_classes=["accent-breathing"])

        send.click(respond, [user_input, chatbot], [chatbot, user_input, status])
        user_input.submit(respond, [user_input, chatbot], [chatbot, user_input, status])
        clear.click(lambda: ([], ""), outputs=[chatbot, status])
        refresh.click(load_history, outputs=emotion_table)
        breathing.click(lambda: "我现在很紧张，能不能推荐一个呼吸练习？", outputs=user_input)

    return demo
