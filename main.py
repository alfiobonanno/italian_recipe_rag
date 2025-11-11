import gradio as gr
from retrieval_generation import *

def chat_fn(message, history):
    history = history or []
    history.append((message, ""))

    # stream tokens
    partial = ""
    for chunk in generate_response(message):
        partial += chunk
        history[-1] = (message, partial)
        yield history

with gr.Blocks() as demo:
    gr.Markdown("<h1>🍝 Italian Recipe Chef – Local RAG Assistant</h1>")
    
    chatbot = gr.Chatbot(height=450)
    user_input = gr.Textbox(
        placeholder="Ask something like: 'Give me a quick tomato pasta'...",
        label="Your Message"
    )

    clear_btn = gr.Button("Clear Chat")

    user_input.submit(chat_fn, [user_input, chatbot], chatbot)
    clear_btn.click(lambda: [], None, chatbot)

demo.launch()
