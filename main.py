import gradio as gr
from storage.vectordb import build_chroma_db
from retrieval_generation import chat, reset_memory

def main():
    # Build or load the Chroma vector database
    chroma_client = build_chroma_db()
    return chroma_client

def chat_fn(message, history):
    answer = chat(message)
    return history + [[message, answer]]

with gr.Blocks() as demo:
    gr.Markdown("<h1>🍝 Italian Recipe Chef – Local RAG Assistant</h1>")
    
    chatbot = gr.Chatbot(height=450)
    user_input = gr.Textbox(
        placeholder="Ask something like: 'Give me a quick tomato pasta'...",
        label="Your Message"
    )

    clear_btn = gr.Button("Clear Chat")

    def clear_chat():
        reset_memory()
        return []

    user_input.submit(chat_fn, [user_input, chatbot], chatbot)
    clear_btn.click(clear_chat, None, chatbot)

if __name__ == "__main__":
    main()
    demo.launch()