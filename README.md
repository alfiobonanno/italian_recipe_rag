# 🍝 Italian Recipe Chef - Local RAG Assistant

An intelligent conversational assistant powered by Retrieval-Augmented Generation (RAG) that helps you discover, modify, and create authentic Italian recipes. Built with LangChain, local LLMs using Ollama, ChromaDB vector storage, and a Gradio interface.

## 📋 Overview

This project demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline that:
- Retrieves relevant Italian recipes from a vector database based on natural language queries
- Uses local LLM models (DeepSeek-R1) to generate intelligent, contextual responses
- Provides recipe recommendations, modifications, and original creations inspired by traditional Italian cuisine
- Runs entirely locally with no external API dependencies

## ✨ Features

- **🔍 Intelligent Recipe Search**: Natural language queries retrieve the most relevant recipes from a database of Italian dishes
- **🤖 AI-Powered Chef Assistant**: Generate detailed cooking instructions in clear English while respecting Italian culinary traditions
- **🎨 Recipe Customization**: Request modifications (healthier versions, dietary restrictions, ingredient substitutions)
- **💡 Creative Inspiration**: Generate original recipes inspired by the database of authentic Italian dishes
- **💬 Conversation Memory**: Maintains chat history for context-aware multi-turn conversations
- **🌐 Gradio Web Interface**: User-friendly chat interface for seamless interaction
- **🔗 LangChain Integration**: Leverages LangChain for modular RAG pipeline and easy model swapping
- **🏠 Fully Local**: No API keys required - runs entirely on your machine using Ollama

## 🏗️ Architecture

### RAG Pipeline
1. **Data Preparation** (`utils/data_cleaning.ipynb`)
   - Load and clean Italian recipe dataset
   - Format ingredients and cooking steps
   - Generate embeddings using Snowflake Arctic Embed model

2. **Vector Storage** (`storage/vectordb.py`)
   - Initialize ChromaDB persistent client
   - Store recipe embeddings with metadata (name, category, link)
   - Enable semantic search capabilities

3. **Query & Generation** (`retrieval_generation.py`)
   - LangChain-based RAG pipeline with modular components
   - Embed user queries using OllamaEmbeddings
   - Retrieve top-k relevant recipes via cosine similarity
   - Generate responses using DeepSeek-R1 with context-aware prompts
   - Conversation memory for multi-turn dialogues

4. **User Interface** (`main.py`)
   - Gradio-based chat interface
   - Real-time response generation
   - Conversation history display and management
   - Clear chat functionality with memory reset

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- 8GB+ RAM recommended

### Quick Start (Ready to Use!)

The project comes **pre-configured with the vector database** and is ready to run immediately!

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd italian_recipes
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Required packages:
   - `gradio` - Web interface
   - `ollama` - Local LLM inference
   - `chromadb` - Vector database
   - `pandas` - Data manipulation
   - `tqdm` - Progress bars
   - `langchain` - RAG framework
   - `langchain-community` - Community integrations
   - `langchain-core` - Core LangChain components

3. **Install Ollama and pull models**
   - Download and install [Ollama](https://ollama.ai/)
   - Pull the required models:
   ```bash
   ollama pull snowflake-arctic-embed2:568m
   ollama pull deepseek-r1:8b
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The Gradio interface will open in your browser at `http://localhost:7860`

**System Requirements:**
- 8GB+ RAM recommended
- GPU optional (CPU works but slower)
- ~10GB disk space for models

---

### Optional: Rebuild Vector Database from Scratch

If you want to understand the full pipeline or modify the dataset, follow these steps:

1. **Prepare the data**
   
   Run the data preparation notebook to generate embeddings:
   ```bash
   jupyter notebook utils/data_cleaning.ipynb
   ```
   
   This will:
   - Clean the raw recipe data
   - Format ingredients and steps
   - Generate embeddings using Ollama
   - Save `italian_recipes_embedded.csv`

2. **Initialize the vector database**
   ```bash
   python storage/vectordb.py
   ```
   
   This creates the ChromaDB collection and loads all recipe embeddings.

**Note:** These steps are only needed if you want to rebuild the database or add new recipes. The included database is ready to use!

### Usage Examples

Once the app is running, try these queries:
- "Give me a quick tomato pasta recipe"
- "I want a healthier version of carbonara"
- "Suggest a recipe with mushrooms and cream"
- "How do I make authentic pizza dough?"
- "Create an original dish with eggplant and mozzarella"

## 📁 Project Structure

```
italian_recipes/
├── main.py                       # Gradio web interface
├── retrieval_generation.py       # RAG logic & LLM generation
├── storage/
│   ├── vectordb.py               # Vector database initialization
│   └── chroma_recipes_db/        # ChromaDB storage (included)
├── utils/                        # Data preparation & datasets
│   ├── data_cleaning.ipynb       # Embedding generation pipeline
│   ├── italian_recipes.csv       # Raw recipe data
│   └── italian_recipes_embedded.csv  # Recipes with embeddings
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Technical Details

### LangChain RAG Pipeline
The project uses **LangChain** to build a modular RAG pipeline:
- **Retriever**: LangChain wrapper around ChromaDB for semantic search
- **Prompt Template**: Multi-message template with system instructions, context, and chat history
- **Runnable Chain**: Composable pipeline connecting retrieval → prompt → LLM → output parsing
- **Memory**: ChatMessageHistory for maintaining conversation context

### Models Used
- **Embedding Model**: `snowflake-arctic-embed2:568m` (568M parameters)
  - Generates 768-dimensional embeddings
  - Optimized for semantic similarity search
  - Integrated via `OllamaEmbeddings`
  
- **Language Model**: `deepseek-r1:8b` (8B parameters)
  - Reasoning-focused architecture
  - Generates contextual, creative responses
  - Integrated via LangChain's `Ollama` wrapper

### Vector Database
- **ChromaDB** with cosine similarity
- Stores ~5000+ Italian recipes with embeddings
- Metadata: recipe name, category, source link
- LangChain integration for seamless retrieval

### Prompt Engineering
The system uses a carefully crafted ChatPromptTemplate that:
- Instructs the LLM to act as a professional Italian chef
- Prevents direct copying while allowing inspiration
- Includes retrieved context from the vector database
- Maintains chat history for contextual conversations
- Handles multiple request types (recommendations, modifications, creations)
- Ensures responses are in clear English with metric measurements

## 🎯 Use Cases

- **Home Cooks**: Discover new Italian recipes and learn authentic techniques
- **Dietary Needs**: Adapt recipes for gluten-free, vegan, or other restrictions
- **Culinary Students**: Understand Italian cooking traditions and ingredient combinations
- **Food Bloggers**: Generate creative recipe variations for content creation

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more recipes to the database
- Improve the prompt engineering
- Enhance the UI/UX
- Optimize the RAG retrieval pipeline

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Italian recipe dataset source: https://www.kaggle.com/datasets/edoardoscarpaci/italian-food-recipes 
- LangChain for the RAG framework and modular pipeline architecture
- Ollama for local LLM infrastructure
- ChromaDB for vector storage capabilities
- Gradio for the intuitive web interface

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a portfolio project demonstrating RAG implementation with local LLMs. Recipe accuracy and culinary authenticity may vary - always verify recipes before cooking!
