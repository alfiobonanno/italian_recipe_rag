import ollama
import chromadb

client = chromadb.PersistentClient(path="storage/chroma_recipes_db")
collection = client.get_collection("recipes_collection")

# Function to embed text using the ollama model.

def embed_query(text):
    result = ollama.embed(model = "snowflake-arctic-embed2:568m", input = text)
    return result['embeddings'][0]

#Function to retrieve relevant documents from the vector database. This function also prints the documents, to see how effective the retrieval is.
def retrieve(query, k=10, show=True):
    q_vec = embed_query(query)
    results = collection.query(query_embeddings=[q_vec], n_results=k)

    if show:
        print("\n=== Retrieved Documents ===")
        for meta in results["metadatas"][0]:
            print(meta["Nome"])

    return results

#This function formats the retrieved documents into a single string to be used as context in the prompt.

def format_context(results):
     return "\n\n---\n\n".join(results["documents"][0])

#Function to generate a response using the ollama model, given a user query. It retrieves the relevant documents, formats them as context, builds the prompt and calls the model to generate the response.
def generate_response(query):
    results = retrieve(query)
    context = format_context(results)
    prompt = build_prompt(query, context)
    response = ollama.generate(
    model="deepseek-r1:8b",
    prompt=prompt
    )
    return response["response"]


def build_prompt(query, context):
    return f"""
You are a professional Italian chef AI assistant.
The user asks questions in English. Your job is to help them cook something delicious.

You have access to Italian recipes retrieved from a recipe database. 
These recipes are written in Italian and appear below as reference material.

IMPORTANT RULES:
- Do NOT copy the original text or steps directly.
- You may use the retrieved recipes only as inspiration.
- If the user wants a recommendation, suggest the most suitable existing recipes and explain them in English.
- If the user wants a variation (e.g., healthier, gluten-free, egg-free), modify the recipe intelligently and in line with Italian Tradition.
- If the user wants an original creation, invent a completely new recipe inspired by the retrieved examples.
- Always answer in CLEAR, NATURAL English.
- Provide structured, practical cooking instructions.

USER QUESTION:
{query}

RETRIEVED RECIPE DOCUMENTS (Italian inspiration only, do not copy):
{context}

Now produce the best answer based on the user request:
- If they want a recipe: give a full recipe (title, ingredients, steps). All ingredients must be in English but use metric units for measurements.
- If they want a recommendation: suggest the most suitable existing recipes and explain them by providing recipe steps in English in a way that they are clear.
- If they want modifications: rewrite the recipe with the requested changes.
- If they want a new creation: create a brand-new dish.
- Always specify if the recipe is taken from the database or is an original creation inspired by the retrieved recipes.

Your response begins below:
"""