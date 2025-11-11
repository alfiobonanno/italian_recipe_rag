import chromadb
import pandas as pd
import ast

# Creation of persistent ChromaDB client
client = chromadb.PersistentClient("storage/chroma_recipes_db")

## Creation or retrieval of collection

recipes_collection = client.get_or_create_collection(
    name="recipes_collection",
    metadata={"hnsw:space": "cosine"}
)

#Preparing data for insertion
df = pd.read_csv("utils/italian_recipes_embedded.csv")
ids = df.index.astype(str).tolist()
df["embeddings"] = df["embeddings"].apply(ast.literal_eval)
embeddings = df.embeddings.to_list()
documents = df.embed_text.to_list()
metadatas = df[["Nome", "Categoria","Link"]].to_dict(orient="records")

# Inserting data into the collection

BATCH_SIZE = 1000
total = len(ids)

for i in range(0, total, BATCH_SIZE):
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]
    batch_documents = documents[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]

    recipes_collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )

    print(f"Inserted {min(i+BATCH_SIZE, total)} / {total}")



