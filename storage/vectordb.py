import chromadb
import pandas as pd
import ast
import os
import shutil

DB_PATH = "storage/chroma_recipes_db"

def build_chroma_db():
    # Step 1: ensure directory exists cleanly
    if os.path.exists(DB_PATH):
        try:
            client = chromadb.PersistentClient(DB_PATH)
            # Test query — this triggers a load of the index
            _ = client.list_collections()
            print("Existing Chroma DB loaded successfully.")
            return client
        except Exception as e:
            print(f"Corrupted Chroma DB detected: {e}")
            print("Rebuilding from scratch...")
            shutil.rmtree(DB_PATH)

    # Step 2: create fresh DB client
    client = chromadb.PersistentClient(DB_PATH)

    # Step 3: prepare data
    df = pd.read_csv("utils/italian_recipes_embedded.csv")
    df["embeddings"] = df["embeddings"].apply(ast.literal_eval)

    ids = df.index.astype(str).tolist()
    embeddings = df.embeddings.to_list()
    documents = df.embed_text.to_list()
    metadatas = df[["Nome", "Categoria", "Link"]].to_dict(orient="records")

    recipes_collection = client.get_or_create_collection(
        name="recipes_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # Step 4: insert in batches
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

    print("Chroma DB built successfully.")
    return client


# Usage:
client = build_chroma_db()
recipes_collection = client.get_or_create_collection("recipes_collection")
