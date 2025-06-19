from langchain_ollama import OllamaEmbeddings #
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import sqlite3

# conn = sqlite3.connect('nutrition.db')
# df = pd.read_sql_query("SELECT name, calories, protein, carbs, fat FROM foods", conn)
# conn.close()

df = pd.read_csv("FOOD-DATA-GROUP1.csv")
df = df.astype(str)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./nutrition_db_2"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            #page_content=f"{row['food']} (Calories: {row['Caloric Value']}, Protein: {row['Protein']}g)",
            page_content=row["food"] + " " + row["Caloric Value"] + " " + row['Protein'],
            metadata={"Fat": row["Fat"], 
                      "Sugars": row["Sugars"],
                      "Carbohydrates": row["Carbohydrates"],
                      "Cholesterol": row["Cholesterol"],
                      "Protein": row["Protein"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="nutrition_1",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} #look up 5 relevant docs
)