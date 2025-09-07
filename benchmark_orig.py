from my_rag_new import SQLDocumentStore, FAISSDatabase
import pandas as pd
import re
import numpy as np
import torch

sql_store = SQLDocumentStore(db_path="CUHK/documents.db")
faiss_db = FAISSDatabase(db_type="withIndexIDMap", db_dimension=768)
# sql_store.clear_database("CUHK/documents.db") # figure out why this does not work
# Read CSV file
df = pd.read_csv('generations5.csv')

indexer = 0
for index, row in df.iterrows():
    acronym = row[0]
    description = row[1]
    question = row[2]
    description_embedding = np.fromstring(row[3][1:-1], sep=',')  # Convert CSV string to NumPy array
    question_embedding = np.fromstring(row[4][1:-1], sep=',')

    documents = [description, row[5], row[6], row[7], row[8], row[9]]
    embeddings = [
        description_embedding,
        np.fromstring(row[10][1:-1], sep=','),
        np.fromstring(row[11][1:-1], sep=','),
        np.fromstring(row[12][1:-1], sep=','),
        np.fromstring(row[13][1:-1], sep=','),
        np.fromstring(row[14][1:-1], sep=',')
    ]

    doc_ids = []
    for text in documents:
        if pd.notna(text):  # Avoid adding NaN values
            sql_store.add_document(indexer, text)
            doc_ids.append(indexer)
            indexer += 1
    embedding_tensor = torch.Tensor(embeddings)
    print(len(embedding_tensor), len(doc_ids), index)
    print(doc_ids)
    faiss_db.add_documents(embedding_tensor, doc_ids)

faiss_db.save_index("CUHK/faiss.index")


def extract_UUID(text):

    uuid_pattern = r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
    return re.findall(uuid_pattern, text)


def benchmark():
    correct = 0
    num_elements = len(df)  

    for index, row in df.iterrows():
        query = row[2]
        query_embedding = np.fromstring(row[4][1:-1], sep=',')


        distances, indices = faiss_db.search(query_embedding.reshape(1, -1), k=5)

        top = indices[0][0]  # First element of top matches
        document_ids = list(indices[0])
        document = sql_store.fetch_document(int(document_ids[0]))

        query_uuid = extract_UUID(query)
        document_uuid = extract_UUID(document)
        print(f'Query: {query} \nQuery Results:  \n\n ')
        for i in range(5):
            idx_doc = int(document_ids[i])
            print(f"Result {i+1}: {indices[0][i]} (distance: {distances[0][i]}) (document: {sql_store.fetch_document(idx_doc)} \n ")


        if query_uuid == document_uuid:
            correct += 1

    return (correct / num_elements), correct, num_elements


if __name__ == "__main__":
    benchmark_nums = benchmark()
    print("Benchmark score: " + str(benchmark_nums[0]) + "\n" + "Correct: " + str(benchmark_nums[1]) + " Trials: " + str(benchmark_nums[2]))

