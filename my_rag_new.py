import faiss
import torch
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from transformers import BertModel, BertTokenizer

class SQLDocumentStore:
    def __init__(self, db_path="documents.db"):
        """Initializes the SQLite document store."""
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Creates the documents table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def add_document(self, doc_id, text):
        """Inserts a new document into SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO documents (id, text) VALUES (?, ?)", (doc_id, text))
        conn.commit()
        conn.close()

    def update_document(self, doc_id, new_text):
        """Updates a document's text in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET text = ?, timestamp = ? WHERE id = ?", 
                       (new_text, datetime.now(), doc_id))
        conn.commit()
        conn.close()

    def delete_document(self, doc_id):
        """Deletes a document from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()

    def fetch_document(self, doc_id):
        """Fetches a document by ID from SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def list_documents(self):
        """Lists all stored documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents")
        results = cursor.fetchall()
        conn.close()
        return results


    def clear_database(self,db_path):
        '''

        Note: this does not currently work, need to add implementation 
        '''
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"DELETE FROM {table_name};")  # Delete all records
                cursor.execute(f"VACUUM;")  # Reclaim space

            conn.commit()
            conn.close()
            print("Database cleared successfully.")
        except Exception as e:
            print(f"Error: {e}")

class FAISSDatabase:
    def __init__(self, db_type="HNSW", db_dimension=768, hnsw_m=32, index_path=None):
        """
        Initializes the FAISS database.

        Args:
            db_type (str): FAISS index type ("IndexFlatL2", "HNSW").
            db_dimension (int): Embedding dimension (e.g., 768 for BERT).
            hnsw_m (int): HNSW graph neighbors.
            index_path (str, optional): Path to load FAISS index.
        """
        self.db_type = db_type
        self.db_dimension = db_dimension
        self.hnsw_m = hnsw_m
        self.index = self._initialize_index(index_path)

    def _initialize_index(self, index_path):
        """Initializes or loads a FAISS index."""
        if index_path:
            return faiss.read_index(index_path)

        if self.db_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.db_dimension)


        if self.db_type == "withIndexIDMap":
             return faiss.IndexIDMap(faiss.IndexFlatL2(self.db_dimension))
        elif self.db_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.db_dimension, self.hnsw_m)
            index.hnsw.efConstruction = 200  
            index.hnsw.efSearch = 64  
            return index
        else:
            raise NotImplementedError(f"FAISS index type {self.db_type} not supported.")

    def save_index(self, path="faiss_index.index"):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, path)
        print(f"FAISS index saved at {path}")

    def load_index(self, path="faiss_index.index"):
        """Loads the FAISS index from disk."""
        self.index = faiss.read_index(path)
        print(f"FAISS index loaded from {path}")

    def add_documents(self, embeddings, doc_ids):
        """Adds document embeddings to FAISS."""
        self.index.add_with_ids(embeddings.numpy(), np.array(doc_ids))
        print(f"Added {len(doc_ids)} documents to FAISS index.")

    def search(self, query_embedding, k=5):
        """Searches the FAISS index."""
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def update_document(self, doc_id, new_embedding):
        """Updates a document by removing and reinserting it."""
        self.remove_document(doc_id)
        self.add_documents([new_embedding], [doc_id])
        print(f"Updated document {doc_id}.")

    def remove_document(self, doc_id):
        """Rebuilds the FAISS index without the specified document."""
        ids = np.array(range(self.index.ntotal))
        mask = ids != doc_id
        filtered_ids = ids[mask]

        embeddings = np.zeros((len(filtered_ids), self.db_dimension))
        for i, idx in enumerate(filtered_ids):
            embeddings[i] = self.index.reconstruct(idx)

        self.index = self._initialize_index(None)
        self.add_documents(embeddings, filtered_ids)
        print(f"Removed document {doc_id} and rebuilt index.")


class BERTEmbedder:
    def __init__(self, model_path_or_dir="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path_or_dir)
        self.model = BertModel.from_pretrained(model_path_or_dir)

    def generate_embedding(self, documents):
        """Generates BERT embeddings."""
        encoding = self.tokenizer.batch_encode_plus(documents, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding


# ------------------ Example Usage ------------------

if __name__ == "__main__":

    sql_store = SQLDocumentStore()
    
    # Initialize SQLite Store & FAISS
#     sql_store = SQLDocumentStore()
#     faiss_db = FAISSDatabase(db_type="withIndexIDMap", db_dimension=768)
#     embedder = BERTEmbedder(model_path_or_dir='rag_models/bert_model')
# 
     # Sample Documents
    documents = {
         101: "Billionaires influence international affairs significantly.",
         102: "The war in Ukraine shifts geopolitical alliances."
     }
    # for idx, text in documents.items():
     #    print(idx, text)
      #   sql_store.add_document(idx, text)

    print(sql_store.fetch_document(102))
#     # Store in SQL and generate embeddings
     
#     doc_ids = list(documents.keys())
#     texts = list(documents.values())
#     embeddings = embedder.generate_embedding(texts)
#     # Add to FAISS
#     faiss_db.add_documents(embeddings, doc_ids)
# 
#     # Save index
#     faiss_db.save_index("withnanidmap.index")
# 
#     # Query Example
#     query = "Do billionaires have global influence?"
#     query_embedding = embedder.generate_embedding([query])
#     distances, indices = faiss_db.search(query_embedding.numpy(), k=2)
# 
#     print(f"Query Results: {indices} with distances {distances}")
# 
#     # Update a document
#     updated_text = "Billionaires impact global politics."
#     sql_store.update_document(101, updated_text)
#     updated_embedding = embedder.generate_embedding(updated_text)
#     faiss_db.update_document(101, updated_embedding)
# 
#     # Remove a document
#     sql_store.delete_document(102)
#     faiss_db.remove_document(102)
# 
