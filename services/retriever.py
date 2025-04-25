from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, db_dir="chroma_pdf_db"):
        # Initialize the embedding model for query and documents
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectordb = Chroma(persist_directory=db_dir, embedding_function=self.embedding_model)

    def retrieve(self, query, top_k=5, show_chunks=False):
        # Perform initial similarity search
        docs = self.vectordb.similarity_search(query, k=top_k)

        if show_chunks:
            print("\nRetrieved Chunks (Before Reranking):")
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Chunk #{i} ---\n{doc.page_content.strip()[:500]}...")

        if not docs:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings = [self.embedding_model.embed_query(doc.page_content) for doc in docs]
        scores = cosine_similarity([query_embedding], doc_embeddings)[0]
        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        if show_chunks:
            print("\nReranked Chunks (Top Ranked First):")
            for i, (doc, score) in enumerate(reranked, 1):
                print(f"\n--- Reranked #{i} | Score: {score:.4f} ---\n{doc.page_content.strip()[:500]}...")

        return [doc for doc, _ in reranked]
    


