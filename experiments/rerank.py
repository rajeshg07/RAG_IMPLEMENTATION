# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity


# class QueryReranker:
#     def __init__(self, persist_dir="chroma_pdf_db", model_name="all-MiniLM-L6-v2"):
#         self.persist_dir = persist_dir
#         self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
#         self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding_model)


#     def search_and_rerank(self, query, k=5):
#         retrieved_docs = self.db.similarity_search(query, k=k)
#         query_embedding = self.embedding_model.embed_query(query)
#         doc_embeddings = [self.embedding_model.embed_query(doc.page_content) for doc in retrieved_docs]
#         similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
#         reranked = sorted(zip(retrieved_docs, similarities), key=lambda x: x[1], reverse=True)
#         return reranked  
