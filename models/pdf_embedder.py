
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class PDFEmbedder:
    def __init__(self, pdf_path, persist_dir="chroma_pdf_db"):
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir

    def load_split_embed(self):
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load_and_split()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=self.persist_dir)
        vectordb.persist()
        return vectordb


