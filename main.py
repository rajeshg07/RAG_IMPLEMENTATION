
from models.pdf_embedder import PDFEmbedder
from services.retriever import Retriever

from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import ollama
import os

def get_user_input():
    print("\nChoose generation style:")
    print("1. One sentence")
    print("2. Paragraph")
    choice = input("Enter 1 or 2: ").strip()
    return choice

def generate_answer(chain, docs, query, style="sentence"):
    if not docs:
        return "No relevant documents found."

    prompt = f"Answer in {'one sentence' if style == 'sentence' else 'a paragraph'}: {query}"
    result = chain.run(input_documents=docs, question=prompt)
    return result

def main():
    pdf_path = "/Users/user/Documents/Veel Project/RAG_IMPLEMENTATION/attention.pdf"
    loader = PDFEmbedder(pdf_path)
    loader.load_split_embed()
    retriever = Retriever()

    query = input("\nEnter your question: ")

    docs = retriever.retrieve(query, show_chunks=True)

    choice = get_user_input()
    style = "sentence" if choice == "1" else "paragraph"

    llm = ollama.Ollama(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")

    answer = generate_answer(chain, docs, query, style)
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()




