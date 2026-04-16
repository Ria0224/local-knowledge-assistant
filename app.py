from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Load LLM
llm = Ollama(model="llama3")

# 2. Load data
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

# 3. Split text
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

# 4. Create embeddings
embeddings = OllamaEmbeddings()

# 5. Store in vector DB
db = FAISS.from_documents(docs, embeddings)

# 6. Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 7. Prompt (anti-hallucination)
prompt = PromptTemplate.from_template(
    """You are a strict AI assistant.

Answer ONLY from the context below.
If the answer is not present, say:
"This information is not available in the provided data."

Context:
{context}

Question:
{question}
"""
)

# 8. Format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 9. Create RAG chain (NEW METHOD — no RetrievalQA)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 10. Chat loop
print("\n🚀 Local Knowledge Assistant is running!")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        print("Goodbye 👋")
        break

    try:
        answer = rag_chain.invoke(query)
        print("\n✅ Answer:\n", answer, "\n")
        print("="*50)

    except Exception as e:
        print("Error:", str(e))