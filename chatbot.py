from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Fine-tuned model
pipe = pipeline(
    "text2text-generation",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)

# Vector DB
def create_vectordb():
    loader = CSVLoader(file_path="faqs.csv")
    docs = loader.load()
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local("./faiss_index")
