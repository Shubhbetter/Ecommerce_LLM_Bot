from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import pipeline

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load fine-tuned model
pipe = pipeline(
    "text2text-generation",
    model="./models/fine_tuned_model",
    tokenizer="./models/fine_tuned_model",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)

# Create Vector DB
def create_vectordb():
    loader = CSVLoader(file_path="./data/faqs.csv")
    docs = loader.load()
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local("./faiss_index")

# Get response from bot
def get_response(query):
    vectordb = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based only on this context.
    If the answer is not found, reply with "I don't know."

    CONTEXT: {context}
    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    response = chain.invoke(
        {"input_documents": retriever.get_relevant_documents(query), "question": query},
        return_only_outputs=True
    )["output_text"]

    return response
