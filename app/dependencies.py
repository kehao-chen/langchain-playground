import os

from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_settings import BaseSettings

load_dotenv()
class Settings(BaseSettings):
    langchain_debug: bool = bool(os.getenv("LANGCHAIN_DEBUG", False))
    openai_default_model: str = os.getenv("OPENAI_DEFAULT_MODEL_NAME", "gpt-3.5-turbo")

settings = Settings()
set_debug(settings.langchain_debug)

# Load Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()


template = """Answer the question based only on the following context:

<context>
{context}
</context>

Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name=settings.openai_default_model)
output_parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def get_llm_chain():
    return prompt | llm | output_parser


def get_rag_chain():
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
