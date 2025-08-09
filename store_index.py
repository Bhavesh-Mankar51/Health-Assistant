from dotenv import load_dotenv 
import os 
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    print("PINCONE API key set in environment.")
else:
    print("Warning: PINCONE_API_KEY environment variable not found.")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("OpenAI API key set in environment.")
else:
    print("Warning: OPENAI_API_KEY environment variable not found.")



extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(minimal_docs)

embedding = download_embeddings()  


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "health-assistant"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)