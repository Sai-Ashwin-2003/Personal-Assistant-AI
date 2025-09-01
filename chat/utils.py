from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file_path):
    ext = file_path.split('.')[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encodincdg="utf-8")
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    return documents