import os
from dotenv import load_dotenv
import time
import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core._api.deprecation import suppress_langchain_deprecation_warning
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
import whisper, torch
import glob, sys, signal
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import shutil
import uuid
import uvicorn

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from constants import CHROMA_SETTINGS


app = FastAPI()
load_dotenv()
UPLOAD_DIR = "media_uploads"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
whispermodel = str(os.environ.get('WHISPER_MODEL'))
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
llmapiport = int(os.environ.get('LLMAPI_PORT'))
llmapihost = str(os.environ.get('LLMAPI_HOST'))
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
chunk_size = 500
chunk_overlap = 50
chat_history = []

class Parameters(BaseModel):
    question: str

model = whisper.load_model(whispermodel).to(device)

def signal_handler(sig, frame):
    print("Received SIGINT. Stopping LLMAPI...")
    sys.exit(0)

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def ingest():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

# try:
#     with suppress_langchain_deprecation_warning():
#         embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
#         args = parse_arguments()
#         print("VectorStores exists! loading the model..")
#         db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#         retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
#         callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
#         llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
#         qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)
# except Exception as e:
#     print("Error loading model:", e)

@app.post('/ask/')
async def ask_question(params: Parameters):
    try:
        if not params.question:
            raise HTTPException(status_code=400, detail="Question not provided")
    except ValidationError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    with suppress_langchain_deprecation_warning():
        try:
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
            args = parse_arguments()
            print("VectorStores exists! loading the model..")
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
            callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)
        except Exception as e:
            print("Error loading model:", e)
        start = time.time()
        res = qa(params.question, chat_history)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()
        time_taken = f"Answer (took {round(end - start, 2)} s.)"
        for document in docs:
            document_metadata = document.metadata["source"]
            document_content = document.page_content
    chat_history.append((params.question, answer))
    return {
        "answer": answer,
        "document-used": document_metadata,
        "time-taken": time_taken
    }

@app.post("/upload_and_ingest/")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        elif file.filename.endswith(".mp3"):
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            textfile = os.path.join(source_directory, str(uuid.uuid4()) + ".txt")
            with open(file_path, "wb") as f:
                f.write(await file.read())
            time.sleep(4)
            data = model.transcribe(file_path)
            with open(textfile, "w") as f:
                f.write(data["text"])
            ingest()
            # os.remove(textfile)
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        elif file.filename.endswith(".txt") or file.filename.endswith(".pdf"):
            if not os.path.exists(source_directory):
                os.makedirs(source_directory)
            textfile = os.path.join(source_directory, file.filename)
            with open(textfile, "wb") as f:
                f.write(await file.read())
            ingest()
            # os.remove(textfile)
        else:
            raise HTTPException(status_code=400, detail="Only MP3 and TXT files are allowed for upload.")
        return {"message": f"{file.filename} uploaded and ingested successfully."}
    except Exception as e:
        print("Error uploading file:", e)
        raise HTTPException(status_code=500, detail=f"Failed to upload file {e}")

@app.post("/clear_source_docs/")
def clear_source_docs():
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return {"message": "Source documents cleared successfully."}

@app.post("/getinfo/")
def get_info():
    source_docs_files = os.listdir(source_directory)
    model_files = os.listdir("models")
    return {"source_docs_files": source_docs_files, "model_files": model_files}

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    if not os.path.exists("db") or len(os.listdir("db")) == 0:
        print(f"DB does not exist, creating it by ingesting the documents available in {source_directory}..")
        ingest()
        uvicorn.run(app, host=llmapihost, port=llmapiport)
    elif not os.path.exists("models") or len(os.listdir("models")) == 0:
        print("No model files found, please create a folder named models in this directory, download the models from huggingface.co and place them in the models directory, then try running the API again!")
        exit(1)
    else:
        uvicorn.run(app, host=llmapihost, port=llmapiport)