import os
import tempfile
import boto3
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import docx2txt
import json
import zlib
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load environment variables ---
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
CORRELATION_ID = os.environ.get("CORRELATION_ID")

MAX_METADATA_BYTES = 40960

# --- Init Pinecone and Index ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host='https://cv-matching-index-xihp51s.svc.aped-4627-b74a.pinecone.io')

# --- Init S3 client ---
s3 = boto3.client("s3")

# --- Utility to download S3 object ---
def download_s3_file(key):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        print(f"Downloading from S3 key: {key}")
        s3.download_fileobj(S3_BUCKET, key, tmp)
        tmp.flush()
        return tmp.name

# --- Read CV or JD text from file ---
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

# --- Truncate metadata with compression ---
def truncate_metadata(metadata):
    allowed_keys = {'correlation_id', 'type', 'chunk'}
    metadata = {k: v for k, v in metadata.items() if k in allowed_keys}

    compressed = zlib.compress(json.dumps(metadata).encode('utf-8'))
    if len(compressed) <= MAX_METADATA_BYTES:
        return metadata

    if 'correlation_id' in metadata:
        metadata['correlation_id'] = metadata['correlation_id'][:36]
    if 'type' in metadata:
        metadata['type'] = metadata['type'][:10]
    if 'chunk' in metadata:
        metadata['chunk'] = int(metadata['chunk'])

    return metadata

# --- Store Embeddings with chunking ---
def store_embeddings(correlation_id, cv_text, jd_text):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    cv_chunks = splitter.split_text(cv_text)
    jd_chunks = splitter.split_text(jd_text)

    texts = cv_chunks + jd_chunks
    metadatas = [
        truncate_metadata({"correlation_id": correlation_id, "type": "cv", "chunk": i}) for i in range(len(cv_chunks))
    ] + [
        truncate_metadata({"correlation_id": correlation_id, "type": "jd", "chunk": i}) for i in range(len(jd_chunks))
    ]

    db = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
    db.add_texts(texts=texts, metadatas=metadatas)

# --- Match CV with JD ---
def match_cv_with_jd(correlation_id, fallback_jd_text, fallback_cv_text):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

    results = db.similarity_search(query="", k=1, filter={"correlation_id": correlation_id, "type": "jd"})
    jd_text = results[0].page_content if results else fallback_jd_text

    matches = db.similarity_search(jd_text, k=1, filter={"correlation_id": correlation_id, "type": "cv"})
    cv_text = matches[0].page_content if matches else fallback_cv_text

    prompt_template = PromptTemplate(
        input_variables=["jd", "cv"],
        template="""
        Given the following job description:
        {jd}

        And the following candidate CV:
        {cv}

        Provide a match score (0-100) and a short justification for the match.
        """
    )

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"jd": jd_text, "cv": cv_text})
    return result

# --- Main Execution ---
if __name__ == "__main__":
    for ext in ["pdf", "docx"]:
        potential_key = f"uploads/{CORRELATION_ID}_cv.{ext}"
        try:
            cv_path = download_s3_file(potential_key)
            break
        except Exception:
            cv_path = None

    if not cv_path:
        raise Exception("CV file not found with supported extensions (.pdf or .docx)")

    jd_key = f"uploads/{CORRELATION_ID}_job-description.txt"
    jd_path = download_s3_file(jd_key)

    cv_text = extract_text_from_file(cv_path)
    jd_text = extract_text_from_file(jd_path)

    print("Storing embeddings...")
    store_embeddings(CORRELATION_ID, cv_text, jd_text)

    print("Matching CV and JD...")
    result = match_cv_with_jd(CORRELATION_ID, jd_text, cv_text)
    print("Result:", result)

    result_key = f"results/{CORRELATION_ID}_result.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=result_key, Body=result.encode("utf-8"))
    print(f"Stored result at s3://{S3_BUCKET}/{result_key}")
