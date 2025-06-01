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

# --- Store Embeddings ---
def store_embeddings(correlation_id, cv_text, jd_text):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    texts = [cv_text, jd_text]
    metadatas = [
        truncate_metadata({"correlation_id": correlation_id, "type": "cv"}),
        truncate_metadata({"correlation_id": correlation_id, "type": "jd"})
    ]
    db = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
    db.add_texts(texts=texts, metadatas=metadatas)

# --- Match CV with JD ---
def match_cv_with_jd(correlation_id):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

    results = db.similarity_search(query="", k=1, filter={"correlation_id": correlation_id, "type": "jd"})
    jd_text = results[0].page_content if results else ""

    matches = db.similarity_search(jd_text, k=1, filter={"correlation_id": correlation_id, "type": "cv"})
    cv_text = matches[0].page_content if matches else ""

    if not jd_text or not cv_text:
        return "Could not find matching JD or CV in the vector database."

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

def truncate_metadata(metadata):
    import json
    while True:
        serialized = json.dumps(metadata)
        if len(serialized.encode("utf-8")) <= MAX_METADATA_BYTES:
            return metadata
        # truncate fields
        if 'correlation_id' in metadata:
            metadata['correlation_id'] = metadata['correlation_id'][:36]  # UUID size
        if 'type' in metadata and len(metadata['type']) > 10:
            metadata['type'] = metadata['type'][:10]
        # remove any extra fields if needed
        keys = list(metadata.keys())
        for k in keys:
            if k not in ['correlation_id', 'type']:
                del metadata[k]
                break


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
    result = match_cv_with_jd(CORRELATION_ID)
    print("Result:", result)

    result_key = f"results/{CORRELATION_ID}_result.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=result_key, Body=result.encode("utf-8"))
    print(f"Stored result at s3://{S3_BUCKET}/{result_key}")
