import os
import tempfile
import json
import boto3
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tiktoken import encoding_for_model

# --- Load environment variables ---
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
CORRELATION_ID = os.environ.get("CORRELATION_ID")

MAX_METADATA_BYTES = 40960
MAX_TOKENS = 16385
MODEL_NAME = "gpt-4"

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

# --- Token counter ---
def count_tokens(text):
    encoder = encoding_for_model(MODEL_NAME)
    return len(encoder.encode(text))

# --- Truncate JD and CV to fit within LLM context ---
def truncate_to_fit_limit(jd_text, cv_text, max_tokens=MAX_TOKENS, reserved_for_prompt=2000):
    encoder = encoding_for_model(MODEL_NAME)
    limit = max_tokens - reserved_for_prompt
    jd_tokens = encoder.encode(jd_text)
    cv_tokens = encoder.encode(cv_text)

    if len(jd_tokens) + len(cv_tokens) > limit:
        jd_max = int(limit * 0.6)
        cv_max = limit - jd_max
        jd_tokens = jd_tokens[:jd_max]
        cv_tokens = cv_tokens[:cv_max]

    return encoder.decode(jd_tokens), encoder.decode(cv_tokens)

# --- Metadata Extractors ---
def extract_metadata_from_jd(text):
    prompt = PromptTemplate(
        input_variables=["jd"],
        template="""
        Extract the following metadata from the job description below:
        - Role Title
        - Required Experience (years)
        - Tech Stack (as list)
        - Location

        Job Description:
        {jd}

        Return as JSON.
        """
    )
    chain = LLMChain(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), prompt=prompt)
    return json.loads(chain.run({"jd": text}))

def extract_metadata_from_cv(text):
    prompt = PromptTemplate(
        input_variables=["cv"],
        template="""
        Extract the following metadata from the CV below:
        - Total Years of Experience
        - Tech Stack (as list)
        - Past Companies
        - Current Company
        - Education (Degrees and Institutions)

        CV:
        {cv}

        Return as JSON.
        """
    )
    chain = LLMChain(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), prompt=prompt)
    return json.loads(chain.run({"cv": text}))

# --- Truncate metadata to fit byte limit ---
def truncate_metadata(metadata):
    while True:
        serialized = json.dumps(metadata)
        if len(serialized.encode("utf-8")) <= MAX_METADATA_BYTES:
            return metadata
        keys = list(metadata.keys())
        for k in keys:
            if k not in ['correlation_id', 'type']:
                del metadata[k]
                break

# --- Store Embeddings ---
def store_embeddings(correlation_id, cv_text, jd_text):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)

    cv_chunks = text_splitter.split_text(cv_text)
    jd_chunks = text_splitter.split_text(jd_text)

    cv_meta = extract_metadata_from_cv(cv_text)
    jd_meta = extract_metadata_from_jd(jd_text)

    cv_meta['type'] = 'cv'
    cv_meta['correlation_id'] = correlation_id
    jd_meta['type'] = 'jd'
    jd_meta['correlation_id'] = correlation_id

    metadatas = [truncate_metadata(cv_meta.copy()) for _ in cv_chunks] + \
                [truncate_metadata(jd_meta.copy()) for _ in jd_chunks]

    db = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
    db.add_texts(texts=cv_chunks + jd_chunks, metadatas=metadatas)

# --- Match CV with JD ---
def match_cv_with_jd(correlation_id, cv_text, jd_text):
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

    jd_local = db.similarity_search("", k=3, filter={"correlation_id": correlation_id, "type": "jd"})
    jd_context = "\n".join([r.page_content for r in jd_local]) if jd_local else jd_text

    cv_local = db.similarity_search(jd_context, k=3, filter={"correlation_id": correlation_id, "type": "cv"})
    cv_context = "\n".join([m.page_content for m in cv_local]) if cv_local else cv_text

    tokens_used = count_tokens(jd_context) + count_tokens(cv_context)
    remaining_tokens = MAX_TOKENS - tokens_used
    chunk_allowance = int(remaining_tokens * 0.5)

    global_jds, global_cvs = [], []
    if remaining_tokens > 1000:
        jds = db.similarity_search(jd_context, k=10, filter={"type": "jd"})
        for j in jds:
            if count_tokens(j.page_content) + tokens_used < chunk_allowance:
                global_jds.append(j.page_content)
                tokens_used += count_tokens(j.page_content)

        cvs = db.similarity_search(cv_context, k=10, filter={"type": "cv"})
        for c in cvs:
            if count_tokens(c.page_content) + tokens_used < chunk_allowance:
                global_cvs.append(c.page_content)
                tokens_used += count_tokens(c.page_content)

    augmented_jd = jd_context + "\n" + "\n".join(global_jds)
    augmented_cv = cv_context + "\n" + "\n".join(global_cvs)

    trimmed_jd, trimmed_cv = truncate_to_fit_limit(augmented_jd, augmented_cv)

    prompt_template = PromptTemplate(
        input_variables=["jd", "cv"],
        template="""
        Given the following job description:
        {jd}

        And the following candidate CV:
        {cv}

        Analyze and provide:
        - A match score (0-100)
        - Key areas from the CV that closely match the job description
        - Specific gaps or areas where the CV does not meet the job description

        Format your response in a clear and concise summary.
        """
    )

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"jd": trimmed_jd, "cv": trimmed_cv})
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
    result = match_cv_with_jd(CORRELATION_ID, cv_text, jd_text)
    print("Result:", result)

    result_key = f"results/{CORRELATION_ID}_result.txt"
    s3.put_object(Bucket=S3_BUCKET, Key=result_key, Body=result.encode("utf-8"))
    print(f"Stored result at s3://{S3_BUCKET}/{result_key}")
