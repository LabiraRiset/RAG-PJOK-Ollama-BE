from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")
import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain.llms import Ollama
import logging

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Directory for saving PDF files
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Environment and API config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "ollama-labira-rag"
EMBEDDING_MODEL = "llama3.2:3b-instruct-fp16"
MODEL="llama3:latest"

MAX_METADATA_SIZE = 40960
NAMESPACE= "PJOK"
embedding_engine = OllamaEmbeddings(model=EMBEDDING_MODEL, show_progress=True, num_gpu=28)
# Initialize the generative model
generative_model = Ollama(model=MODEL, num_gpu=28)

# Initialize Pinecone vector store
def vector_db():
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone.Index(INDEX_NAME, namespace="PJOK")
    return index
    

# Clean text function
def clean_text(text):
    cleaned_text = re.sub(r'[^\x00-\x7F]+|[^a-zA-Z0-9\s\.,;?!-]', ' ', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    document_text = ""
    for page in pdf_reader.pages:
        document_text += page.extract_text()
    return document_text

# Function to check and truncate metadata size
def truncate_text(text):
    encoded_text = json.dumps({"text": text}).encode("utf-8")
    if len(encoded_text) > MAX_METADATA_SIZE:
        estimated_length = int(MAX_METADATA_SIZE / 2)
        return text[:estimated_length] + "..."
    return text

# Dynamically adjust chunk size if metadata size exceeds limit
def adjust_chunk_size(documents):
    adjusted_docs = []
    for doc in documents:
        if len(json.dumps({"text": doc.page_content}).encode("utf-8")) > MAX_METADATA_SIZE:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            adjusted_docs.extend(splitter.create_documents([doc.page_content]))
        else:
            adjusted_docs.append(doc)
    return adjusted_docs
    
def batch_embed_texts(texts, batch_size=10):
    """Embed texts in batches to optimize GPU usage."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_engine.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

# Insert documents into vector database
def insert_knowledge(vectors, texts, index, batch_size=50):
    prepped = []
    for i, em, text in zip(range(len(vectors)), vectors, texts):
        doc_id = str(uuid.uuid4())
        v = {"id": doc_id, "values": em, "metadata": {"text": truncate_text(text)}}
        prepped.append(v)
        if len(prepped) >= batch_size:
            index.upsert(prepped, namespace=NAMESPACE)
            prepped = []
    if prepped:
        index.upsert(prepped, namespace=NAMESPACE)


def get_embeddings(articles, embeddings):
   return embeddings.embed_documents(texts= articles)

@app.route('/v2/kasihlabira', methods=["POST"])
def upsert_knowledge_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "Body `file` tidak ditemukan"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File harus diinput"}), 400

    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "File harus berbentuk .pdf"}), 400

    if file:
        index = vector_db()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        pdf_content = extract_text_from_pdf(file)
        text = clean_text(pdf_content)

        # Set smaller chunk size to reduce metadata size
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=500, chunk_overlap=100)
        documents = text_splitter.create_documents([text])

        # Adjust chunk size dynamically if metadata exceeds limit
        documents = adjust_chunk_size(documents)
        
        texts = [doc.page_content for doc in documents]

        vectors = []
        for doc, i in zip(texts, range(len(texts))):
            emb = embedding_engine.embed_documents(texts=[doc])
            if emb[0] is not None:
                vectors.append(emb[0])
            else:
                print(f"document {i}, not treated first 20 chars {doc[:20]}")

        insert_knowledge(vectors, texts, index)

    return jsonify({"text": "Berhasil mempelajari data pdf " + file.filename}), 200


# Step 1: Get the embedding for the question
def get_question_embedding(question):
    question_embedding = embedding_engine.embed_documents([question])
    return question_embedding[0]  # Returns the embedding vector

# Step 2: Query Pinecone for similar documents
def retrieve_similar_docs(question_embedding, namespace="PJOK", top_k=3):
    index = vector_db()
    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True  # Fetch metadata for context
    )
    # Extract relevant texts from the results
    retrieved_texts = [match["metadata"]["text"] for match in results["matches"]]
    return retrieved_texts


def retrieve_similar_docs_with_score(question_embedding, namespace="PJOK", top_k=3):
    index = vector_db()
    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True  # Fetch metadata for context
    )
    
    # Extract relevant texts and their scores from the results
    retrieved_data = [{"text": match["metadata"]["text"]} for match in results["matches"]]
    retrieved_score = [{"text": match["metadata"]["text"], "score": match["score"]} for match in results["matches"]]
    return retrieved_data, retrieved_score
    

# Step 3: Generate a response using the Ollama model with a chat template
def generate_response_with_context(question, context_texts):
    context = "\n".join(context_texts)
    prompt = (
        f"You are a educational assistance in Indonesia Elementary School, please answer the question with indonesia language accurately and concisely and if you don't know just say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Generate the answer using the Ollama model
    response = generative_model(prompt)
    return response

    # # Set the main context from retrieved documents
    # main_context = context_texts[0] if context_texts else "No relevant information found."

    # # Define chat template with structured messages for the generative model
    # messages = [
    #     {'role': 'system', 'content': 'Use the following information to answer the question.'},
    #     {'role': 'user', 'content': main_context},
    #     {'role': 'user', 'content': question}
    # ]

    # # Generate the answer using the Ollama model without streaming
    # response = ollama.chat(
    #     model=MODEL,
    #     messages=messages,
    #     stream=False  # Set stream to False
    # )

    # # Check if the response contains content
    # response_text = response.get('content', '')  # Safely access 'content' key
    # if not response_text:
    #     logging.debug("Received an empty response or no 'content' in response.")

    # logging.debug(f"Final response text: {response_text}")
    # return response_text

def ragas_generate_response_with_context(question):
    question_embedding = get_question_embedding(question)
    with app.app_context():
        context_texts, score = retrieve_similar_docs_with_score(question_embedding)
    # Combine the context documents and the question
    context = "\n".join(context_texts)
    # prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    prompt = (
       f"""You are a helpful assistant. If the question below requires specific knowledge, use the context provided. Otherwise, answer the question directly.

        Contexts:
        {context}

        Query: {question_embedding}"""
    )

    # Generate the answer using the Ollama model
    response = generative_model(prompt)

    search_result = [{"content" : x.page_content, "score" : score} for x, score in score]

    return jsonify({"answer": response, "context": search_result[0]}), 200


   
@app.route('/v2/tanyalabira', methods=["POST"])
def get_embedding():
    body = request.get_json()
    query = body.get("question")
    indexname = body.get("index_name")
    namespace = body.get("namespace")

    if not query:
        return jsonify({'error': '[ERROR] `question` required'}), 400
    
    if not indexname:
        return jsonify({'error': '[ERROR] `index_name` required'}), 400
    
    if not namespace:
        return jsonify({'error': '[ERROR] `namespace` required'}), 400
    
    # index = vector_db()
    
    # embed = get_embeddings([query], embedding_engine)

    # res = index.query(vector=embed, top_k=3, include_metadata=True)

    # Usage Example
    question_embedding = get_question_embedding(query)
    context_texts = retrieve_similar_docs(question_embedding)
    response = generate_response_with_context(query, context_texts)

    return jsonify({"text": response}), 200



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
