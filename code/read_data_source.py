import os
import shutil
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama 


def search_data_files(path, path_of_files = []):
    print(path)
    for file in os.scandir(path):
        if file.is_file():
            path_of_files.append(file.path)
    print(path_of_files)
    return path_of_files

def read_data_files(path_of_files):
    d_of_text = {}
    for f in path_of_files:
        text = ""
        with open(f, 'rb') as file:
            reader = PdfReader(file)

            for page in reader.pages:
                text += page.extract_text()
        d_of_text[f.split('\\')[-1].replace('.pdf','')] = text
    return d_of_text

def character_text_splitter(dict_of_pdfs):
    pdfs_split_text = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
    for key, text in dict_of_pdfs.items():
        pdfs_split_text[key] = text_splitter.split_text(text)
    return pdfs_split_text

def pdfs_chunk_dict_to_doc(pdfs_split_text):
    documents  = []
    for filename, chunks in pdfs_split_text.items():
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i
                }
            )
            documents.append(doc)
    print(f"Total documents: {len(documents)}")
    return documents

def embed_index_and_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") #all-mpnet-base-v2(sligtly better but still poor), all-MiniLM-L6-v2(bad results)
    index_path = os.path.join(os.path.dirname(__file__), "..", "database", "faiss_index")

    if os.path.exists(index_path):
        try:
            print("Loading existing index")
            db = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            # Verify dimensions match before returning
            test_vec = embeddings.embed_query("test")
            db.index.search(__import__("numpy").array([test_vec], dtype="float32"), 1)
            return db
        except Exception as e:
            print(f"Index incompatible ({e}), recreating...")
            shutil.rmtree(index_path)
    
    # Create new index
    print("Creating new index...")
    os.makedirs(index_path, exist_ok=True)
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local(index_path)
    print("Done")
    return vector_db

def add_context_to_prompt(query, vector_db, top_k=5, SCORE_THRESHOLD=0.5):
    results = vector_db.similarity_search_with_relevance_scores(query, k=top_k)
    context = ""
    filtered = []
    system_prompt = """You are a concise information assistant. 
    Answer questions directly without any preamble or reference to sources.
    Do not say 'The answer is', 'According to', or 'Based on'.
    Just provide the answer."""
    for doc, score in results:
        if score >= SCORE_THRESHOLD:
            # source = f"Source: {doc.metadata['source']}"
            # chunk_id = f"Chunk ID: {doc.metadata['chunk_id']}"
            # content = f"Content: {doc.page_content}"
            # combined = "\n\n".join([source, chunk_id, content])
            combined = doc.page_content
            filtered.append(combined)
    print(f"Filtered results: {len(filtered)}")
    if not filtered:
        return None
    else:
        context = "\n\n".join(filtered)
        prompt = f"""{system_prompt}

        Context:
        {context}

        Question: {query}
        Answer:"""
        return prompt

def response_generation(prompt):
    if not prompt:
        return "Sorry unfortunately this information isn't available."
    llm = Ollama(model="orca-mini", temperature=0.4) #tinyllm wasn't following system prompt
    response = llm.invoke(prompt)
    return response


path = r"C:\Users\prach\RAG\data"
path_of_files = search_data_files(path)
data_text_dict = read_data_files(path_of_files)
pdfs_split_text = character_text_splitter(data_text_dict)
pdfs_documents = pdfs_chunk_dict_to_doc(pdfs_split_text)
vector_db = embed_index_and_store(pdfs_documents)

query = "Why are the advantages of AWS Lambda?"
augumented_prompt = add_context_to_prompt(query, vector_db)
response = response_generation(augumented_prompt)
print(response)