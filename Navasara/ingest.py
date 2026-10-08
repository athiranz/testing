import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(BASE_DIR, "data", "nasavara_data.json")

# Dua folder penampung PDF
PDF_FOLDERS = [
    os.path.join(BASE_DIR, "data", "scientific_papers"),
    os.path.join(BASE_DIR, "data", "folklore_articles")
]

CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def load_json_documents(json_path):
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: File JSON tidak ditemukan di {json_path}")
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        sources_str = ", ".join([s.get("filename", "") for s in item.get("source_papers", [])])
        
        metadata = {
            "id": item.get("id", ""),
            "topic": item.get("topic", ""),
            "category": item.get("category", ""),
            "sub_category": item.get("sub_category", ""),
            "title": item.get("title", ""),
            "sources": sources_str
        }
        
        content_text = f"Title: {item.get('title')}\nCategory: {item.get('category')} ({item.get('sub_category')})\n\n{item.get('content')}"
        doc = Document(page_content=content_text, metadata=metadata)
        documents.append(doc)
        
    print(f"✅ Berhasil memuat {len(documents)} entri dari JSON.")
    return documents

def load_pdf_documents(folders):
    pdf_docs = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"⚠️ Warning: Folder PDF tidak ditemukan di {folder}")
            continue

        pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            file_path = os.path.join(folder, pdf_file)
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    page.metadata["source_file"] = pdf_file
                    page.metadata["folder"] = os.path.basename(folder)
                pdf_docs.extend(pages)
                print(f"📄 Memuat PDF [{os.path.basename(folder)}]: {pdf_file} ({len(pages)} hlm)")
            except Exception as e:
                print(f"❌ Gagal membaca PDF {pdf_file}: {e}")
            
    return pdf_docs

def main():
    print("🚀 Memulai Ingestion NAVASARA Data...")
    
    json_docs = load_json_documents(DATA_JSON_PATH)
    pdf_docs = load_pdf_documents(PDF_FOLDERS)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    split_pdf_docs = text_splitter.split_documents(pdf_docs) if pdf_docs else []
    all_documents = json_docs + split_pdf_docs
    
    if not all_documents:
        print("❌ Tidak ada dokumen yang siap di-ingest.")
        return

    print(f"📦 Total segmen dokumen: {len(all_documents)}")

    # Model Multilingual agar data B.Indo tetap optimal di-query dengan B.Inggris
    print("🧠 Memuat model embedding multilingual...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    print(f"💾 Menyimpan data ke Chroma DB di: '{CHROMA_PATH}'...")
    Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    
    print("🎉 Ingestion Selesai! Vector DB siap digunakan.")

if __name__ == "__main__":
    main()