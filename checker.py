import os
import re
import time
import pdfplumber
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain.schema import Document

from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found in environment"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

@dataclass
class ProcessingStats:
    start_time: float
    pdf_pages: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    requirements_processed: int = 0

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

class AdvancedFinancialRAG:
    def __init__(self, use_gpu: bool = True):
        self.stats = ProcessingStats(start_time=time.time())
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'

        print(f"🚀 Initializing AdvancedFinancialRAG on {self.device}")

        print("🔄 Loading embedding model...")
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
        )
        print("✅ Embedding model ready")

        print("🔄 Loading reranker model...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)
        print("✅ Reranker model ready")

        self.vector_store = None


def load_and_prepare_checklist(filepath: str) -> List[Dict]:
    df = pd.read_excel(filepath, sheet_name=0, header=None).fillna("")
    df.columns = ['A', 'B', 'C'] + list(df.columns[3:])
    structured = []
    current_category = current_main = None
    main_id_counter = 0

    for idx, row in df.iterrows():
        text = row['A'].strip()
        if not text or text.upper() in ['PRESENTATIE', 'JAARVERSLAGGEVING WAARDERINGSGRONDSLAGEN', 'N/A']:
            continue
        if len(text.split()) <= 3 and text.isupper():
            current_category = text
            continue
        if text.endswith(('.', ':')) or len(text.split()) > 6:
            current_main = {
                'id': f"main_{main_id_counter}",
                'main_condition': text,
                'category': current_category,
                'sub_conditions': [],
                'row_index': idx
            }
            structured.append(current_main)
            main_id_counter += 1
            continue
        if text.startswith(('-', '•', '→')) and current_main:
            current_main['sub_conditions'].append({
                'id': f"{current_main['id']}_sub",
                'text': re.sub(r'^[-•→\s]+', '', text),
                'main_id': current_main['id'],
                'main_condition': current_main['main_condition'],
                'category': current_category,
                'row_index': idx
            })

    requirements = []
    for item in structured:
        requirements.append({**item, 'text': item['main_condition'], 'is_main': True})
        for sub in item['sub_conditions']:
            requirements.append({**sub, 'is_main': False})
    return requirements[:2]  # limit to 2 for testing

def process_financial_pdf(pdf_path: str) -> List[Document]:
    with pdfplumber.open(pdf_path) as pdf:
        pages = [{
            'page': i + 1,
            'content': page.extract_text() or "",
            'section': "unknown"
        } for i, page in enumerate(pdf.pages)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    documents = []
    for page in pages:
        for chunk in splitter.split_text(page['content']):
            documents.append(Document(
                page_content=chunk,
                metadata={'page': page['page'], 'section': page['section'], 'source': f"Page {page['page']}"}
            ))
    return documents

def build_vector_store(documents: List[Document], embedder) -> FAISS:
    return FAISS.from_documents(documents, embedder)

def retrieve_evidence(requirement: str, vector_store: FAISS, reranker, top_k=5) -> List[Tuple[Document, float]]:
    docs = vector_store.similarity_search(requirement, k=20)
    if not docs:
        return []
    scores = reranker.predict([(requirement, d.page_content) for d in docs], batch_size=8)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]

def check_compliance(requirement: str, evidence: List[Tuple[Document, float]]) -> Dict:
    if not evidence:
        return {"compliance": "Not found", "pages": []}

    context = "\n\n".join(f"[Page {d.metadata['page']} Score {s:.2f}]\n{d.page_content}"[:3000] for d, s in evidence)
    prompt = f"""You are a financial auditor. Evaluate the following requirement against the provided report context.

REQUIREMENT:
{requirement}

CONTEXT:
{context}

Reply in format:
Compliance: Found/Not found
Pages: [comma-separated page numbers]
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01,
            max_tokens=256
        ).choices[0].message.content
        match = re.search(r'Compliance: *(Found|Not found)', response, re.IGNORECASE)
        pages = re.findall(r'\d+', response)
        return {
            "compliance": match.group(1).capitalize() if match else "Error",
            "pages": sorted(set(int(p) for p in pages))[:5]
        }
    except Exception as e:
        return {"compliance": "Error", "pages": []}

def main():
    checklist = load_and_prepare_checklist("checklist.xlsx")
    documents = process_financial_pdf("financial.pdf")
    rag = AdvancedFinancialRAG()
    rag.vector_store = build_vector_store(documents, rag.embedder)

    results = []
    for req in checklist:
        query = req['text'] if req['is_main'] else f"{req['main_condition']}: {req['text']}"
        evidence = retrieve_evidence(query, rag.vector_store, rag.reranker)
        compliance = check_compliance(query, evidence)
        results.append({**req, **compliance})

    df = pd.DataFrame(results)
    df.to_excel("groq_compliance_results.xlsx", index=False)
    print("✅ Report saved to groq_compliance_results.xlsx")

if __name__ == '__main__':
    main()
