# Updated AdvancedFinancialRAG system with all suggested improvements

import os
import re
import time
import pdfplumber
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from langchain.docstore.in_memory import InMemoryDocstore
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "GROQ_API_KEY not found in environment"

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

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

        print(f"\n🚀 Initializing AdvancedFinancialRAG on {self.device}")

        print("🔄 Loading embedding model...")
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": 16, "normalize_embeddings": True}
        )
        print("✅ Embedding model ready")

        print("🔄 Loading reranker model...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)
        print("✅ Reranker model ready")

        self.vector_store = None

    def build_optimized_vector_store(self, documents: List[Document]) -> FAISS:
        print("🔍 Building optimized vector store...")
        start_time = time.time()

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embedder.embed_documents(texts)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]

        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128

        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        docstore = InMemoryDocstore({str(i): documents[i] for i in range(len(documents))})
        index_to_id = {i: str(i) for i in range(len(documents))}

        vector_store = FAISS(
            embedding_function=self.embedder.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_id,
        )

        print(f"✅ Vector store built with {len(documents)} documents in {time.time() - start_time:.2f}s")
        return vector_store

def load_and_prepare_checklist(filepath: str) -> List[Dict]:
    df = pd.read_excel(filepath, sheet_name=0, header=None).fillna("")
    df.columns = ['A', 'B', 'C'] + list(df.columns[3:])
    structured = []
    current_category = current_main = None
    main_id_counter = 0

    print("📋 Processing checklist...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['A']).strip()
        if not text or text.upper() in ['PRESENTATIE', 'JAARVERSLAGGEVING WAARDERINGSGRONDSLAGEN', 'N/A']:
            continue

        if len(text.split()) <= 3 and (text.isupper() or text.startswith('**')):
            current_category = re.sub(r'[\*\:]+', '', text).strip()
            continue

        if text.endswith(('.', ':')):
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

        if any(text.startswith(c) for c in ['-', '•', 'a.', 'b.', 'c.', 'd.', 'e.', 'f.']) and current_main:
            cleaned_text = re.sub(r'^[-•→a-f]\s*\.?\s*', '', text).strip()
            current_main['sub_conditions'].append({
                'id': f"{current_main['id']}_sub",
                'text': cleaned_text,
                'main_id': current_main['id'],
                'main_condition': current_main['main_condition'],
                'category': current_category,
                'row_index': idx
            })

    requirements = []
    for item in structured:
        requirements.append({
            'id': item['id'],
            'text': item['main_condition'],
            'category': item['category'],
            'is_main': True,
            'row_index': item['row_index']
        })
        for sub in item['sub_conditions']:
            requirements.append({
                'id': sub['id'],
                'text': sub['text'],
                'main_id': sub['main_id'],
                'main_condition': sub['main_condition'],
                'category': sub['category'],
                'is_main': False,
                'row_index': sub['row_index']
            })

    print(f"✅ Processed {len(requirements)} requirements")
    return requirements

def process_financial_pdf(pdf_path: str) -> List[Document]:
    print(f"📄 Processing PDF: {pdf_path}")
    start_time = time.time()
    section_patterns = {
        'notes': r'toelichting|notes|verklaring',
        'accounting_policies': r'waardering|valuation|grondslagen',
        'balance_sheet': r'balans|balance',
        'income_statement': r'resultaten|overzicht|baten|lasten',
        'audit_report': r'controleverklaring|audit',
        'risk_management': r'risico|risicobeheer'
    }

    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            section = "other"
            for sec_name, pattern in section_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    section = sec_name
                    break

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", "(?<=\. )", "; ", ", ", " "]
            )
            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        'page': i + 1,
                        'section': section,
                        'source': f"Page {i+1}"
                    }
                ))
    print(f"✅ Created {len(documents)} chunks in {time.time() - start_time:.2f}s")
    return documents

def generate_queries(requirement: str) -> List[str]:
    return [
        requirement,
        requirement.lower(),
        requirement.replace("waardering", "valuation"),
        f"Controle of: {requirement}"
    ]

def retrieve_evidence(requirement: str, vector_store: FAISS, reranker, top_k=3) -> List[Tuple[Document, float]]:
    docs = []
    for q in generate_queries(requirement):
        docs.extend(vector_store.similarity_search(q, k=5))
    unique_docs = list({d.page_content: d for d in docs}.values())
    pairs = [(requirement, d.page_content) for d in unique_docs]
    scores = reranker.predict(pairs, batch_size=16)
    scored_docs = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]

def make_context(evidence: List[Tuple[Document, float]], max_words: int = 1600) -> str:
    context = ""
    word_count = 0
    for doc, score in evidence[:3]:
        chunk = f"[Page {doc.metadata['page']} | Section: {doc.metadata['section']} | Score: {score:.2f}]\n{doc.page_content.strip()}\n\n"
        wc = len(chunk.split())
        if word_count + wc > max_words:
            break
        context += chunk
        word_count += wc
    return context.strip()

def check_compliance(requirement: str, evidence: List[Tuple[Document, float]]) -> Dict:
    if not evidence:
        return {"compliance": "Not found", "page": None}

    context = make_context(evidence)
    prompt = f"""### Role ###
You are an expert dutch financial auditor reviewing a dutch annual financial report for a company. Your task is to determine if 
the report satisfies a specific compliance requirement and identify the single most 
relevant page where this requirement is addressed.

### Requirement ###
{requirement}

### Context Excerpt ###
{context}

### Instructions ###
1. Analyze if the requirement is satisfied (fully or partially) based on the context
2. If satisfied, respond with the SINGLE most relevant page number (just the number)
3. If not satisfied, respond with "Not found"
4. If you think it's even partially satisfied, respond with "Found"

### Response Format ###
Compliance: [Found/Not found]
Page: [number/null]
"""

    try:
        response = client.chat.completions.create(
        model="llama3-70b-8192",  # Upgraded to a production-grade reasoning model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.05,
        max_tokens=128,
        top_p=0.95
    ).choices[0].message.content

        compliance_match = re.search(r'Compliance:\s*(Found|Not found|Uncertain)', response, re.IGNORECASE)
        page_match = re.search(r'Page:\s*(\d+)', response)

        compliance = compliance_match.group(1).capitalize() if compliance_match else "Uncertain"
        page = int(page_match.group(1)) if page_match else None

        return {"compliance": compliance, "page": page}
    except Exception as e:
        print(f"⚠️ API Error: {str(e)}")
        return {"compliance": "Error", "page": None}

def main():
    print("🚀 Starting Advanced Compliance Checker")
    checklist = load_and_prepare_checklist("checklist.xlsx")
    documents = process_financial_pdf("financial.pdf")
    rag = AdvancedFinancialRAG()
    rag.vector_store = rag.build_optimized_vector_store(documents)
    results = []

    with tqdm(total=len(checklist), desc="Checking compliance") as pbar:
        for req in checklist:
            query = req['text'] if req['is_main'] else f"{req['main_condition']}: {req['text']}"
            evidence = retrieve_evidence(query, rag.vector_store, rag.reranker)
            compliance = check_compliance(query, evidence)

            result = {
                "id": req['id'],
                "requirement": req['text'],
                "category": req['category'],
                "is_main": req['is_main'],
                "row_index": req['row_index'],
                "compliance_status": compliance["compliance"],
                "page": compliance["page"]
            }
            if not req['is_main']:
                result["main_id"] = req['main_id']
                result["main_requirement"] = req['main_condition']
            results.append(result)
            pbar.update(1)

    df = pd.DataFrame(results)
    columns = ['id', 'is_main', 'category', 'requirement', 'compliance_status', 'page', 'row_index']
    if 'main_id' in df.columns:
        columns.insert(1, 'main_id')
    if 'main_requirement' in df.columns:
        columns.insert(2, 'main_requirement')
    df = df[columns]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"compliance_report_{timestamp}.xlsx"
    df.to_excel(output_file, index=False)

    print(f"✅ Report saved to {output_file}")
    print(f"📋 Requirements processed: {len(results)}")
    found = sum(1 for r in results if r['compliance_status'] == 'Found')
    print(f"📈 Accuracy: {found}/{len(results)} = {found / len(results) * 100:.2f}%")

if __name__ == '__main__':
    main()
