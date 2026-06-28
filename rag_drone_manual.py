import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# CONFIGURAZIONE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "manuale.pdf")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print("--- 1. Caricamento PDF e Splitting ---")
loader = PyPDFLoader(PDF_PATH, extract_images=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=700,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_number"] = i + 1

print(f"Documento diviso in {len(chunks)} chunk.")


# EMBEDDINGS E VECTOR STORE
print("\n--- 2. Generazione Embeddings (Modello Multilingue) ---")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'} 
)

if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    print("Database vettoriale trovato in locale. Caricamento...")
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
else:
    print("Database non presente. Generazione embeddings in corso (potrebbe richiedere qualche minuto)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("Nuovo database vettoriale creato.")

# LOGIC CORE 
llm = OllamaLLM(model="llama3", temperature=0.0)

PROMPT_TEMPLATE = """Sei un ingegnere e assistente tecnico esperto e rigoroso. Rispondi alla domanda basandoti SOLO ed ESCLUSIVAMENTE sui 
chunk di testo forniti nel contesto.

REGOLE TASSATIVE DI COMPORTAMENTO:
1. Distinzione dei componenti: Fai estrema attenzione a non confondere componenti o dispositivi diversi che condividono parole chiave simili nel 
testo (es. batteria del drone vs batteria del radiocomando).
2. Dati certi e fattuali: Riporta solo i dati numerici, le specifiche e le istruzioni scritte esplicitamente. 
Se noti cifre palesemente sballate o unite dall'OCR, prediligi i dati strutturati chiaramente negli altri chunk.
3. Integrita' delle liste: Se nel contesto e' presente un elenco o una serie di punti protettivi, digitali tutti in ordine senza saltare o troncare nulla.
4. Cita le fonti: Alla fine di ogni blocco di dati o punto elenco, scrivi sempre il (Chunk Numero X) da cui proviene.

Se il contesto non e' sufficiente o e' del tutto assente, di': "Non trovo questa informazione nel manuale."

CONTESTO:
{context}

DOMANDA: {question}

RISPOSTA:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

def query_rag(query_text):
    results = vector_store.similarity_search(query_text, k=3)
    
    if not results:
        return "Non trovo questa informazione nel manuale.", ["Nessun documento trovato nel database."]
    
    results = sorted(results, key=lambda x: x.metadata.get("chunk_number", 0))
    
    context_text = ""
    sources_info = []
    
    for doc in results:
        chunk_num = doc.metadata.get("chunk_number", "Sconosciuto")
        page_num = doc.metadata.get("page", "Sconosciuta")
        
        context_text += f"\n--- [Chunk Numero {chunk_num}] ---\n{doc.page_content}\n"
        sources_info.append(f"Chunk {chunk_num} (Pagina {page_num + 1})")

    full_prompt = prompt.format(context=context_text, question=query_text)
    response = llm.invoke(full_prompt)
    
    return response, sources_info

# INTERFACCIA
if __name__ == "__main__":
    print("\n--- Sistema RAG Pronto ---")
    
    while True:
        user_query = input("\nPoni una domanda al manuale (o 'esci'): ")
        if user_query.lower() == 'esci':
            break
        if not user_query.strip():
            continue
            
        print("Ricerca nei vettori e generazione risposta...")
        answer, sources = query_rag(user_query)
        
        print("\n================ RISPOSTA ================")
        print(answer)
        print("==========================================")
        print("Fonti estratte (ordinate per rilevanza):")
        for src in sources:
            print(f" - {src}")