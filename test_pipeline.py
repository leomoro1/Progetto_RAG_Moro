import os
from rag_drone_manual import query_rag

# Definizione del set di 10 domande basate sul manuale del DJI Mini 2
test_questions = [

    # 5 Domande PRESENTI nel manuale
    "Quali sono le caratteristiche della batteria del drone?",
    "Quali sono i requisiti dell'ambiente di volo?",
    "Quali sono le tre modalita' di volo disponibili e dove si puo' passare dall'una all'altra?",
    "Quali sono le caratteristiche del radiocomando?",
    "Quali sono le quattro sottomodalita' previste per la funzione di volo intelligente QuickShot?",
    
    # 5 Domande ASSENTI dal manuale
    "Come si cucina una pizza margherita?",
    "Come si sostituisce il motore elettrico del drone?",
    "Come si configura l'account DJI Fly usando un profilo Google o Apple?",
    "Qual e' la ricetta tradicional per cucinare una carbonara perfetta?",
    "Quali sono le sanzioni previste se si vola sopra un'autostrada senza autorizzazione?"
]

def run_evaluation():
    output_filename = "report_test_dji.txt"
    print(f"Avvio del test di valutazione su {len(test_questions)} domande...")
    
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("==================================================\n")
        file.write("REPORT DI VERIFICA E TESTING - PIPELINE RAG (DJI Mini 2)\n")
        file.write("==================================================\n\n")
        
        for i, question in enumerate(test_questions, 1):
            section_type = "PRESENTI NEL MANUALE" if i <= 5 else "ASSENTI DAL MANUALE"
            
            print(f"[{i}/10] Elaborazione domanda ({section_type})...")
            
            answer, sources = query_rag(question)
            
            # Scrittura formattata dei risultati nel file di log
            file.write(f"Domanda {i} [{section_type}]:\n")
            file.write(f"-> {question}\n\n")
            file.write(f"Risposta LLM:\n{answer}\n\n")
            file.write(f"Chunk estratti dal DB: {', '.join(sources)}\n")
            file.write("-" * 60 + "\n\n")
            
    print(f"\nTest completato! I risultati sono stati salvati in '{output_filename}'.")

if __name__ == "__main__":
    run_evaluation()