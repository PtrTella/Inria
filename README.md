# SimCache: Similarity-Based Caching for Vector Embeddings

SimCache is a Python framework for implementing and evaluating similarity-based caches. Unlike traditional caches that require exact key matches, SimCache uses vector embeddings and similarity metrics to serve "near-misses" as hits, optimizing performance for AI and retrieval workflows.

---

## 🚀 Come Iniziare

Per eseguire il progetto, usa l'entry point unificato `main.py` dalla cartella principale:

1.  **Dashboard Interattiva**:
    ```bash
    python3 main.py dashboard
    ```
2.  **Benchmark (Linea di Comando)**:
    ```bash
    python3 main.py benchmark --policies LRU LFU --num-requests 500
    ```

---

## 📂 Struttura del Progetto

- `main.py`: Punto di ingresso unico per avviare il progetto.
- `src/simcache/`: Il "cuore" del progetto (Package Python).
- `notebooks/`: Notebook organizzati per Ricerca ed Hub di Valutazione.
- `docs/`: Documentazione tecnica e schemi architetturali.

## 🛠️ Caratteristiche Principali

- **Multiple Backends**: Supporto per Faiss (Flat, IVF, HNSW), Annoy, e Linear index.
- **Politiche Flessibili**: LRU, LFU, TTL, Greedy, OSA, Duel e altro.
- **Simulatore Integrato**: Per misurare Hit Rate e costi di servizio.

## 📝 Note per lo Sviluppo
Tutti i file all'interno di `src/simcache/` sono moduli di un pacchetto. Per utilizzarli in nuovi script, assicurati di aggiungere `src` al tuo `PYTHONPATH` o usa `main.py` come base.

---
*Progetto sviluppato per la ricerca sulla Similarity Caching.*
