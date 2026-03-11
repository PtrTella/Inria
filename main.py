import sys
import os
import subprocess

# Aggiunge 'src' al path di Python
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("==========================================")
    print("   Inria Similarity Caching - Menu        ")
    print("==========================================")
    print("1) Avvia Dashboard Interattiva (Browser)")
    print("2) Avvia Benchmark (Linea di Comando)")
    print("3) Informazioni sul Progetto")
    print("q) Esci")
    print("------------------------------------------")
    
    choice = input("Scegli un'opzione: ").strip().lower()

    # Prepara l'ambiente con PYTHONPATH che include 'src'
    env = os.environ.copy()
    src_path = os.path.join(os.path.dirname(__file__), "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    if choice == '1':
        print("\n--- Avviando il Dashboard ---")
        subprocess.run([sys.executable, "-m", "simcache.Dashboard"], env=env)
    
    elif choice == '2':
        print("\n--- Configurazione Benchmark ---")
        print("Eseguo test base (LRU/LFU, 500 requests)...")
        subprocess.run([sys.executable, "-m", "simcache.benchmark_cache_policies", "--num-requests", "500"], env=env)
    
    elif choice == '3':
        print("\n--- Informazioni ---")
        print("SimCache è un framework per testare cache di similarità.")
        print("Documentazione disponibile in 'docs/' e 'README.md'.")
        print("Core logic in 'src/simcache/'.")

    else:
        print("Arrivederci!")

if __name__ == "__main__":
    main()
