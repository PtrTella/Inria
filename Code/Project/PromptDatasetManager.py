from typing import Dict, List, Optional, Tuple, Union
import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import random

class PromptDatasetManager:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.emb_matrix: np.ndarray = None
        self.prompts_arr: np.ndarray = None
        self.embs_arr: np.ndarray = None
        self.user_to_idx: dict[str, np.ndarray] = {}
        self.session_to_idx: dict[tuple[str,int], np.ndarray] = {}

    def load_local_metadata(
        self,
        path: str,
        max_rows: Optional[int] = None,
        add_columns: Optional[List[str]] = None,
        load_embeddings: bool = True
    ):
        # 1) Lettura Parquet
        cols = ['prompt','clip_emb','user_name','timestamp']
        if add_columns:
            cols.extend(add_columns)
        table = pq.read_table(path, columns=cols)
        if max_rows:
            table = table.slice(0, max_rows)
        self.df = table.to_pandas()
        # 2) Timestamp -> datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        # 3) Embedding matrix
        if load_embeddings:
            self.emb_matrix = np.vstack(self.df['clip_emb'].values).astype(np.float32)
        else:
            N = len(self.df)
            self.emb_matrix = np.zeros((N,0), dtype=np.float32)
        # 4) Prompt array
        self.prompts_arr = self.df['prompt'].to_numpy(dtype=object)
        # 5) Precompute sorted order and delta
        self._precompute_sorted()
        # 6) Global time order for sample_prompts
        self.time_sorted_idx = self.df.sort_values('timestamp').index.to_numpy()
        # 7) Build user->indices map
        self._build_user_indices()

    def _precompute_sorted(self):
        """
        Ordina per user_name, timestamp e calcola delta (minuti) tra record consecutivi.
        """
        order = self.df.sort_values(['user_name','timestamp']).index.to_numpy()
        self.sorted_idx = order
        users = self.df['user_name'].to_numpy()[order]
        # Estrai timestamps come datetime64[ns] e ordina
        times = self.df['timestamp'].to_numpy(dtype='datetime64[ns]')[order]
        # Converti a nanosecondi interi
        ts_int = times.astype('int64')
        # Differenze consecutive in ns -> minuti
        dt_ns = ts_int[1:] - ts_int[:-1]
        dt_min = dt_ns.astype(np.float32) / 1e9 / 60.0
        # Assegna delta
        delta = np.empty(len(order), dtype=np.float32)
        delta[0] = np.inf
        same = users[1:] == users[:-1]
        delta[1:] = np.where(same, dt_min, np.inf)
        self.sorted_delta = delta

    def _build_user_indices(self):
        groups = self.df.groupby('user_name', sort=False).indices
        self.user_to_idx = {u: np.asarray(idxs, dtype=np.int32) for u, idxs in groups.items()}
        

    def sample_sessions(
        self,
        gap: float,
        num_sessions: Optional[int] = None,
        max_prompts: Optional[int] = None,
        random_order: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Campiona sessioni definite da gap (minuti) tra prompt dello stesso user:
        - gap: soglia per separare sessioni
        - num_sessions=None -> tutte le sessioni (in ordine se random_order=False)
                       else -> campione casuale o primi
        - max_prompts=None -> tutti i prompt
                       else -> primi max_prompts di ogni sessione
        - random_order=False -> restituisce in ordine sequenziale di apparizione
        """
        # 1) breakpoints dove gap superato
        breaks = np.nonzero(self.sorted_delta > gap)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends   = np.concatenate((breaks, [len(self.sorted_delta) - 1]))
        segments = list(zip(starts, ends))
        # 2) campionamento
        if num_sessions is not None and num_sessions < len(segments):
            if random_order:
                segments = random.sample(segments, num_sessions)
            else:
                segments = segments[:num_sessions]
        # 3) estrai batches
        out = []
        for s, e in segments:
            length = e - s + 1
            take = length if max_prompts is None else min(length, max_prompts)
            idxs = self.sorted_idx[s:s+take]
            out.append((self.prompts_arr[idxs], self.emb_matrix[idxs]))
        return out

    def sample_user_prompts(
        self,
        num_users: Optional[int] = None,
        random_order: bool = True,
        per_user: bool = False
    ) -> Union[List[Tuple[str, np.ndarray]], Dict[str, List[Tuple[str, np.ndarray]]]]:
        """
        Campiona prompt per un sottoinsieme di utenti:
        - num_users=None -> tutti gli utenti; altrimenti scegli num_users utenti (random o sequenziali)
        - per_user=False -> lista unica di (prompt, emb) concatenata per gli utenti selezionati
        - per_user=True  -> dict user->lista di (prompt, emb)
        - random_order=True/False determina l'ordine dei prompt per ciascun utente
        """
        # Seleziona utenti
        users = list(self.user_to_idx.keys())
        if num_users is not None and num_users < len(users):
            if random_order:
                selected_users = random.sample(users, num_users)
            else:
                selected_users = users[:num_users]
        else:
            selected_users = users

        # Prepara output
        if per_user:
            out: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        else:
            out: List[Tuple[str, np.ndarray]] = []

        # Per ciascun utente selezionato, estrai tutti i prompt
        for user in selected_users:
            idxs = self.user_to_idx.get(user, np.array([], dtype=int))
            if idxs.size == 0:
                if per_user:
                    out[user] = []
                continue
            # eventualmente mescola i prompt per utente
            order = np.random.permutation(idxs) if random_order else idxs
            items = [(self.prompts_arr[i], self.emb_matrix[i]) for i in order]
            if per_user:
                out[user] = items
            else:
                out.extend(items)

        return out

    def sample_prompts(
        self,
        num_prompts: Optional[int] = None,
        random_order: bool = True
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Campiona prompt globali:
        - num_prompts=None -> tutti
        - num_prompts=k    -> primi k in sequenza o k random
        - random_order=False -> sequenza cronologica globale (timestamp)
        - random_order=True  -> shuffle
        """
        N = len(self.prompts_arr)
        if random_order:
            k = min(num_prompts, N) if num_prompts is not None else N
            idxs = np.random.choice(N, size=k, replace=False)
        else:
            # indice solo per timestamp
            time_order = self.df.sort_values('timestamp').index.to_numpy()
            idxs = time_order if num_prompts is None else time_order[:num_prompts]
        return [(self.prompts_arr[i], self.emb_matrix[i]) for i in idxs]
    

    def get_k_nearest_images(self, row_index: int = None, k: int = 5) -> pd.DataFrame:
        """
        Restituisce le k immagini più vicine (escludendo quella stessa) usando FAISS Flat Index.
        
        :param row_index: indice della riga di riferimento
        :param k: numero di immagini simili da restituire (esclusa quella stessa)
        :return: DataFrame con le k righe più vicine (metadati inclusi)
        """
        if row_index is None:
            row_index = random.randint(0, len(self.df) - 1)

        if self.emb_matrix is None or self.df is None:
            raise ValueError("emb_matrix o df non sono inizializzati.")

        vectors = self.emb_matrix.astype('float32')
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        query = vectors[row_index].reshape(1, -1)
        distances, indices = index.search(query, k + 1)  # +1 perché include se stesso

        # Rimuove l'indice della query se incluso, in realtà è sempre il primo e lo voglio
        #result_indices = [i for i in indices[0] if i != row_index][:k]

        return self.df.iloc[indices[0]], distances[0]
