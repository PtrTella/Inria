# Testing and running the different cache policies implemented in the cluster

from PromptDatasetManager import PromptDatasetManager



manager = PromptDatasetManager()
manager.load_local_metadata("normalization_embeddings.parquet")


# Mana