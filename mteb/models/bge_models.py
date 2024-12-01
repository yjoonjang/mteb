from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import torch

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.requires_package import requires_package

from .wrapper import Wrapper

BGE_M3_SUPPORTING_LANGUAGES = []


class BGEM3Wrapper(Wrapper):
    def __init__(
        self,
        model: str,
        type: Literal["dense", "sparse", "multi_vector"] | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> None:
        requires_package(self, "FlagEmbedding", "FlagEmbedding")
        from FlagEmbedding import BGEM3FlagModel

        if isinstance(model, str):
            self.model = BGEM3FlagModel(model, **kwargs)
        else:
            self.model = model
        self.type = type

    def encode(self, sentences: Sequence[str], **kwargs) -> np.ndarray:
        print("Encode kwargs:", kwargs)  # Print the kwargs dictionary
        
        embeddings = None
        if self.type == "dense":
            embeddings = self.model.encode(
                sentences,
                batch_size=kwargs.get("batch_size", 1),
                # **kwargs,
            )
        elif self.type == "sparse":
            print("Sparse")
            embeddings = self.model.encode(
                sentences,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=kwargs.get("batch_size", 1),
                # **kwargs,
            )
            print("Sparse embeddings:", embeddings)
        elif self.type == "multi_vector":
            embeddings = self.model.encode(
                sentences,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                batch_size=kwargs.get("batch_size", 1),
                # **kwargs,
            )

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings

    def _predict(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        return self.model.predict(
            sentences,
            convert_to_numpy=True,
            **kwargs,
        )


model_prompts = {"query": "Represent this sentence for searching relevant passages: "}

bge_small_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=24_000_000,
    memory_usage=None,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
)

bge_base_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=438_000_000,
    memory_usage=None,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=1_340_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
)

bge_m3_dense = ModelMeta(
    loader=partial(BGEM3Wrapper, model="BAAI/bge-m3", type="dense"),
    name="BAAI/bge-m3/dense",
    languages=BGE_M3_SUPPORTING_LANGUAGES,
    open_weights=True,
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    release_date="2024-01-30",
    n_parameters=560_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/BAAI/bge-m3",
    similarity_fn_name="cosine",
    framework=["FlagEmbedding"],
    use_instructions=False,
)

bge_m3_sparse = ModelMeta(
    loader=partial(BGEM3Wrapper, model="BAAI/bge-m3", type="sparse"),
    name="BAAI/bge-m3/sparse",
    languages=BGE_M3_SUPPORTING_LANGUAGES,
    open_weights=True,
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    release_date="2024-01-30",
    n_parameters=560_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/BAAI/bge-m3",
    similarity_fn_name="cosine",
    framework=["FlagEmbedding"],
    use_instructions=False,
)

bge_m3_multi_vector = ModelMeta(
    loader=partial(BGEM3Wrapper, model="BAAI/bge-m3", type="multi_vector"),
    name="BAAI/bge-m3/multi_vector",
    languages=BGE_M3_SUPPORTING_LANGUAGES,
    open_weights=True,
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    release_date="2024-01-30",
    n_parameters=560_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=8192,
    reference="https://huggingface.co/BAAI/bge-m3",
    similarity_fn_name="cosine",
    framework=["FlagEmbedding"],
    use_instructions=False,
)
