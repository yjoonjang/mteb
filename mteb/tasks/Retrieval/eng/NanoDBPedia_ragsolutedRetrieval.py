from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoDBPedia_ragsolutedRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoDBPedia_ragsolutedRetrieval",
        description="NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.",
        reference="https://huggingface.co/datasets/zeta-alpha-ai/NanoDBPedia",
        dataset={
            "path": "yjoonjang/nanodbpedia_ragsoluted",
            "revision": "438f1c25129f05db6238699b5afdc9c6b58d2096",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2015-01-01", "2015-12-31"],
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{lehmann2015dbpedia, title={DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia}, author={Lehmann, Jens and et al.}, journal={Semantic Web}, year={2015}}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "yjoonjang/nanodbpedia_ragsoluted",
            "corpus",
            revision="5cc47157896010f6306e29cc9ede54cda8a01655",
        )
        self.queries = load_dataset(
            "yjoonjang/nanodbpedia_ragsoluted",
            "queries",
            revision="5cc47157896010f6306e29cc9ede54cda8a01655",
        )
        self.relevant_docs = load_dataset(
            "yjoonjang/nanodbpedia_ragsoluted",
            "qrels",
            revision="5cc47157896010f6306e29cc9ede54cda8a01655",
        )

        self.corpus = {
            split: {
                sample["_id"]: {"_id": sample["_id"], "text": sample["text"]}
                for sample in self.corpus[split]
            }
            for split in self.corpus
        }

        self.queries = {
            split: {sample["_id"]: sample["text"] for sample in self.queries[split]}
            for split in self.queries
        }

        relevant_docs = {}

        for split in self.relevant_docs:
            relevant_docs[split] = defaultdict(dict)
            for query_id, corpus_id in zip(
                self.relevant_docs[split]["query-id"],
                self.relevant_docs[split]["corpus-id"],
            ):
                relevant_docs[split][query_id][corpus_id] = 1
        self.relevant_docs = relevant_docs

        self.data_loaded = True
