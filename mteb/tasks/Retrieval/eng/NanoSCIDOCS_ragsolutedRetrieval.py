from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class NanoSCIDOCS_ragsolutedRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSCIDOCS_ragsolutedRetrieval",
        description="NanoFiQA2018 is a smaller subset of "
        + "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
        + " prediction, to document classification and recommendation.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "yjoonjang/nanoscidocs_ragsoluted",
            "revision": "484eb90549fc3f0b9c42b3551e80ceb999515537",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}""",
        prompt={
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset(
            "yjoonjang/nanoscidocs_ragsoluted",
            "corpus",
            revision="d55eca8722edf368ba411ab4282d19fb6b7f091a",
        )
        self.queries = load_dataset(
            "yjoonjang/nanoscidocs_ragsoluted",
            "queries",
            revision="d55eca8722edf368ba411ab4282d19fb6b7f091a",
        )
        self.relevant_docs = load_dataset(
            "yjoonjang/nanoscidocs_ragsoluted",
            "qrels",
            revision="d55eca8722edf368ba411ab4282d19fb6b7f091a",
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
