from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQ_simplifiedRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ_simplifiedRetrieval",
        dataset={
            "path": "yjoonjang/nq_simplified",
            "revision": "b774495ed302d8c44a3a7ea25c90dbce03968f31",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{47761,title	= {Natural Questions: a Benchmark for Question Answering Research},
        author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh 
        and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee 
        and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le 
        and Slav Petrov},year	= {2019},journal	= {Transactions of the Association of Computational 
        Linguistics}}""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
    )

