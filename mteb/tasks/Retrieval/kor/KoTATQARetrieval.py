from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoTATQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoTATQARetrieval",
        description="This dataset is constructed for TAT (Textual and Tabular) QA retrieval tasks based on information collected from Korean financial reports. It enables the evaluation of Korean RAG performance in the financial domain by providing publicly accessible textual and tabular data, queries, and answers.",
        reference="https://arxiv.org/pdf/2502.07131",
        dataset={
            "path": "nmixx-fin/twice_tat_qa_retrieval",
            "revision": "13654f0de74889de493e149503e2f9880cad2a3c",
        },
        type="Retrieval",
        prompt="Retrieve text based on user query.",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-02-01", "2025-02-01"),
        domains=["Social", "Financial"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{hwang2025twiceadvantageslowresourcedomainspecific,
      title={TWICE: What Advantages Can Low-Resource Domain-Specific Embedding Model Bring? -- A Case Study on Korea Financial Texts}, 
      author={Yewon Hwang and Sungbum Jung and Hanwool Lee and Sara Yu},
      year={2025},
      eprint={2502.07131},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07131}, 
}
""",
    )
