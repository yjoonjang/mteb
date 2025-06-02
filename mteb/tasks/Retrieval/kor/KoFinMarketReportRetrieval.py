from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoFinMarketReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoFinMarketReportRetrieval",
        description="This dataset is constructed for retrieval tasks related to the stock market, based on Korean financial reports. It enables the evaluation of Korean RAG performance in the financial domain by providing publicly accessible market reports, queries, and answers.",
        reference="https://arxiv.org/pdf/2502.07131",
        dataset={
            "path": "nmixx-fin/twice_kr_market_report_retrieval",
            "revision": "9e3a77a3210b257dc51b3c9bdc038b8898dc20ab",
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
