from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoSquadv1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KoSquadv1Retrieval",
        description="KorQuAD 1.0 is a large-scale question-and-answer dataset constructed for Korean machine reading comprehension, and investigate the dataset to understand the distribution of answers and the types of reasoning required to answer the question. This dataset benchmarks the data generating process of SQuAD v1.0 to meet the standard.",
        reference="https://arxiv.org/abs/1909.07005",
        dataset={
            "path": "yjoonjang/squad_kor_v1",
            "revision": "2b4ee1f3b143a04792da93a3df21933c5fe9eed3",
        },
        type="Retrieval",
        prompt="Retrieve text based on user query.",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2019-09-16", "2019-09-16"),
        domains=["Non-fiction", "Academic", "Encyclopaedic", "Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{lim2019korquad1,
  title={Korquad1. 0: Korean qa dataset for machine reading comprehension},
  author={Lim, Seungyoung and Kim, Myungji and Lee, Jooyoul},
  journal={arXiv preprint arXiv:1909.07005},
  year={2019}
}
""",
    )
