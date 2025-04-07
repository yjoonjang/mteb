from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ToolRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ToolRetrieval",
        description="This dataset enables the evaluation of tool retrieval performance.",
        reference="https://arxiv.org/abs/2503.01763",
        dataset={
            "path": "yjoonjang/toolret",
            "revision": "50e91b691d2a347735bc2c1b4dd235f4047d1984",
        },
        type="Retrieval",
        prompt="Retrieve tools based on user query.",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-04-07", "2025-04-07"),
        domains=["Web"],
        task_subtypes=["Article retrieval", "Code retrieval", "Conversational retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{shi2025retrieval,
  title={Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models},
  author={Shi, Zhengliang and Wang, Yuhan and Yan, Lingyong and Ren, Pengjie and Wang, Shuaiqiang and Yin, Dawei and Ren, Zhaochun},
  journal={arXiv preprint arXiv:2503.01763},
  year={2025}
}""",
    )
