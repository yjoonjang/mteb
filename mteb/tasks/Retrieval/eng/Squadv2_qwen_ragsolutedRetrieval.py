from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Squadv2_ragsolutedRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Squadv2_qwen_ragsolutedRetrieval",
        description="from squad v2",
        reference="https://huggingface.co/datasets/yjoonjang/squad_v2",
        dataset={
            "path": "yjoonjang/squad_v2_ragsoluted_qwen",
            "revision": "ff1541cd6892d4c55420e666992ffaf7385958c7",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@article{tan2023towards,
  title={Towards benchmarking and improving the temporal reasoning capability of large language models},
  author={Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
  journal={arXiv preprint arXiv:2306.08952},
  year={2023}
}
""",
        prompt={"query": "Given the following question, retrieve the correct answer."},
    )
