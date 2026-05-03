import argparse
import ast
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm  # 新增

from config import (
    BGE3_MODEL_NAME,
    TOP_K,
    VECTOR_STORE_DIRNAME,
    RAG_SPLIT_MODE,
    RAG_TRAIN_RATIO,
    RAG_RANDOM_SEED,
    RAG_USE_TRAIN_ONLY,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings




class LogVectorStore:
    def __init__(
        self,
        dataset: str = "BGL",
        outputs_root: str = "outputs",
        split_mode: str = RAG_SPLIT_MODE,
        train_ratio: float = RAG_TRAIN_RATIO,
        random_seed: int = RAG_RANDOM_SEED,
        use_train_only: bool = RAG_USE_TRAIN_ONLY,
    ):
        self.dataset = dataset
        self.split_mode = split_mode
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.use_train_only = use_train_only
        self.repo_root = Path(__file__).resolve().parents[2]
        self.outputs_root = (self.repo_root / outputs_root).resolve()
        self.dataset_dir = self.outputs_root / self.dataset
        self.csv_path = self.dataset_dir / "grouped_logs.csv"
        self.index_dir = self.dataset_dir / self._build_index_dirname()
        self.embedding = HuggingFaceEmbeddings(
            model_name=BGE3_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_db = None

    def _build_index_dirname(self) -> str:
        if not self.use_train_only:
            return f"{VECTOR_STORE_DIRNAME}_all"
        ratio_tag = str(self.train_ratio).replace(".", "p")
        return f"{VECTOR_STORE_DIRNAME}_{self.split_mode}_{ratio_tag}_seed{self.random_seed}"

    def _split_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_train_only:
            print("ℹ️ 使用全量样本构建向量库")
            return df.reset_index(drop=True)

        if not (0 < self.train_ratio <= 1):
            raise ValueError(f"train_ratio 必须在 (0,1]，当前: {self.train_ratio}")
        if self.split_mode not in {"ordered", "random"}:
            raise ValueError(f"split_mode 必须是 ordered/random，当前: {self.split_mode}")

        train_size = int(len(df) * self.train_ratio)
        if train_size <= 0:
            raise ValueError("训练切分后样本数为 0，请提高 train_ratio")

        if self.split_mode == "ordered":
            split_df = df.iloc[:train_size].copy()
        else:
            split_df = df.sample(n=train_size, random_state=self.random_seed).copy()

        print(
            f"ℹ️ 向量库按训练集构建: mode={self.split_mode}, "
            f"train_ratio={self.train_ratio}, rows={len(split_df)}/{len(df)}"
        )
        return split_df.reset_index(drop=True)

    @staticmethod
    def _parse_templates(raw_templates) -> List[str]:
        if isinstance(raw_templates, list):
            return [str(item).strip() for item in raw_templates if str(item).strip()]
        if pd.isna(raw_templates):
            return []
        text = str(raw_templates).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [text]

    @staticmethod
    def _normalize_template(template: str) -> str:
        return str(template).strip()

    def _build_documents_from_grouped_logs(self) -> List[Document]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"未找到文件: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_cols = {"Templates", "Label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"grouped_logs.csv 缺少列: {sorted(missing_cols)}")

        df = self._split_df(df)
        deduped_docs = {}

        for row in tqdm(df.itertuples(index=False), total=len(df), desc="📝 构建日志模板文档"):
            templates = self._parse_templates(getattr(row, "Templates", ""))
            label = getattr(row, "Label", None)
            label_value = int(label) if pd.notna(label) else None
            for template in templates:
                normalized = self._normalize_template(template)
                if not normalized:
                    continue

                if normalized not in deduped_docs:
                    deduped_docs[normalized] = {
                        "page_content": normalized,
                        "metadata": {
                            "Label": label_value,
                            "dup_count": 1,
                        },
                    }
                else:
                    meta = deduped_docs[normalized]["metadata"]
                    meta["dup_count"] += 1
                    if label_value is not None:
                        meta["Label"] = int(max(meta["Label"] or 0, label_value))

        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in deduped_docs.values()
        ]

    def build_from_grouped_logs(self, force_rebuild: bool = False) -> int:
        if self.index_dir.exists() and not force_rebuild:
            self.vector_db = FAISS.load_local(
                str(self.index_dir),
                self.embedding,
                allow_dangerous_deserialization=True,
            )
            return self.vector_db.index.ntotal

        print("🔨 开始读取并解析日志模板...")
        documents = self._build_documents_from_grouped_logs()
        if not documents:
            raise ValueError(f"{self.csv_path} 中没有可用的 Templates")

        # 向量入库也加进度条
        print(f"🚀 开始生成向量库 (去重后共 {len(documents)} 条模板)...")
        self.vector_db = FAISS.from_documents(documents, self.embedding)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db.save_local(str(self.index_dir))
        return len(documents)

    def search(self, log: str, top_k: int = TOP_K):
        if self.vector_db is None:
            self.build_from_grouped_logs()
        return self.vector_db.similarity_search(log, k=top_k)

    def search_with_scores(self, log: str, top_k: int = TOP_K):
        if self.vector_db is None:
            self.build_from_grouped_logs()
        return self.vector_db.similarity_search_with_score(log, k=top_k)


def build_faiss_for_dataset(
    dataset: str,
    force_rebuild: bool = False,
    split_mode: str = RAG_SPLIT_MODE,
    train_ratio: float = RAG_TRAIN_RATIO,
    random_seed: int = RAG_RANDOM_SEED,
    use_train_only: bool = RAG_USE_TRAIN_ONLY,
):
    store = LogVectorStore(
        dataset=dataset,
        split_mode=split_mode,
        train_ratio=train_ratio,
        random_seed=random_seed,
        use_train_only=use_train_only,
    )
    count = store.build_from_grouped_logs(force_rebuild=force_rebuild)
    return store.index_dir, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 grouped_logs.csv 构建 BGE3 + FAISS 向量库")
    parser.add_argument("--dataset", default="BGL", help="数据集名称，例如 BGL/HDFS/Spirit/Thunderbird")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建向量库")
    parser.add_argument("--split-mode", default=RAG_SPLIT_MODE, choices=["ordered", "random"], help="切分方式")
    parser.add_argument("--train-ratio", type=float, default=RAG_TRAIN_RATIO, help="训练集比例")
    parser.add_argument("--random-seed", type=int, default=RAG_RANDOM_SEED, help="随机切分种子")
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        help="使用全量数据构建（默认仅训练集）",
    )
    parser.add_argument("--query", default=None, help="本地测试检索的日志文本")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="检索返回条数")
    args = parser.parse_args()

    store = LogVectorStore(
        dataset=args.dataset,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        use_train_only=not args.use_all_data,
    )
    count = store.build_from_grouped_logs(force_rebuild=args.force_rebuild)
    index_dir = store.index_dir
    print(f"\n✅ 向量库构建完成: {index_dir}")
    print(f"✅ 文档数量: {count}")

    if args.query:
        results = store.search_with_scores(args.query, top_k=args.top_k)
        print("\n🔎 检索结果")
        for i, (doc, score) in enumerate(results, start=1):
            print(f"[{i}] score={score:.4f} label={doc.metadata.get('Label')} dup_count={doc.metadata.get('dup_count')}")
            print(f"    {doc.page_content}")
