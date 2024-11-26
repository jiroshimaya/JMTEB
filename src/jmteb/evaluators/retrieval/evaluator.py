from __future__ import annotations

import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch
import tqdm
from loguru import logger
from torch import Tensor
from scipy.sparse.csr import csr_matrix
from torch import distributed as dist

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import (
    RetrievalDoc,
    RetrievalDocDataset,
    RetrievalPrediction,
    RetrievalQuery,
    RetrievalQueryDataset,
)

T = TypeVar("T")


class RetrievalEvaluator(EmbeddingEvaluator):
    """
    Evaluator for retrieval task.

    Args:
        val_query_dataset (RetrievalQueryDataset): validation dataset
        test_query_dataset (RetrievalQueryDataset): query dataset
        doc_dataset (RetrievalDocDataset): document dataset
        doc_chunk_size (int): The maximum size of corpus chunk. Smaller chunk requires less memory but lowers speed.
        ndcg_at_k (list[int] | None): top k documents to consider in NDCG (Normalized Documented Cumulative Gain).
        accuracy_at_k (list[int] | None): accuracy in top k hits.
        query_prefix (str | None): prefix for queries. Defaults to None.
        doc_prefix (str | None): prefix for documents. Defaults to None.
        log_predictions (bool): whether to log predictions of each datapoint. Defaults to False.
        top_n_docs_to_log (int): log only top n documents that are predicted as relevant. Defaults to 5.
    """

    def __init__(
        self,
        val_query_dataset: RetrievalQueryDataset,
        test_query_dataset: RetrievalQueryDataset,
        doc_dataset: RetrievalDocDataset,
        doc_chunk_size: int = 10000,
        accuracy_at_k: list[int] | None = None,
        ndcg_at_k: list[int] | None = None,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        log_predictions: bool = False,
        top_n_docs_to_log: int = 5,
    ) -> None:
        self.val_query_dataset = val_query_dataset
        self.test_query_dataset = test_query_dataset
        self.doc_dataset = doc_dataset

        self.doc_chunk_size = doc_chunk_size

        self.accuracy_at_k = accuracy_at_k or [1, 3, 5, 10]
        self.ndcg_at_k = ndcg_at_k or [10]
        self.max_top_k = max(sum([self.accuracy_at_k, self.ndcg_at_k], []))
        self.main_metric = f"ndcg@{self.ndcg_at_k[0]}"

        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.log_predictions = log_predictions
        self.top_n_docs_to_log = top_n_docs_to_log
    def __call__(
        self,
        model: TextEmbedder,
        cache_dir: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
    ) -> EvaluationResults:
        if not model.is_sparse_model:
            model.set_output_tensor()
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        batch_encode_query_func = model.batch_encode_with_cache
        batch_encode_doc_func = model.batch_encode_with_cache
        if model.__class__.__name__ in ("BM25Embedder", "TfidfEmbedder"):
            doc_text_list = [item.text for item in self.doc_dataset]
            model.fit(doc_text_list)
            batch_encode_query_func = model.batch_encode_query_with_cache
            batch_encode_doc_func = model.batch_encode_doc_with_cache
            
        val_query_embeddings = batch_encode_query_func(
            text_list=[item.query for item in self.val_query_dataset],
            prefix=self.query_prefix,
            cache_path=Path(cache_dir) / "val_query.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        if self.val_query_dataset == self.test_query_dataset:
            test_query_embeddings = val_query_embeddings
        else:
            test_query_embeddings = batch_encode_query_func(
                text_list=[item.query for item in self.test_query_dataset],
                prefix=self.query_prefix,
                cache_path=Path(cache_dir) / "test_query.bin" if cache_dir is not None else None,
                overwrite_cache=overwrite_cache,
            )

        doc_embeddings = batch_encode_doc_func(
            text_list=[item.text for item in self.doc_dataset],
            prefix=self.doc_prefix,
            cache_path=Path(cache_dir) / "corpus.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )

        logger.info("Start retrieval")

        dist_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            "cosine_similarity": Similarities.cosine_similarity,
            "dot_score": Similarities.dot_score,
            "euclidean_distance": Similarities.euclidean_distance,
        }
        if model.is_sparse_model:	
            # remove euclidean_distance for sparse model	
            dist_functions.pop("euclidean_distance")

        val_results = {}
        for dist_name, dist_func in dist_functions.items():
            val_results[dist_name], _ = self._compute_metrics(
                query_dataset=self.val_query_dataset,
                query_embeddings=val_query_embeddings,
                doc_embeddings=doc_embeddings,
                dist_func=dist_func,
            )
        sorted_val_results = sorted(val_results.items(), key=lambda res: res[1][self.main_metric], reverse=True)
        optimal_dist_name = sorted_val_results[0][0]

        test_scores, test_predictions = self._compute_metrics(
            query_dataset=self.test_query_dataset,
            query_embeddings=test_query_embeddings,
            doc_embeddings=doc_embeddings,
            dist_func=dist_functions[optimal_dist_name],
        )
        test_results = {optimal_dist_name: test_scores}

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_dist_name][self.main_metric],
            details={
                "optimal_distance_metric": optimal_dist_name,
                "val_scores": val_results,
                "test_scores": test_results,
            },
            predictions=test_predictions,
        )

    def _compute_metrics(
        self,
        query_dataset: RetrievalQueryDataset,
        query_embeddings: np.ndarray | Tensor | csr_matrix,
        doc_embeddings: np.ndarray | Tensor | csr_matrix,
        dist_func: Callable[[Tensor, Tensor], Tensor],
    ) -> tuple[dict[str, dict[str, float]], list[RetrievalPrediction]]:
        results: dict[str, float] = {}
        predictions: list[RetrievalPrediction] = [] if self.log_predictions else None
        is_csr_matrix = isinstance(doc_embeddings, csr_matrix) # type: ignore
        if is_csr_matrix:	
            doc_len = doc_embeddings.shape[0]	
            query_embeddings = query_embeddings.toarray() # type: ignore	
        else:	
            doc_len = len(doc_embeddings)
        with tqdm.tqdm(total=doc_len, desc="Retrieval doc chunks") as pbar:
            top_k_indices_chunks: list[np.ndarray] = []
            top_k_scores_chunks: list[np.ndarray] = []
            for offset in range(0, doc_len, self.doc_chunk_size):
                doc_embeddings_chunk = doc_embeddings[offset : offset + self.doc_chunk_size]

                if torch.cuda.is_available():
                    if dist.is_torchelastic_launched():
                        device = f"cuda:{dist.get_rank()}"
                    else:
                        device = "cuda"
                else:
                    device = "cpu"
                
                query_embeddings = to_tensor(query_embeddings, device=device)
                if is_csr_matrix:
                    doc_embeddings_chunk = to_sparse_tensor(
                        doc_embeddings_chunk,
                        device=device,
                    )
                else:
                    doc_embeddings_chunk = to_tensor(doc_embeddings_chunk, device=device)
                similarity = dist_func(query_embeddings, doc_embeddings_chunk)

                top_k = min(self.max_top_k, similarity.shape[1])  # in case the corpus is smaller than max_top_k
                top_k_scores, top_k_indices = torch.topk(
                    similarity,
                    k=top_k,
                    dim=1,
                )

                top_k_indices_chunks.append(top_k_indices + offset)
                top_k_scores_chunks.append(top_k_scores)

                pbar.update(len(doc_embeddings_chunk))

        top_k_indices = torch.cat(top_k_indices_chunks, axis=1)
        top_k_scores = torch.cat(top_k_scores_chunks, axis=1)

        top_k = min(self.max_top_k, top_k_indices.shape[0])
        sorting_indices_for_top_k = torch.argsort(-top_k_scores, axis=1)[:, :top_k]
        sorted_top_k_indices = torch.take_along_dim(top_k_indices, sorting_indices_for_top_k, axis=1).tolist()

        golden_doc_ids = [item.relevant_docs for item in query_dataset]
        retrieved_doc_ids = [[self.doc_dataset[i].id for i in indices] for indices in sorted_top_k_indices]

        predictions = (
            self._format_predictions(query_dataset, self.doc_dataset, retrieved_doc_ids, self.top_n_docs_to_log)
            if self.log_predictions
            else None
        )

        for k in self.accuracy_at_k:
            results[f"accuracy@{k}"] = accuracy_at_k(golden_doc_ids, retrieved_doc_ids, k)
        for k in self.ndcg_at_k:
            results[f"ndcg@{k}"] = ndcg_at_k(golden_doc_ids, retrieved_doc_ids, k)
        results[f"mrr@{self.max_top_k}"] = mrr_at_k(golden_doc_ids, retrieved_doc_ids, self.max_top_k)

        return results, predictions

    @staticmethod
    def _format_predictions(
        query_dataset: RetrievalQueryDataset,
        doc_dataset: RetrievalDocDataset,
        retrieved_doc_ids: list[list],
        top_n_to_log: int,
    ) -> list[RetrievalPrediction]:
        predictions = []
        for q, pred_docids in zip(query_dataset, retrieved_doc_ids):
            q: RetrievalQuery
            golden_docs: list[RetrievalDoc] = [
                doc_dataset[doc_dataset.docid_to_idx[docid]] for docid in q.relevant_docs
            ]
            pred_docids = pred_docids[:top_n_to_log]
            pred_docs: list[RetrievalDoc] = [
                doc_dataset[doc_dataset.docid_to_idx[pred_docid]] for pred_docid in pred_docids
            ]
            prediction = RetrievalPrediction(
                query=q.query,
                relevant_docs=golden_docs,
                predicted_relevant_docs=pred_docs,
            )
            predictions.append(prediction)
        return predictions


def accuracy_at_k(relevant_docs: list[list[T]], top_hits: list[list[T]], k: int) -> float:
    acc = 0
    for query_rel_docs, query_top_hits in zip(relevant_docs, top_hits):
        if len(query_rel_docs) == 0:
            warnings.warn("Query with no relevant documents found. Skip that from metric calculation.")
            continue

        for hit in query_top_hits[0:k]:
            if hit in query_rel_docs:
                acc += 1
                break
    return acc / len(relevant_docs)


def mrr_at_k(relevant_docs: list[list[T]], top_hits: list[list[T]], k: int) -> float:
    mrr = 0
    for query_rel_docs, query_top_hits in zip(relevant_docs, top_hits):
        if len(query_rel_docs) == 0:
            warnings.warn("Query with no relevant documents found. Skip that from metric calculation.")
            continue

        for rank, hit in enumerate(query_top_hits[0:k], start=1):
            if hit in query_rel_docs:
                mrr += 1.0 / rank
                break
    return mrr / len(relevant_docs)


def ndcg_at_k(relevant_docs: list[list[T]], top_hits: list[list[T]], k: int) -> float:
    total_ndcg_scores = 0
    num_valid_queries = 0
    for query_rel_docs, query_top_hits in zip(relevant_docs, top_hits):
        if len(query_rel_docs) == 0:
            warnings.warn("Query with no relevant documents found. Skip that from metric calculation.")
            continue

        dcg = 0
        for rank, hit in enumerate(query_top_hits[0:k], start=1):
            if hit in query_rel_docs:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = sum([1 / np.log2(rank + 1) for rank in range(1, len(query_rel_docs) + 1)])
        total_ndcg_scores += dcg / idcg

        num_valid_queries += 1
    return total_ndcg_scores / len(relevant_docs)


def to_tensor(embeddings: np.ndarray | Tensor, device: str) -> Tensor:
    if not isinstance(embeddings, Tensor):
        embeddings = torch.tensor(
            embeddings, dtype=torch.float32
        )  # 明示的にfloat32を指定
    else:
        embeddings = embeddings.float()  # 既にTensorの場合、float32に変換
    if len(embeddings.shape) == 1:
        embeddings = embeddings.unsqueeze(0)
    return embeddings.to(device=device)

def to_sparse_tensor(embeddings: csr_matrix | Tensor, device: str) -> Tensor:
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.float()  # float32に変換
        return embeddings.to_sparse().to(device=device)
    elif isinstance(embeddings, csr_matrix):  # type: ignore
        coo = embeddings.tocoo()
        indices = np.vstack((coo.row, coo.col))
        indices_tensor = torch.from_numpy(indices).long()
        values_tensor = torch.from_numpy(coo.data).float()  # float32に変換
        shape = coo.shape
        sparse_tensor = torch.sparse_coo_tensor(
            indices_tensor, values_tensor, torch.Size(shape), dtype=torch.float32
        )
        return sparse_tensor.to(device=device)
    else:
        raise TypeError("embeddings must be a csr_matrix or a Tensor")
    
@dataclass
class Similarities:
    @staticmethod
    def cosine_similarity(e1: Tensor, e2: Tensor) -> Tensor:
        """
        コサイン類似度を計算します。e2 がスパーステンソルの場合に対応しています。
        """
        e1_norm = torch.nn.functional.normalize(e1, p=2, dim=1)

        if e2.is_sparse:
            # スパーステンソルをコアリッシュ
            e2 = e2.coalesce()

            # 各行のノルムを計算
            row_norms = torch.sparse.sum(e2 * e2, dim=1).values().sqrt()
            row_norms = row_norms + 1e-10  # ゼロ除算防止

            # 値を正規化
            normalized_values = e2.values() / row_norms[e2.indices()[0]]
            e2_norm = torch.sparse_coo_tensor(
                e2.indices(), normalized_values, e2.size(), device=e2.device
            )

            # スパース×密の行列積
            similarity = torch.sparse.mm(e2_norm, e1_norm.transpose(0, 1))
            return similarity.transpose(0, 1)
        else:
            e2_norm = torch.nn.functional.normalize(e2, p=2, dim=1)
            return torch.mm(e1_norm, e2_norm.transpose(0, 1))

    @staticmethod
    def manhattan_distance(e1: Tensor, e2: Tensor) -> Tensor:
        """
        マンハッタン距離を計算します。e2 がスパーステンソルの場合、密テンソルに変換して計算します。
        注意: スパーステンソルを密テンソルに変換するとメモリ消費が増加します。
        """
        if e2.is_sparse:
            e2 = e2.coalesce().to_dense()
        # マンハッタン距離の計算
        x = e1.unsqueeze(1)  # (N, 1, D)
        y = e2.unsqueeze(0)  # (1, M, D)
        manhattan_dist = (x - y).abs().sum(dim=-1)  # (N, M)
        return 100 / (manhattan_dist + 1e-4)

    @staticmethod
    def euclidean_distance(e1: Tensor, e2: Tensor) -> Tensor:
        """
        ユークリッド距離を計算します。e2 がスパーステンソルの場合、密テンソルに変換して計算します。
        注意: スパーステンソルを密テンソルに変換するとメモリ消費が増加します。
        """
        if e2.is_sparse:
            e2 = e2.coalesce().to_dense()
        # ユークリッド距離の計算
        euclidean_dist = torch.cdist(e1, e2)  # (N, M)
        return 100 / (euclidean_dist + 1e-4)

    @staticmethod
    def dot_score(e1: Tensor, e2: Tensor) -> Tensor:
        """
        ドット積を計算します。e2 がスパーステンソルの場合に対応しています。
        """
        if e2.is_sparse:
            # スパーステンソルをコアリッシュ
            e2 = e2.coalesce()
            return torch.sparse.mm(e2, e1.transpose(0, 1)).transpose(0, 1)
        else:
            return torch.mm(e1, e2.transpose(0, 1))