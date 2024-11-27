from __future__ import annotations

from os import PathLike
from pathlib import Path
from collections import Counter
import math

import numpy as np
import torch
import tqdm
from loguru import logger
from scipy.sparse import load_npz, save_npz, vstack
from scipy.sparse.csr import csr_matrix
import tinysegmenter
from typing import Callable


from jmteb.embedders.base import TextEmbedder

def tokenize_with_tinysegmenter(text: str) -> list[str]:
    return tinysegmenter.tokenize(text)

def tokenize_with_ngrams(text: str, ngram_range: tuple[int, int] = (3, 3)) -> list[str]:
    """
    指定された範囲のn-gramを生成する関数

    :param text: 入力テキスト
    :param ngram_range: n-gramの範囲 (start, end)
    :return: n-gramのリスト
    """
    ngrams = []
    for i in range(ngram_range[0], ngram_range[1] + 1):
        ngrams.extend([''.join(text[j:j+i]) for j in range(len(text)-i+1)])
    return ngrams


class BM25Vectorizer:
    # rank-bm25 の実装を参考にした
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25, tokenizer: Callable[[list[str]], list[list[str]]] | None = None):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = tokenizer
    
    def _tokenize_corpus(self, corpus: list[str]) -> list[list[str]]:
        return [tokenize_with_ngrams(text) for text in corpus]
            
    def fit(self, corpus: list[list[str]] | list[str],
            initialize: bool = True,
            batch_size: int = 10**4,
            show_progress_bar: bool = False):
        if initialize:
            self.corpus_size = 0
            self.doc_len = 0
            self.avgdl = 0
            self.idf = {}
            self.nd: Counter[str] = Counter()
        
        with tqdm.tqdm(total=len(corpus), desc="Fitting BM25") as pbar:
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i : i + batch_size]
                if self.tokenizer is not None:
                    batch = self._tokenize_corpus(batch)
                self._calc_nd(batch)
                pbar.update(len(batch))
        
        self._calc_idf(self.nd)        

        self.vocabulary = list(self.idf.keys())
        self.word_to_id = {word: i for i, word in enumerate(self.vocabulary)}
        self.idf_array = np.array([self.idf[word] for word in self.vocabulary])  # idfをNumPy配列として保持

    def _calc_nd(self, corpus: list[list[str]]):
        for document in corpus:
            self.nd.update(set(document))
        
        self.doc_len += sum(len(document) for document in corpus)
        self.corpus_size += len(corpus)
        self.avgdl = self.doc_len / self.corpus_size

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def transform(self, queries: list[list[str]] | list[str], show_progress_bar: bool = False) -> csr_matrix:
        
        rows = []
        cols = []
        data = []
        
        if self.tokenizer is not None:
            queries = self._tokenize_corpus(queries)
        enumerate_queries = tqdm.tqdm(enumerate(queries)) if show_progress_bar else enumerate(queries)
        
        for i, query in enumerate_queries:
            query_len = len(query)
            query_count = Counter(query)
            
            for word, count in query_count.items():
                if word in self.word_to_id:
                    word_id = self.word_to_id[word]
                    tf = count
                    idf = self.idf.get(word, 0)
                    
                    # BM25 scoring formula
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * query_len / self.avgdl)
                    
                    score = numerator / denominator
                    
                    rows.append(i)
                    cols.append(word_id)
                    data.append(score)
        
        return csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocabulary)))
                    
    def count_transform(self, queries: list[list[str]] | list[str], show_progress_bar: bool = False) -> csr_matrix:
        
        rows = []
        cols = []
        data = []
        
        if self.tokenizer is not None:
            queries = self._tokenize_corpus(queries)
        enumerate_queries = tqdm.tqdm(enumerate(queries)) if show_progress_bar else enumerate(queries)
        
        for i, query in enumerate_queries:
            for word in query:
                if word in self.word_to_id:
                    word_id = self.word_to_id[word]
                    rows.append(i)
                    cols.append(word_id)
                    data.append(1)  # Count is always 1 for each occurrence
        
        return csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocabulary)))



class BM25Embedder(TextEmbedder):
    """BM25 embedder."""

    def __init__(
        self,
        #model_name_or_path: str,
        batch_size: int = 10**5,
        #device: str | None = None,
        max_seq_length: int | None = 10**6,
        model_kwargs: dict | None = None,
        # tokenizer_kwargs: dict | None = None,
        use_count_vector_for_query: bool = False
    ) -> None:
        self.model_kwargs = model_kwargs or {}
        
        self.tokenize = tokenize_with_ngrams
        self.model = BM25Vectorizer(tokenizer=self.tokenize, **self.model_kwargs)
        

        self.batch_size = batch_size
        #self.device = device
        self.max_seq_length = max_seq_length
        self.is_sparse_model = True
        self.convert_to_tensor = False
        self.convert_to_numpy = False
        
        self.use_count_vector_for_query = use_count_vector_for_query

    def fit(self, corpus: list[str]):
        self.model.fit(corpus, initialize=True, batch_size=self.batch_size)
    
    def encode(
        self,
        text: str | list[str],
        prefix: str | None = None,
        show_progress_bar: bool = False,
        transform_to_count_vector: bool = False,
        
    ) -> csr_matrix | torch.Tensor:
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
        if prefix:
            text = [prefix + t for t in text]
        if self.max_seq_length is not None:
            text = [t[:self.max_seq_length] for t in text]
        transform_func = self.model.count_transform if transform_to_count_vector else self.model.transform
        embeddings = transform_func(text, show_progress_bar=show_progress_bar)
        if is_single_text:
            embeddings = embeddings[:1]  # csr_matrix の場合、単一行を取得

        if self.convert_to_tensor:
            embeddings = torch.Tensor(
                embeddings.toarray()  # type: ignore
            )  # csr_matrix を Tensor に変換

        return embeddings  # csr_matrix または torch.Tensor

    def get_output_dim(self) -> int:
        return self.model.vocab_size

    def _batch_encode_and_save_on_disk(
        self,
        text_list: list[str],
        save_path: str | PathLike[str],
        prefix: str | None = None,
        batch_size: int = 64,
        dtype: str = "float32",
        transform_to_count_vector: bool = False,
    ) -> csr_matrix | torch.Tensor:
        """
        テキストのリストをエンコードし、memmap または csr_matrix を使用してディスクに保存します。

        Args:
            text_list (list[str]): テキストのリスト
            save_path (str): 埋め込みを保存するパス
            prefix (str, optional): エンコード時に使用するプレフィックス。デフォルトは None。
            dtype (str, optional): データ型。デフォルトは "float32"。
            batch_size (int): バッチサイズ。デフォルトは 64。
        """

        num_samples = len(text_list)
        output_dim = self.get_output_dim()

        if self.is_sparse_model:
            embeddings = []
        elif self.convert_to_numpy:
            embeddings = np.memmap(
                save_path, dtype=dtype, mode="w+", shape=(num_samples, output_dim)
            )
        else:
            embeddings = torch.empty(
                (num_samples, output_dim),
                dtype=self._torch_dtype_parser(dtype),  # type: ignore
            )

        with tqdm.tqdm(total=num_samples, desc=f"Encoding / bs {batch_size}") as pbar:
            for i in range(0, num_samples, batch_size):
                batch = text_list[i : i + batch_size]
                batch_embeddings: csr_matrix | torch.Tensor = self.encode(
                    batch, prefix=prefix, transform_to_count_vector=transform_to_count_vector,
                    show_progress_bar=True
                )
                if self.is_sparse_model:
                    embeddings.append(batch_embeddings)  # type: ignore
                elif self.convert_to_numpy:
                    embeddings[i : i + batch_size] = batch_embeddings
                else:
                    embeddings[i : i + batch_size] = batch_embeddings
                pbar.update(len(batch))

        if self.is_sparse_model:
            # csr_matrix を縦方向に連結
            concatenated_embeddings = vstack(embeddings)
            save_npz(save_path, concatenated_embeddings)
            return concatenated_embeddings
        elif self.convert_to_numpy:
            embeddings.flush()  # type: ignore
            return np.memmap(
                save_path, dtype=dtype, mode="r", shape=(num_samples, output_dim)
            )
        else:
            torch.save(embeddings, save_path)
            return embeddings

    def batch_encode_with_cache(
        self,
        text_list: list[str],
        prefix: str | None = None,
        cache_path: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
        dtype: str = "float32",
        transform_to_count_vector: bool = False,
    ) -> csr_matrix | torch.Tensor:
        """
        テキストのリストをエンコードし、cache_path が指定されている場合は memmap または csr_matrix を使用してディスクに保存します。

        Args:
            text_list (list[str]): テキストのリスト
            prefix (str, optional): エンコード時に使用するプレフィックス。デフォルトは None。
            cache_path (str, optional): 埋め込みを保存するパス。デフォルトは None。
            overwrite_cache (bool, optional): キャッシュを上書きするかどうか。デフォルトは False。
            dtype (str, optional): データ型。デフォルトは "float32"。
        """

        if (
            self.is_sparse_model
            and cache_path is not None
            and not str(cache_path).endswith(".npz")
        ):
            cache_path = str(cache_path) + ".npz"

        if cache_path is None:
            logger.info(f"Encoding embeddings {transform_to_count_vector=}")
            return self.encode(text_list, prefix=prefix, transform_to_count_vector=transform_to_count_vector)

        cache_path = Path(cache_path)

        if cache_path.exists() and not overwrite_cache:
            logger.info(f"Loading embeddings from {cache_path}")
            if self.is_sparse_model:
                # scipy の load_npz を使用して csr_matrix を読み込む
                return load_npz(cache_path)
            elif self.convert_to_numpy:
                return np.memmap(
                    cache_path,
                    dtype=dtype,
                    mode="r",
                    shape=(len(text_list), self.get_output_dim()),
                )
            else:
                return torch.load(cache_path)

        logger.info(f"Encoding and saving embeddings to {cache_path}")
        embeddings = self._batch_encode_and_save_on_disk(
            text_list,
            cache_path,
            prefix=prefix,
            batch_size=self._chunk_size,
            dtype=dtype,
            transform_to_count_vector=transform_to_count_vector,
        )

        return embeddings
    
    def batch_encode_query_with_cache(self, *args, **kwargs):
        return self.batch_encode_with_cache(*args, **kwargs, transform_to_count_vector=self.use_count_vector_for_query)
    
    def batch_encode_doc_with_cache(self, *args, **kwargs):
        return self.batch_encode_with_cache(*args, **kwargs, transform_to_count_vector=False)
