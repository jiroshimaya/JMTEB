from __future__ import annotations

from os import PathLike
from pathlib import Path
from collections import Counter
import concurrent.futures

import numpy as np
import torch
import tqdm
from loguru import logger
from scipy.sparse import load_npz, save_npz, vstack
from scipy.sparse.csr import csr_matrix
from rank_bm25 import BM25Okapi
import tinysegmenter


from jmteb.embedders.base import TextEmbedder

class BM25Vectorizer(BM25Okapi):
    def __init__(self, corpus: list[list[str]], **bm25_params):            
        super().__init__(corpus, **bm25_params, tokenizer=None) # tokenizeは使わない
        self.vocabulary = list(self.idf.keys())
        self.word_to_id = {word: i for i, word in enumerate(self.vocabulary)}
        self.idf_array = np.array([self.idf[word] for word in self.vocabulary])  # idfをNumPy配列として保持

    #override
    def _initialize(self, corpus: list[list[str]]):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in tqdm.tqdm(corpus, desc="Initializing BM25"):
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        self.nd = nd # add this line
        return nd
    def transform(self, queries: list[list[str]], show_progress_bar: bool = False) -> csr_matrix:
        # クエリを頻度ベクトルに変換
        query_matrix = self.count_transform(queries, show_progress_bar=show_progress_bar)

        # 各クエリの長さを計算
        query_lengths = np.array(query_matrix.sum(axis=1)).flatten()

        # BM25スコアの計算
        tf = query_matrix
        idf = self.idf_array

        # スパース行列の要素ごとの演算
        numerator = tf.multiply(idf * (self.k1 + 1))
        denominator = tf + self.k1 * (1 - self.b + self.b * query_lengths[:, None] / self.avgdl)

        scores = numerator.multiply(1 / denominator)
        
        scores = csr_matrix(scores)

        return scores
            
    def count_transform(self, queries: list[list[str]], show_progress_bar: bool = False) -> csr_matrix:
        
        rows = []
        cols = []
        data = []
        
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
        batch_size: int = 10**4,
        #device: str | None = None,
        max_seq_length: int | None = 10**6,
        model_kwargs: dict | None = None,
        # tokenizer_kwargs: dict | None = None,
        use_count_vector_for_query: bool = False
    ) -> None:
        self.model_kwargs = model_kwargs or {}
        
        def tokenize(x, show_progress_bar: bool = False):
            if show_progress_bar:
                x = tqdm.tqdm(x, desc="Tokenizing")
            return [tinysegmenter.tokenize(text) for text in x]
        
        # 並列処理を試みているがうまくいっていない
        def _tokenize(x, show_progress_bar: bool = False, timeout: int = 3600):
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # futures のリストを作成
                futures = []
                for text in x:
                    future = executor.submit(tinysegmenter.tokenize, text)
                    futures.append(future)
                
                tokenized = []
                try:
                    # 完了したフューチャーを追跡
                    completed = 0
                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(futures, timeout=timeout),
                        total=len(futures),
                        desc="Tokenizing"
                    ):
                        tokenized.append(future.result())
                        completed += 1
                        
                except concurrent.futures.TimeoutError:
                    remaining = len(futures) - completed
                    logger.warning(f"Tokenization timed out after {timeout} seconds")
                    logger.info(f"Completed: {completed}/{len(futures)} texts")
                    logger.info(f"Remaining: {remaining} texts")
                    
                    # 未完了のタスクをキャンセル
                    logger.info("Cancelling remaining tasks...")
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    
                    # プロセスプールを強制終了
                    logger.info("Shutting down process pool...")
                    executor.shutdown(wait=False)
                    
                    raise TimeoutError(f"Tokenization took longer than {timeout} seconds")
                
                finally:
                    logger.info("Cleanup completed")
            
            logger.info(f"Tokenized {len(tokenized)} tokens")
            return tokenized
        self.tokenize = tokenize
        

        self.batch_size = batch_size
        #self.device = device
        self.max_seq_length = max_seq_length
        self.is_sparse_model = True
        self.convert_to_tensor = False
        self.convert_to_numpy = False
        
        self.use_count_vector_for_query = use_count_vector_for_query

    def fit(self, corpus: list[str]):
        tokenized_corpus = self.tokenize(corpus, show_progress_bar=True)
        self.model = BM25Vectorizer(tokenized_corpus, **self.model_kwargs)
    
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
        tokenized_text = self.tokenize(text, show_progress_bar=show_progress_bar)
        if self.max_seq_length is not None:
            tokenized_text = [t[:self.max_seq_length] for t in tokenized_text]
        transform_func = self.model.count_transform if transform_to_count_vector else self.model.transform
        embeddings = transform_func(tokenized_text, show_progress_bar=show_progress_bar)
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
