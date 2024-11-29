# このフォークリポジトリについて
JMTEBのRetrievalタスクをBM25やTFIDFで評価する機能を追加しました。
本家同様に`poetry install`を実行後、以下のようにして評価を実行できます。

- BM25
```bash
poetry run python -m jmteb \
  --embedder BM25Embedder \ 
  --embedder.use_count_vector_for_query true \ # queryを単語カウントベクトルにするときはtrue、重みベクトルにするときはfalse
  --evaluators src/jmteb/configs/tasks/jagovfaqs_22k.jsonnet
```

- TFIDF
```bash
poetry run python -m jmteb \
  --embedder TfidfEmbedder \ 
  --evaluators src/jmteb/configs/tasks/jagovfaqs_22k.jsonnet
```

## 結果

|Model|Avg.|jagovfaqs_22k<br>(ndcg@10)|jaqket<br>(ndcg@10)|mrtydi<br>(ndcg@10)|nlp_journal_abs_intro<br>(ndcg@10)|nlp_journal_title_abs<br>(ndcg@10)|nlp_journal_title_intro<br>(ndcg@10)|
|---|---|---|---|---|---|---|---|
|count-bm25-dot-product|NA|0.5706|0.5976|NA|0.9915|0.9484|0.9485|
|bm25-bm25-dot-product|NA|0.5875|0.5796|NA|0.9944|0.9506|0.9511|
|tfidf-tfidf-cosine-sim|NA|0.5347|0.2920|NA|0.9824|0.9305|0.91012|

- model名は「クエリベクトルの種類-文書ベクトルの種類-類似度指標」の規則になっています。例えば「count-bm25-dot-product」はクエリのカウントベクトルと文書のBM25重みベクトルの内積、ということです。
- tokenizeは高速化のため形態素解析は使わずトリグラムで実施しています。
- mrtydiについてはデータ件数が多く、現状実装を手元のPCで完了させることができなかったため、上記表に含まれていません。
## ToDo
- mrtydiのスコア計算

以降は本家リポジトリのREADMEです。
# JMTEB: Japanese Massive Text Embedding Benchmark

<h4 align="center">
    <p>
        <b>README</b> |
        <a href="./leaderboard.md">leaderboard</a> |
        <a href="./submission.md">submission guideline</a>
    </p>
</h4>

[JMTEB](https://huggingface.co/datasets/sbintuitions/JMTEB) is a benchmark for evaluating Japanese text embedding models. It consists of 5 tasks.

This is an easy-to-use evaluation script designed for JMTEB evaluation.

JMTEB leaderboard is [here](leaderboard.md). If you would like to submit your model, please refer to the [submission guideline](submission.md).

## Quick start

```bash
git clone git@github.com:sbintuitions/JMTEB
cd JMTEB
poetry install
poetry run pytest tests
```

The following command evaluate the specified model on the all the tasks in JMTEB.

```bash
poetry run python -m jmteb \
  --embedder SentenceBertEmbedder \
  --embedder.model_name_or_path "<model_name_or_path>" \
  --save_dir "output/<model_name_or_path>"
```

> [!NOTE]
> In order to gurantee the robustness of evaluation, a validation dataset is mandatorily required for hyperparameter tuning.
> For a dataset that doesn't have a validation set, we set the validation set the same as the test set.

By default, the evaluation tasks are read from `src/jmteb/configs/jmteb.jsonnet`.
If you want to evaluate the model on a specific task, you can specify the task via `--evaluators` option with the task config.

```bash
poetry run python -m jmteb \
  --evaluators "src/configs/tasks/jsts.jsonnet" \
  --embedder SentenceBertEmbedder \
  --embedder.model_name_or_path "<model_name_or_path>" \
  --save_dir "output/<model_name_or_path>"
```

> [!NOTE]
> Some tasks (e.g., AmazonReviewClassification in classification, JAQKET and Mr.TyDi-ja in retrieval, esci in reranking) are time-consuming and memory-consuming. Heavy retrieval tasks take hours to encode the large corpus, and use much memory for the storage of such vectors. If you want to exclude them, add `--eval_exclude "['amazon_review_classification', 'mrtydi', 'jaqket', 'esci']"`. Similarly, you can also use `--eval_include` to include only evaluation datasets you want.

> [!NOTE]
> If you want to log model predictions to further analyze the performance of your model, you may want to use `--log_predictions true` to enable all evaluators to log predictions. It is also available to set whether to log in the config of evaluators.

## Multi-GPU support

There are two ways to enable multi-GPU evaluation.

* New class `DataParallelSentenceBertEmbedder` ([here](src/jmteb/embedders/data_parallel_sbert_embedder.py)).

```bash
poetry run python -m jmteb \
  --evaluators "src/configs/tasks/jsts.jsonnet" \
  --embedder DataParallelSentenceBertEmbedder \
  --embedder.model_name_or_path "<model_name_or_path>" \
  --save_dir "output/<model_name_or_path>"
```

* With `torchrun`, multi-GPU in [`TransformersEmbedder`](src/jmteb/embedders/transformers_embedder.py) is available. For example,

```bash
MODEL_NAME=<model_name_or_path>
MODEL_KWARGS="\{\'torch_dtype\':\'torch.bfloat16\'\}"
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    src/jmteb/__main__.py --embedder TransformersEmbedder \
    --embedder.model_name_or_path ${MODEL_NAME} \
    --embedder.pooling_mode cls \
    --embedder.batch_size 4096 \
    --embedder.model_kwargs ${MODEL_KWARGS} \
    --embedder.max_seq_length 512 \
    --save_dir "output/${MODEL_NAME}" \
    --evaluators src/jmteb/configs/jmteb.jsonnet
```

Note that the batch size here is global batch size (`per_device_batch_size` × `n_gpu`).

