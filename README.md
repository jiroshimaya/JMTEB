# このフォークリポジトリについて
JMTEBのRetrievalタスクをBM25で評価する機能を追加しました。
本家同様に`poetry install`を実行後、以下のようにしてBM25による評価を実行できます。

```bash
poetry run python -m jmteb \
  --embedder BM25Embedder \
  --embedder.use_count_vector_for_query true \ # queryを単語カウントベクトルにするときはtrue、重みベクトルにするときはfalse
  --evaluators src/jmteb/configs/tasks/jagovfaqs_22k.jsonnet
```
## 結果
| **Dataset**               | **Metric** | **Metric Value** | **Val Scores (Accuracy@1,3,5,10)**                          | **Test Scores (Accuracy@1,3,5,10)**                        | **Additional Metrics (Val: NDCG@10, MRR@10 / Test: NDCG@10, MRR@10)**                |
|---------------------------|------------|------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| jagovfaqs_22k             | NDCG@10    | 0.5319           | Val: 41.47, 55.28, 61.95, 67.59                                    | Test: 40.26, 55.35, 60.50, 66.64                                    | Val: 0.5416, 0.499 / Test: 0.5319, 0.489                                                      |
| jaqket                    | NDCG@10    | 0.5981           | Val: 48.14, 61.81, 68.44, 74.57                                    | Test: 46.74, 62.29, 67.00, 73.32                                    | Val: 0.6090, 0.566 / Test: 0.5981, 0.555                                                      |
| nlp_journal_abs_intro     | NDCG@10    | 0.9839           | Val: 100, 100, 100, 100                                            | Test: 96.78, 99.26, 99.50, 99.50                                    | Val: 1.0000, 1.000 / Test: 0.9839, 0.980                                                      |
| nlp_journal_title_abs     | NDCG@10    | 0.9388           | Val: 93.00, 97.00, 99.00, 99.00                                    | Test: 89.11, 95.30, 96.29, 98.27                                    | Val: 0.9634, 0.955 / Test: 0.9388, 0.925                                                      |
| nlp_journal_title_intro   | NDCG@10    | 0.9038           | Val: 92.00, 94.00, 94.00, 95.00                                    | Test: 83.17, 92.57, 95.30, 96.53                                    | Val: 0.9349, 0.930 / Test: 0.9038, 0.883                                                      |

mrtydiについてはデータ件数が多く、現状実装を手元のPCで完了させることができなかったため、上記表に含まれていません。
## ToDo

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

