# ChatGDE

Code and data for our paper on automating the review of California Groundwater Sustainability Plans (GSPs) using large language models.

## What this is

Under California's Sustainable Groundwater Management Act (SGMA), Groundwater Sustainability Agencies must submit GSPs for state review — a process that involves manually scoring each plan against a detailed rubric. This project tests whether LLMs can do that scoring reliably enough to assist human reviewers.

The pipeline retrieves the most relevant chunks from each GSP PDF using hybrid BM25 + cosine retrieval, reranks them with a fine-tuned cross-encoder, and passes the top 15 chunks to an LLM to answer each rubric question (Yes / Somewhat / No + confidence).

We evaluated nine model configurations across five trial GSPs: GPT-3.5 FT, GPT-4o (base and fine-tuned), GPT-4.1 (base and fine-tuned), GPT-5.5, OpenAI o3, Claude Sonnet 4.6, and Claude Opus 4.7 (vision). The best model (fine-tuned GPT-4.1 v4) was then run on all 62 high- and medium-priority California GSPs.

Best binary accuracy on trial GSPs: **75.9%** (GPT-4.1 FT v4). Among base models, Claude Sonnet 4.6 and o3 both reach **75.5%** without any fine-tuning.

## Repo structure

```
GSP_Drafts/
  Rubrics/       scoring rubric CSVs for all 65 basins
  *.pdf          trial GSP PDFs (Big Valley, Butte, ECC, Fillmore, Sonoma, SLO)

code final/
  run_*.py       eval scripts for each model
  make_*_fig.py  figure generation scripts
  GSP_All.ipynb  main analysis notebook

results/
  results_*.csv                    per-question model outputs, all 9 models
  checkpoint_*.json                resumable checkpoints
  gpt41ft_allgsps_per_gsp_metrics.csv   per-GSP accuracy + AUC for all 62 GSPs

images/
  roc_prc_comparison.png
  binary_accuracy_{overall,by_gsp}.png
  class_accuracy_{overall,by_gsp}.png
  class_recall_comparison.png
  confusion_matrices.png
```

## Results summary

| Model | Binary Acc. | ROC AUC | PRC AUC |
|---|---|---|---|
| GPT-4o (base) | 62.7% | 0.711 | 0.590 |
| GPT-3.5 FT | 72.8% | 0.693 | 0.631 |
| GPT-4o FT | 73.1% | 0.770 | 0.696 |
| GPT-4.1 (base) | 73.0% | 0.759 | 0.639 |
| GPT-5.5 (base) | 71.8% | 0.723 | 0.632 |
| o3 + Reranker | 75.5% | 0.762 | 0.636 |
| Claude Sonnet 4.6 | 75.5% | 0.749 | 0.639 |
| Claude Opus 4.7 (vision) | 73.9% | 0.736 | 0.616 |
| **GPT-4.1 FT v4** | **75.9%** | **0.773** | **0.711** |

Trial GSPs (n=241 after excluding NotApplicable rows). Binary = Yes vs. {Somewhat + No}.

GPT-4.1 FT v4 applied to all 62 California GSPs: **75.6% accuracy** (57 non-trial GSPs), mean ROC AUC = 0.767.

## Notes

API keys are not included. Scripts expect `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` as environment variables. Large embedding caches, model weights, and most GSP PDFs are excluded from this repo via `.gitignore` — see the paper for the full data pipeline.
