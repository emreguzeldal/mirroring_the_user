# Mirroring the User: Decomposing Register Accommodation in LLM Responses

Replication code for *"Mirroring the User: Decomposing Apparent Register Accommodation in LLM Responses,"* a study of register accommodation in real human-LLM conversations using the WildChat-1M corpus.

This work was conducted in the scope of CSSM 530 (Automated Text Processing for Social Sciences, Spring 2026) at Koç University, instructed by Dr. Ali Hürriyetoğlu, with Dr. Merih Angın (Department of International Relations, Koç University) as the domain expert.

## What the study does

We use 86,463 turn-level observations from 19,999 multi-turn English conversations in WildChat-1M to ask: of the apparent mirroring observable in cross-sectional human-LLM data, how much reflects genuine within-conversation accommodation by the model, and how much reflects between-conversation selection (chatty users initiating chatty conversations)? Using parallel lexicon-based composites and a within-conversation fixed-effects design, we decompose apparent mirroring into approximately 40% selection and the larger remainder genuine within-conversation responsiveness.

## Repository contents

```
.
├── measurement.py          # Lexicon-based feature extraction and composite construction
├── run_production.py       # Data pull + feature extraction pipeline
├── run_regressions.py      # All four regression specifications + diagnostics
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Reproducing the results

### Environment setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tested on Python 3.10+. The pipeline runs on CPU; no GPU required.

### Data pull and feature extraction

```bash
python run_production.py --target 20000 --out ./out
```

Outputs:
- `./out/wildchat_raw.parquet` — turn-level rows after filtering
- `./out/multiturn_features.parquet` — features and composites, ready for analysis

Approximate runtime: 60–90 minutes on an 8-core laptop. The pipeline streams from Hugging Face, filters to multi-turn English conversations meeting length and content criteria, extracts lexicon-based features (pronouns, politeness markers, relational verbs, hedges, affect words, Empath social-cluster categories, structural features) on prompts and responses, and constructs the framing and register composites.

### Regressions and diagnostics

```bash
python run_regressions.py --in ./out/multiturn_features.parquet --out ./results
```

Outputs:
- `regression_summary.csv` — headline framing coefficient under all four specifications
- `coef_A_pooled.csv`, `coef_B_topicFE.csv`, `coef_C_convFE.csv`, `coef_D_convFE_lag.csv` — full coefficient tables per specification
- `variance_decomposition.csv` — between- vs within-group variance shares for both composites
- `topic_diag.csv` — silhouette scores over candidate K for topic clustering
- `per_component_framing.csv` — register breakdown by individual component under conversation FE

Approximate runtime: 5–10 minutes.

## Specifications estimated

The headline regression is estimated under four progressively stronger specifications:

| Specification | What it controls for | Identification |
|---|---|---|
| (A) Pooled OLS, HC3 | Token counts, turn index, model version | Cross-sectional |
| (B) Topic FE, HC3 | + 50 topic clusters | Within-topic |
| (C) Conversation FE, cluster-robust | + all time-invariant conversation properties | Within-conversation |
| (D) Conv FE + lag, cluster-robust | + lagged register | Autoregressive control |

The selection-vs-accommodation decomposition is operationalized as the difference between (A) and (C). The lag specification (D) tests whether the within-conversation responsive component is autoregressive (model maintains its prior register) or current-input-driven (model weights current user framing).

## Data source

WildChat-1M is publicly available on Hugging Face: <https://huggingface.co/datasets/allenai/WildChat-1M>. The corpus contains approximately one million conversations between users and OpenAI's ChatGPT (gpt-3.5-turbo-0301 and gpt-4-0314), collected via a chatbot interface where users consented to data release for research. Cite as:

> Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024). WildChat: 1M ChatGPT interaction logs in the wild. arXiv:2405.01470.

## License

Code in this repository is released under the MIT License. The WildChat-1M corpus is governed by its own data license; see the dataset card on Hugging Face for terms.

## Citation

If you use this code or find the analysis useful, please cite the paper (placeholder pending preprint):

```
@misc{guzeldal2026mirroring,
  author       = {Güzeldal, Emre},
  title        = {Mirroring the User: Decomposing Apparent Register Accommodation in LLM Responses},
  year         = {2026},
  note         = {Course paper, CSSM 530, Koç University}
}
```

## Contact

Emre Güzeldal — Koç University
[your email]
