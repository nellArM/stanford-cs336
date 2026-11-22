# Stanford CS336 (Spring 2026) — Language Modeling from Scratch: Assignments 1–5

# 史丹佛 CS336（2026 春季）—「從零開始的語言模型」作業 1–5

This folder contains a self-study solution to the five CS336 assignments. Each assignment has its own
folder with the same layout:

本資料夾為 CS336 五份作業的自學解答。每份作業各自一個資料夾，結構相同：

| File 檔案 | Content 內容 |
|---|---|
| `README.md` | What the assignment covers, what was implemented, results, deviations, how to reproduce（英文說明：作業內容、實作項目、結果、與原始要求的差異、重現方式） |
| `writeup.pdf` | English report: written answers, figures, experiment results（英文報告：書面題解答、圖表、實驗結果） |
| `code.zip` | Archive produced by the assignment's own submission script（由作業本身的打包腳本產生的壓縮檔） |
| `code/` | The same source tree, unpacked for browsing, plus small result files（同一份程式碼的展開版本，另附小型結果檔） |

Datasets, model checkpoints and the handout PDFs are not included.
資料集、模型檢查點與作業說明 PDF 均未包含在內。

---

## Important notes / 重要說明

**English.**
- This is a *self-study* solution; I am not enrolled in the course. Nothing was submitted to any course
  leaderboard, API or grader.
- The handouts assume Stanford's cluster (B200/H100 GPUs, Modal, a Stanford-only training API). The work
  here was done on **two shared RTX A6000 (48 GB) GPUs**. All code, tests and written questions are
  complete, but **every experiment was run at a reduced budget**, and a few items could not be run as
  specified. Each report states its deviations explicitly; the main ones are listed per assignment below.
- Wall-clock timings were measured on GPUs shared with other jobs and are noisy.
- The optional supplement of assignment 5 (safety / instruction fine-tuning) was skipped.

**中文。**
- 本專案為*自學*用途，本人並未修課；沒有向任何課程排行榜、API 或評分系統提交任何內容。
- 作業說明假設使用史丹佛的運算叢集（B200/H100 GPU、Modal、僅限校內的訓練 API）。本專案在
  **兩張共用的 RTX A6000（48 GB）GPU** 上完成。所有程式、測試與書面題皆完整作答，但
  **所有實驗都以縮小的規模執行**，少數項目無法依原規格執行。每份報告都明確列出與原始要求的差異，
  主要差異摘要於下方各作業說明中。
- 執行時間是在與其他工作共用的 GPU 上量測，數值有雜訊。
- 作業 5 的選做補充部分（安全性／指令微調）未做。

---

## Assignment 1 — Basics / 作業 1 — 基礎：從零打造 Transformer 語言模型

**Goal (English).** Build every component needed to train a Transformer language model without using
PyTorch's built-in layers or optimizers: a byte-pair-encoding (BPE) tokenizer, the model itself (RMSNorm,
SwiGLU, rotary position embeddings, causal multi-head attention), cross-entropy, AdamW, a learning-rate
schedule, a training loop and a text decoder; then train on TinyStories and OpenWebText and study
learning rate, batch size and architecture ablations.

**目標（中文）。** 不使用 PyTorch 內建的網路層與優化器，親手實作訓練 Transformer 語言模型所需的全部元件：
BPE 分詞器、模型本體（RMSNorm、SwiGLU、旋轉位置編碼 RoPE、因果多頭注意力）、交叉熵、AdamW、
學習率排程、訓練迴圈與文字解碼器；接著在 TinyStories 與 OpenWebText 上訓練，並研究學習率、
批次大小與架構消融實驗的影響。

**Result / 結果.** Tests: 47 passed, 1 expected failure. TinyStories validation loss **1.329** (target
≤ 1.45); OpenWebText 3.932; local 45-minute run 3.920.
測試 47 項通過（另有 1 項為預期失敗）；TinyStories 驗證損失 **1.329**（目標 ≤ 1.45）；OpenWebText 3.932；
本機 45 分鐘限時訓練 3.920。

**Scaled down / 縮小之處.** Sweeps and ablations use the handout's 40M-token low-resource budget; one
A6000 instead of a B200; the leaderboard run was local only.
超參數掃描與消融實驗採用作業說明中的 4 千萬 token 低資源預算；以一張 A6000 取代 B200；排行榜實驗僅在本機執行。

## Assignment 2 — Systems / 作業 2 — 系統：效能分析、FlashAttention 與分散式訓練

**Goal (English).** Make the model fast and scalable: benchmark and profile the forward/backward passes,
study mixed precision, memory use and gradient checkpointing, implement FlashAttention-2 (in PyTorch and
as a Triton kernel), and implement distributed training — data-parallel training (DDP) with several
communication strategies, a sharded optimizer and fully-sharded data parallelism (FSDP).

**目標（中文）。** 讓模型跑得快、放得大：對前向／反向傳播做基準測試與效能剖析，研究混合精度、
記憶體用量與梯度檢查點，實作 FlashAttention-2（PyTorch 版與 Triton 核心版），並實作分散式訓練——
多種通訊策略的資料平行訓練（DDP）、分片優化器，以及完全分片資料平行（FSDP）。

**Result / 結果.** Tests: 14 passed. Triton FlashAttention forward+backward is 14–20× faster than the
PyTorch baseline at long sequences and runs where the baseline is out of memory; the full xl model
trains under FSDP on 2 GPUs.
測試 14 項全數通過。Triton 版 FlashAttention 在長序列上的前向加反向比 PyTorch 基準快 14–20 倍，
且能在基準版記憶體不足的長度下執行；完整 xl 模型可用 FSDP 在兩張 GPU 上訓練。

**Scaled down / 縮小之處.** xl / 10B training steps do not fit in 48 GB (smaller or layer-reduced models
substituted); all-reduce on 2 GPUs instead of 6; profiler screenshots replaced by tables and plots (no
GUI); a workaround for an NCCL GPU-to-GPU hang makes all distributed timings communication-bound.
xl／10B 模型的訓練步驟放不進 48 GB（改用較小或減少層數的模型）；all-reduce 只用 2 張 GPU 而非 6 張；
無圖形介面，故以表格與圖取代剖析器截圖；為避開本機 NCCL 的 GPU 直連傳輸卡死問題而採用的替代傳輸方式，
使所有分散式實驗的時間皆受通訊速度限制。

## Assignment 3 — Scaling / 作業 3 — 縮放律

**Goal (English).** Learn to predict how model quality scales with compute: reproduce the Chinchilla
IsoFLOP analysis (fit compute-optimal model and data size as power laws of the compute budget), then use
a limited budget of small training runs to fit a scaling law and predict the best configuration, and its
loss, for a budget four times larger.

**目標（中文）。** 學習預測模型品質如何隨運算量縮放：重現 Chinchilla 的 IsoFLOP 分析（把最佳模型大小與
資料量擬合為運算預算的冪次律），再以有限預算進行多個小型訓練，擬合縮放律，並預測在四倍預算下的
最佳模型配置及其損失。

**Result / 結果.** Chinchilla fit: N_opt ∝ C^0.47, D_opt ∝ C^0.53. Local study: predicted loss 3.617 for
the target run, actual **3.465**.
Chinchilla 擬合：N_opt ∝ C^0.47、D_opt ∝ C^0.53。本機研究：對目標訓練的預測損失為 3.617，實際為 **3.465**。

**Scaled down / 縮小之處.** The course's training API is reachable only from Stanford, so the main problem
was replaced by a local study with the same structure (21 scaling runs, then a 4× target run that was
actually trained) at roughly 10^15–10^18 FLOPs.
課程的訓練 API 僅限史丹佛校內使用，因此主要題目改為結構相同的本機研究（21 次縮放實驗，
加上一次實際訓練的 4 倍預算目標實驗），運算量約為 10^15–10^18 FLOPs。

## Assignment 4 — Data / 作業 4 — 資料

**Goal (English).** Turn raw web crawl data into language-model training data: extract text from HTML,
identify language, mask personal information, detect NSFW/toxic content, apply quality heuristics and a
trained quality classifier, remove exact and near duplicates (MinHash + LSH), run the whole pipeline in
parallel over Common Crawl, then tokenize the result and train a language model on it.

**目標（中文）。** 把原始網路爬蟲資料變成語言模型的訓練資料：從 HTML 擷取文字、語言辨識、遮蔽個人資訊、
偵測不雅／惡意內容、套用品質規則與自行訓練的品質分類器、移除完全重複與近似重複的文件（MinHash + LSH），
在 Common Crawl 上平行執行整條流程，最後將結果分詞並用來訓練語言模型。

**Result / 結果.** Tests: 21 passed. 8.55M pages → 706K kept (27.7% of English pages), 709M tokens;
~430M-parameter model reaches validation loss **4.111** on Paloma C4-100-domains.
測試 21 項全數通過。855 萬個網頁 → 保留 70.6 萬（佔英文網頁的 27.7%），共 7.09 億 token；
約 4.3 億參數的模型在 Paloma C4-100-domains 上的驗證損失為 **4.111**。

**Scaled down / 縮小之處.** 400 WET files instead of 2,500; 1,024 training steps instead of 16,384, on
one GPU with gradient accumulation.
使用 400 個 WET 檔而非 2,500 個；訓練 1,024 步而非 16,384 步，並以單張 GPU 搭配梯度累積進行。

## Assignment 5 — Alignment / 作業 5 — 對齊：以強化學習訓練推理能力

**Goal (English).** Improve a language model's math reasoning with reinforcement learning: measure
prompting baselines on GSM8K, implement GRPO (group-relative policy optimization) and related
policy-gradient estimators (Dr. GRPO, MaxRL, off-policy clipping variants), and compare them
experimentally on a 1B-parameter model.

**目標（中文）。** 以強化學習提升語言模型的數學推理能力：在 GSM8K 上量測提示法的基準表現，
實作 GRPO（群組相對策略優化）及相關的策略梯度估計方法（Dr. GRPO、MaxRL、離策略裁剪等變體），
並在 10 億參數的模型上以實驗比較。

**Result / 結果.** Tests: 19 passed. GSM8K prompting baselines: 0.1% zero-shot, 19.3% three-shot.
GRPO on OLMo-2 1B raises validation accuracy from 0.4% to **46%** in 200 steps (target ≥ 25%).
Most variants end at 40–50% (Dr. GRPO, RFT, MaxRL, my own sqrt-MaxRL estimator, off-policy GSPO 47.8%,
off-policy without clipping 50.5%), but seed-to-seed spread is up to 16 points, so they are not ranked.
測試 19 項全數通過。GSM8K 提示法基準：零樣本 0.1%、三樣本 19.3%。GRPO 在 200 步內把 OLMo-2 1B
的驗證準確率從 0.4% 提升到 **46%**（目標 ≥ 25%）。多數變體最終落在 40–50%（Dr. GRPO、RFT、MaxRL、
自行設計的 sqrt-MaxRL 估計器、離策略 GSPO 47.8%、無裁剪離策略 50.5%），但不同隨機種子間差距可達
16 個百分點，因此不對各方法排名。

**Scaled down / 縮小之處.** 200 rollout steps per run with 1–2 seeds; interim evaluation on 256
examples, final evaluation on 1,024; one GPU for vLLM sampling and one for training; learning rate 3e-5
diverged and then ran out of memory; the optional safety/RLHF supplement was skipped.
每次訓練 200 個取樣步、僅 1–2 個隨機種子；中途評估用 256 題、最終評估用 1,024 題；一張 GPU 跑 vLLM
取樣、一張跑訓練；學習率 3e-5 的訓練發散後記憶體不足而中止；選做的安全性／RLHF 補充部分未做。

---

## Licence / 授權

My own code and reports are released under the MIT licence (see `LICENSE`). The course scaffolding
included in the `code/` folders (handout code, tests, vendored staff packages) remains under its original
licence, found in each `code/LICENSE`.

本人撰寫的程式與報告以 MIT 授權釋出（見 `LICENSE`）。`code/` 資料夾內所含的課程原始框架
（作業提供的程式、測試與助教提供的套件）仍適用其原本的授權，見各 `code/LICENSE`。
