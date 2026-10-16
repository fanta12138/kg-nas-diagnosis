# 论文数字硬化审计报告 — main_v2（Applied Sciences 投稿版）

> **稿件**：`G:/submission_KBS/main_v2.tex`（sections in `sections/_draft_v2/`）
> **审计轮次**：`claim-audit_20261016`（零上下文分段审计 → 执行者复验 → 机读改稿 → 重编译）
> **最终裁决**：**WARN**（`corrected_and_reverified`）
> **审稿独立性**：`same-family` —— 零上下文审计者为 `glm-5.3-flash`，与执行者同族；其结论仅作输入，**不作验收**。执行者对每一条发现回到原始 JSON 独立复算后才决定是否采纳。
> **验收状态**：`provisional`（同族审稿，不视为独立验收）

---

## 一、执行摘要

| 项目 | 数值 |
|---|---|
| 审计分段 | 6（S1 摘要/引言/相关工作，S2 方法，S3 代理复现+方差分解，S4 可分辨性+重复搜索，S5 分数策略，S6 讨论+结论）|
| 提取文本断言总数 | **206** |
| 裁决 `exact_match` / `rounding_ok` | 60 |
| 裁决 `missing_evidence` | 106（其中绝大多数为「外部文献值 / 未提供的配置 / 未提供的中间产物」）|
| 裁决 `ambiguous_mapping` | 15 |
| 裁决 `number_mismatch` / `aggregation_mismatch` / `scope_overclaim` / `config_mismatch` / `unsupported_claim` | 25 |
| **执行者复验后确认的真实缺陷** | **5** |
| **执行者复验后判定为误报** | **6** |
| 超出审计范围（外部文献/协议常量/未入包产物） | 3 类 |
| 修复后状态 | 5 条全部改正；重编译 0 error / 0 undefined |

**关键教训**：审计模型在 25 条「异常」中只有 5 条为真，误报率 24%。根因是它**用错了证据文件**——
多次把 `fusion_ksweep/`（第二套脚本、22 个策略）当作 `fusion_policy/`（论文实际来源、82 个策略），
把 `policy_select/`（另一实验的 80/20 划分）当作 `pseudo_select/`（70/30）。
**任何 flagged item 必须回到论文自己声明的来源文件复算后才能采纳。**

---

## 二、确认缺陷与修复（5 条）

| id | 严重度 | 位置 | 论文原述 | 证据 | 问题 | 修复 |
|---|---|---|---|---|---|---|
| **D1** | 高 | `1_introduction.tex:79`、`4_diagnosis.tex:393`、`7_conclusion.tex:22` | 最小可检出差异 **2.98 点 —— 大于**全部 3.98 点极差 | `samplesize_analysis.json`: `mdd_paired_15cat_pp=2.978`，`backbone_spread_pp=3.9795` | **2.98 < 3.98**，「larger than」为算术错误；2.98/3.98 = **74.8%** | 三处均改为「**75% of the entire 3.98-point spread**」 |
| **D2** | 中 | `4_diagnosis.tex:293`、`:304`（Fig.3 caption）| 类目占 95.32%、骨干占 0.54%，**相差 176 倍** | `final_truth_analysis.json`: 95.3212 / 0.54455 = **175.05** | 175.05 应进位为 **175**，不是 176 | 正文与图注 176 → **175** |
| **D3** | 高 | `4_diagnosis.tex:110-112`（表2 caption）、`:122`（表2行）、`:130-133`（正文）、`:151-156`（Fig.2 caption）、`:164-171`（正文）、`7_conclusion.tex:32-33` | **十二个**良性骨干「中位跨度 7.538–7.971、CV **2.2%–12.1%**」 | `phaseC_analysis.json`：按「3 维 CV 取均值」口径，**11 个**骨干跨 2.20%–12.09%；第 12 个是 `efficientnet_b0`，其 **CV = 74.00%**、中位代理 **11.65** | 口径混用：把 b0 算进「十二个」，却只报紧致的 11 个区间 | 全文改为「**eleven tight non-collapsing backbones**」（2.2%–12.1%，跨度 7.54–7.97），b0 单列为 CV 74%；表2 行改为 `2.2--74.0`；5.7% 的表述随之下沉到 11 个骨干 |
| **D4** | 高 | `5_leverage.tex:92-95`、`1_introduction.tex:134` | 「三个种子的聚合策略均值 82.36% / 90.28%」 | `fusion_multiseed_summary.json`: `seeds=[43,44]`、`n_cells=30`、`sum_mean=0.823598`、`sum_topk=0.902848` | 该文件是**两个**种子（43、44）的聚合。真正的三种子合并值（并入 seed-42 的 82.80/90.21）为 **82.51 / 90.26（+7.75）** | 5.2 节同时给出两种口径并说明为何不取平均；引言改为「**from 82.5% to 90.3% — +7.8 points**」 |
| **D5** | 中 | `4_diagnosis.tex:459-461`、`:494-495`、`7_conclusion.tex:34-35` | 「两个易崩骨干」返回 0.0（暗示崩溃**仅限** b1 与 mnv3）；`efficientnet_b0 ... which does not collapse` | `search_repeats.json`（1190 trials）：`efficientnet_b1` 59/72（82%）、`mobilenetv3_small_100` 19/87（22%）、**`efficientnet_b0` 2/79（2.5%，run08 t63、run10 t32）** | b0 并非不会崩；断言「排他性」会支持黑名单而非论文推荐的过滤器 | 补入 b0 的 2/79 计数，把建议重述为「必须是过滤器而非黑名单」；「does not collapse」改为「collapses only rarely and never in those four runs」 |

---

## 三、判定为误报（6 条，**论文原值正确**）

| id | 审计者主张 | 复验结论 |
|---|---|---|
| **F1** | 表 `tab:fusion` 的 top-k 最差类目 78.37 与 `fusion_ksweep` 的 76.46 冲突 | **误报**。`fusion_policy/fusion_policy.json`（论文实际来源，82 个策略）给出 `pill = 78.37` **精确吻合**。`fusion_ksweep` 是第二套脚本的 22 策略子集，76.46 属于另一套网格。 |
| **F2** | 「我们评估 82 个策略」自身枚举得 108+，与 111 冲突 | **误报**。`fusion_policy.json` 恰含 **82** 个策略键，精确复现表中 82.80/42.22/90.21/78.37/90.60/93.72。111 属 `pseudo_select` 实验，写在该表自己的 caption 里。两个数各自正确。 |
| **F3** | 70/30 划分被 `policy_select.json` 的 80/20 推翻 | **误报**。`pseudo_select.json` 及 `_s43`/`_s44` 三个种子下 bottle 均为 `n_fit=146, n_holdout=63`（69.9%/30.1%）。审计者读的是另一实验的 `policy_select.json`。 |
| **F4** | mobilenetv3 排名 7 / 81.85% 与 81.95% 冲突 | **误报**。`final_truth_analysis.json` 给出 `mobilenetv3_small_100 = 0.8185266`，14 个中排第 7。81.85% 精确。审计者误用 `search_repeats_summary.json` 的另一口径。 |
| **F5** | 「210 个检测器」应为 390 | **误报**。`phaseB_truth.json` 共 390 条 = seed42（210 = 14×15）+ seed43/44（各 90）。`final_truth_analysis.json` 的分解用 seed-42 矩阵，`df=[13,182]`，与正文一致。210 与 180 残差自由度各指不同对象，均正确。 |
| **F6** | 「+1.06 个百分点」应为 +1.07（83.08−82.01） | **误报**。表内均值保留 2 位小数；按未取整值算差为 1.06。0.01 pp 属显示取整，非错误。 |

---

## 四、超出审计范围（3 类）

1. **外部文献值**（S1 id 1/12、S6 id 6/16）：PatchCore 99.1%、SimpleNet/DINomaly 99.6%、SPADE/FastFlow 预印本状态、「首个全搜索空间比较」的novelty表述。
   这些是对**已发表记录**的陈述，无法用内部证据文件核实，不属 paper-to-evidence 审计范围。文献条目已单独核查（见 citation audit）。
2. **方法章协议常量**（S2 全部 `missing_evidence`）：epochs / batch size / lr / seeds / k=1% / B=256 等无法追溯到证据，因为 `unsupervised_nas/configs/*.yaml` **未放入本次证据包**。
   **建议后续**：把 config 文件加入证据集后重跑 S2，可获得完全干净的追溯链。**未发现任何矛盾**。
3. **未入包的仓库产物**（S3 多条、S6 全部）：文件存在于发布仓库 `G:/kg-nas-diagnosis/`，只是不在本次压缩证据包内。凡论文引用的，本轮均已直接复算确认。

---

## 五、修复后复验

| 检查 | 方法 | 结果 |
|---|---|---|
| 5 条修正的数值 | 每条都从 `G:/kg-nas-diagnosis/` 原始 JSON 重新计算 | 全部通过 |
| 全文 `2.98` / `176` 残留 | `grep` 全量扫描 | 无残留错误表述 |
| `82.36` / `90.28` 口径 | 与 `fusion_multiseed_summary.json` 逐字段对照 | 已如实标注为 2 种子 |
| 重编译 | `latexmk -pdf main_v2.tex` | 见 `build` 字段 |
| 源文件哈希 | sha256，见 `audited_input_hashes` | 已落盘 |

---

## 六、复现路径

```bash
# 审计驱动
C:/Users/dengpeng/anaconda3/python.exe audit/claim-audit_20261016/run_claim_audit.py
C:/Users/dengpeng/anaconda3/python.exe audit/claim-audit_20261016/rerun_failed.py

# 证据根
G:/kg-nas-diagnosis/{fusion_policy,fusion_ksweep,fusion_multiseed,pseudo_select,proxy_diag,search_repeats}/

# 原始分段报告（含模型原始输出与论文原文对照）
audit/claim-audit_20261016/seg_S{1..6}_*.md
```

**改稿前快照**：`audit/claim-audit_20261016/backup_20261016_132524/`（7 个 section 的 `.tex`）
**上一轮（Springer）报告**：`audit/claim-audit_20261016/prev_springer_round/`
