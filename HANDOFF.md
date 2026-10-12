# SAFE-Gait V2.3 项目交接

更新日期：2026-10-01，Asia/Tokyo。训练停止后的进程快照时间：19:21:19 +09:00。

交接复核：2026-10-01 21:41:52 +09:00 一次性只读核对，九项交接内容均已记录；seed7/17/27 完成标记仍为225/105/0，seed17仍有4个中断和116个待开始任务，未发现仍在运行的Python进程。自动任务v2-3的磁盘状态仍为ACTIVE。未执行NEXT_ACTION、恢复命令、训练或统计分析。算法项目根目录与工作区根目录的HANDOFF.md现均保存完整交接内容。

2026-10-01 21:54:17 +09:00 后续交接操作：按用户指示完成 NEXT_ACTION。磁盘上不再有 `v2-3` 配置；唯一 ACTIVE 的自动化实际 ID 为 `automation`，名称“继续任务的指令”，提示词为“继续未完成的任务。”。已通过 Codex 自动化接口仅将其状态更新为 `PAUSED`，并保留名称、提示词、周期、模型、推理强度、执行环境和 projectless 目标。未启动/恢复训练、统计、bootstrap 或 runner；未读取训练日志正文。

2026-10-01 21:59:05 +09:00 恢复操作：用户已明确授权恢复。恢复前确认未见 Python/worker、`CRITICAL_STOP.json` 不存在，既有 runner 与固定解释器均存在。已启动唯一整体入口 `batch_runner.py`，controller PID 为 34496；新 stdout/stderr 分别为 `controller.resume.20261001_215904.stdout.log` 与 `controller.resume.20261001_215904.stderr.log`。未修改冻结配置、未重跑已完成内容，未并行启动其他脚本。此后由 runner 自主管理 checkpoint、resume、skip-completed、失败记录和阶段核验；Codex 不持续监控。

本文件依据本次对磁盘状态文件、完成标记、现有报告、脚本和进程的一次性交接核对编写。没有重新训练、重新计算统计量、重新执行 bootstrap/replay 或持续读取训练日志。文中的科学结果均摘自已经保存的报告。仅重新读取了三个小型元数据文件的 SHA256；不把历史完整核验冒充本次重新核验。

## 0. 当前生效的指令和停止状态

**用户已于 2026-10-01 21:59 +09:00 明确授权恢复。现有唯一 runner 已启动；禁止再手动启动第二个 controller、单独 seed、统计或汇总脚本，也不持续监控。**

- 工作区根目录：`D:\codex\insole`。
- 算法项目根目录：`D:\codex\insole\SAFE_Gait_V2_3`。以下相对路径除特别说明外均相对于这个算法项目根目录。
- 本次停止了批次 controller PID 35384；其启动时间为 2026-10-01 17:34:28，解释器路径与磁盘启动记录一致。venv launcher 为 PID 35216。
- 停止 controller 后，一次性检查仍发现四个对应的实际 Python worker 进程；按已核对的 PID、解释器路径和启动时间停止了 PID 4976、20648、30320、36016。最后一次检查该批次 controller、launcher、worker 均已不存在。没有删除模型或 checkpoint。
- **`status.json` 中的 TRAINING、`STARTED.json` 中的 RUNNING，以及四个 job 的 RUNNING，是停止前遗留状态，不表示现在仍在运行。** 保留这些原始文件，未伪造 runner 已处理的完成或失败状态。下一次获准恢复时，runner 会将遗留 RUNNING attempt 记为 INTERRUPTED，并恢复已有 checkpoint。
- Windows Job Object 的 README 描述不足以保证本次实际解释器后代立即退出；交接时已单独清理确认。后续遇到中断时应核对真实进程，不能仅相信状态文件或 launcher PID。
- 未发现批次 `CRITICAL_STOP.json`。本次中断是用户要求交接而主动停止，不是科学结果触发 early stopping，也不是已记录的 batch fatal。
- **自动化控制已处理：**历史 ID `v2-3` 的配置已不在磁盘上；当前唯一自动化配置为 `C:\Users\user\.codex\automations\automation\automation.toml`，ID `automation`，原状态为 `ACTIVE`，提示词为“继续未完成的任务。”。在 Codex 自动化接口中已将该实际自动化暂停为 `PAUSED`；未直接改写配置文件，也未改变其余配置字段。后续接手者不得恢复该自动化或据其旧 prompt 自动续跑。

## 1. 研究目标和当前阶段

研究 subject-specific residual correction 在同条件及跨条件 one-shot gait estimation 中的有效性。四个主方法固定为 calibrated population（CP）、simple bias、diagonal shrinkage（DS）、full SAFE-Gait V2.3。评估速度、步长、步时，以及收益与个体 negative transfer；不再优化 gate 或算法结构。

WearGait-PD 的固定 subject split、training seeds 7/17/27 五折比较已经完成。GAITRite 的数据接口、64-channel frontend、数据冻结和 training_seed=7 的 225 个模型及完整统计核验已经完成。当前在 GAITRite 的 training_seed=17 批次中途暂停：105/225 完成，4 个任务中断，116 个尚未开始。training_seed=27 尚未开始。**GAITRite 三-seed 最终报告尚未生成。**

历史文档中的“尚未开始外部验证”只描述该历史时间点，不能覆盖这里记录的实际进度。

## 2. 已冻结的数据、划分、算法和协议

### 数据和 exclusions

- WearGait 原始数据：`D:\codex\insole\1.WearGait-PD`。已完成三-seed结果：`reports\three_seed_frozen_folds_20260930`。WPD021 暂时排除主分析；不使用 HurriedPace_mat、Balance 或不完整压力数据。具体旧 manifest 和输入封存以该系列报告及其依赖的 fold0/fivefold 冻结目录为准。
- GAITRite 原始数据：`D:\codex\insole\insole-gaitrite-dataset`。
- GAITRite 权威冻结目录：`reports\gaitrite_final_freeze_20261001`。
- Confirmatory 数据：19 名 subject、570 个 physical passes；slow/SP 195 passes、natural/NP 190、fast/FP 185。主方向 natural→fast 和 fast→natural 各有18名 subject。其余组合保留为 secondary；不存在 fast 的 subject 不因结果而补选。
- P1、P7、P9 为 development-only，不进入正式训练、validation 或 test。`P12_SP_7` 因左右脚同步仍存在 cycle 歧义而排除，不手工选择 offset。
- GAITRite 的 SP 表示 slow，**不等同于 WearGait 的 SP 标签**。预定问题映射是 GAITRite NP/natural 对应 WearGait SP 的研究问题、FP/fast 对应 HP。
- 64 个压力通道完整保留，顺序 L0..31、R0..31；不裁为32通道，也不直接拼接两个数据库的原始通道。

### 权威冻结文件及 hash

| 文件 | 用途 |
|---|---|
| `final_external_manifest.csv` | 最终 external manifest |
| `pressure_input_index.csv` | 无 reference 值的模型输入索引 |
| `reference_index.csv` | 独立 reference 索引 |
| `subject_cv_split.json` | 全部 outer/inner subject 名单 |
| `frozen_support_query_episodes.csv` | 固定 support/query physical pass 配对 |
| `frozen_protocol.json` | targets、frontend、TCN、下游方法和评估规则 |
| `DATA_SHA256.json` / `CODE_SHA256.json` | 原始数据与代码清单 |
| `frozen_code\` | 冻结源码快照 |
| `OUTPUT_SHA256.json` / `FREEZE_COMPLETE.json` | 冻结产物清单和 seal |

本次读取确认的 SHA256：

- `subject_cv_split.json`：`20424b67c8ff6412c6918087a10ba038c362ab95636a559399b3268c9d33b031`
- `frozen_protocol.json`：`6e529622fd2b614b3e080e7575339e492d4cd3ae3a3e86db8d18e4565f84aa78`
- 批次 `BATCH_LOCK.json`：`5a4f2fd9f31ce97955bbb816337c8b389eab4da7ccd131c7e165f4e6391fad32`

历史 `FREEZE_COMPLETE.json` 记录：冻结输出清单 SHA256 为 `c906932748018506768b38f53bc4a1c7265e8fe5388569b90da0d3a25295a3b1`，bundle hashes SHA256 为 `d0297002c625246217a11c06abe6663adb28cf155bc1b876939254d3684001a4`。这些是历史核验记录，本次未重新扫描所有原始数据。

### 模型、训练与评估规则

- 固定 subject split seed=7。training seeds 17、27 只改变初始化、dropout、minibatch RNG；outer/inner 名单、heldout routing 全部不变。不要重新生成 split。
- 三个独立二条件 family：`NP_FP`（主方向来源）、`SP_NP`、`SP_FP`（slow extension）。每个 family 有5个 outer folds，每 fold 15个模型（5 inner folds ×3个既定内层 repeats），每 training seed 共225个模型。
- 内层 partition seeds `[7,17,27]` 是固定 inner repeats，不代表已运行三个 training seeds。初始化公式为 `1000 + training_seed + repeat*100 + inner_fold`；三批 init base 分别1007、1017、1027。`SEED_SEMANTICS.json` 记录此含义。
- Frontend 使用全部 released pressure clips，pool factor=4，log1p，inner-train subject-balanced channel standardization；不按 reference/sync 裁切或偏移压力序列。
- PopulationTCN：64 input channels、3 targets、hidden=32、dropout=0.1、固定100 epochs；AdamW，lr=0.001、weight_decay=0.001、batch_size=16、gradient_clip=5。target normalization 仅使用 inner-train，loss 为 subject-balanced MSE；无 validation 选 epoch 或 scientific early stopping。
- 严格 training-only subject-level cross-fitting：`residual = reference - population prediction`。三次 routed prediction 按 pass 平均。validation/test 不参与 preprocessing、target scaling、prior 或下游参数估计。
- 下游 SAFE covariance_shrinkage=0.2、min_transfer_correlation=0、damping=false；四方法逻辑保持冻结。gate 仅为旧探索性工作，不进入主方法。
- Primary targets：velocity（cm/s）、step_length（cm）、step_time（s）。reference 为每个 clip 内按时间排序的完整 steps，排除第一个可用 event；velocity=sum(length)/sum(time)，step_length/step_time 分别为均值。不使用 condition summary 替代逐pass reference。此前核对支持 released physical-pass 对齐；不声称已经验证原始 sample-clock drift 精度。
- Support 为该 subject/support condition 最小有效 numeric clip ID；query 为目标 condition 的所有其他有效 physical passes；同一 pass 不能兼作 support/query。
- 先按 subject 聚合 query error，再对 subject 等权统计。Bootstrap=2000、RNG seed=7；CI 未作多重比较校正。跨 training seeds 先平均同一 subject 的 error，再对 subject bootstrap，不能把 seed×subject 当独立人数。

## 3. 已完成 seed / fold / jobs 与失败任务

### WearGait（已完成的历史研究）

`reports\three_seed_frozen_folds_20260930\verification.json` 记录 run seeds 7/17/27、每批5 outer folds，共225个模型（seed7复用75个，新增150个），全部重放通过。不要把这套 WearGait 三-seed报告误当 GAITRite 三-seed完成。

### GAITRite

| Training seed | 完成 `COMPLETE.json` | 已完成训练的 family/folds | 已有 `EVALUATED.json` | 当前状态 |
|---|---:|---|---:|---|
| 7 | 225/225 | 三个 family 的 fold0–4，每 fold15个 | 15 | COMPLETE_VERIFIED |
| 17 | 105/225 | NP_FP fold0–4 共75个；SP_NP fold0–1 共30个 | 0 | 用户要求停止，尚未完成整批 |
| 27 | 0/225 | 无；仅已准备执行脚本和 seed semantics | 0 | 未启动 |

seed17 `job_status.json` 的最后保存计数为 COMPLETED=105、PENDING=116、RUNNING=4。没有记录 FAILED job 或 FAILED attempt。后四个现在是主动中断，不应算作技术失败或训练成功；当前剩余120个未完成任务。

四个中断任务均在 `SP_NP/fold_2`：

| Model | 恢复文件 | 最后保存时间（+09:00） |
|---|---|---|
| `r0_k0` | `training_state.pt` 存在，486869 bytes | 19:20:29 |
| `r0_k1` | `training_state.pt` 存在，486165 bytes | 19:20:30 |
| `r0_k2` | `training_state.pt` 存在，485461 bytes | 19:20:07 |
| `r0_k3` | 仅见 `provenance.json`，未见 `training_state.pt` 或完成 seal | 19:20:28（provenance） |

本次没有加载 checkpoint tensor 来推测具体 epoch。原 trainer 每25 epochs 保存 optimizer/RNG state；前三个应由原 runner 按文件中的 epoch 恢复。第四个若仍无 state，恢复时会从该未完成任务起点开始。不能把 checkpoint 的存在当成任务已经完成。

## 4. 已完成统计、bootstrap、replay/hash 和 leakage 核验

- WearGait `verification.json`：PASS；225个 checkpoint 重放，最大差异 `5.684341886080802e-14`；48个 primary summary、36个 DS paired comparison、90个 coefficient rows 已独立复核；subject cluster bootstrap 与跨seed个体风险指标通过。
- GAITRite seed7 `COMPLETED.json`：COMPLETE_VERIFIED；225个100-epoch模型，全部 checkpoint 重放，最大差异 `2.842170943040401e-14`；144个统计行、108个配对行、90个 coefficient interval 行已核验。
- seed7 training-subject β/TVR bootstrap：15个 training priors ×2000 resamples，共30000次；不使用 outer test 拟合，不重新训练 population，不产生新 gate。
- seed7已保存 train/validation/test 无 subject leakage、每 family 每名subject只进一次 test、排除名单有效、support/query physical pass不同、frozen manifest/split/code 未变化的 PASS 结果。
- seed7 output inventory SHA256（完成 seal 所记）：`6287c77f25b085b36df54224ec20a543fb00d330de2b7b414046fda6fb0d4d11`。
- 新批次 `PREFLIGHT.json`：PASS；核验1887个 seed7封存文件、2724个data files、19个code files、固定split、排除名单及1634个support/query episodes；PyTorch2.7.1+cu118、NumPy2.5.3、RTX4080环境匹配。`initial_frozen_leakage_audit.json` 也记录 PASS。
- 新批次已经完成的105个任务各有 `COMPLETE.json`；train_model 在接受 seal 前执行同任务 checkpoint reload/prediction一致性检查。**seed17整批结束后的独立 replay、scoring、统计、β/TVR bootstrap 尚未开始**，不可把 per-job seal 说成已完成整批核验。
- 本次交接没有重新运行上述验证或分析，只读取现有证据。

## 5. 当前仍运行或未完成的工作

本次恢复前，已核对的本地训练批次进程均不在运行。用于续跑的实际自动化 `automation` 仍暂停；2026-10-01 21:59:05 +09:00 已手动启动唯一 batch controller（PID 34496），见 NEXT_ACTION。

尚未完成的既定实验链为：seed17剩余120 jobs → seed17整批技术/协议核验和分析 → seed27全部225 jobs → seed27整批核验和分析 → GAITRite三-seed最终汇总。该链当前暂停，不能因旧 heartbeat 或旧 README 自动恢复。

批次目录目前没有 `COMPLETED.json`、`THREE_SEED_REPORT.md` 或最终 aggregation seal，也没有 `CRITICAL_STOP.json`。seed17、seed27尚无正式 final statistics；不得解读其部分训练结果或据此修改规则。

最终交付仍须包括三-seed MAE/gain、subject-level bootstrap95%CI、benefit/negative-transfer、β/TVR、四方法比较、WearGait与GAITRite的复现一致性判断及论文可用结论。现有 `aggregate_three.py` 会生成数值汇总和条件化判断；最终接手者仍须核查报告是否完整覆盖与 WearGait 的逐 target/direction 一致性及论文结论，不能仅凭文件存在就宣告科学解释已完成。

## 6. 重要路径和脚本

| 路径（相对于算法项目根目录） | 内容 |
|---|---|
| `reports\gaitrite_training_seeds_17_27_20261001\` | 本地低监督 controller、执行锁和本批次状态 |
| 上述目录 `batch_runner.py` | 唯一整体入口；seed17→核验→seed27→核验→三-seed汇总 |
| `batch_policy.py` / `RUNNER_POLICY.json` | 技术异常分类、重复失败策略、checkpoint seal与subject聚合 |
| `seed_task.py` | 单模型worker或训练结束后的冻结 scoring；scoring模式禁止补训缺失模型 |
| `replay_models.py` | 对所有成功模型独立重放，检查 seed/config、训练名单和预处理 |
| `prepare_batch.py` / `ADAPTER_DIFF.patch` | 已生成的执行adapter来源和差异；**恢复不需要重新生成或重封存** |
| `aggregate_three.py` | 三training-seed subject-level最终汇总；目前未执行 |
| `BATCH_LOCK.json` | 冻结输入、源码、adapter与环境锁；不得改文件后绕过锁 |
| `status.json` / `STARTED.json` | 停止前状态；当前内容可能陈旧，不能当作活进程证明 |
| `events.jsonl` / `controller.stdout.log` / `controller.stderr.log` | 批次事件和控制台日志；本次未读取训练日志正文 |
| `reports\gaitrite_replication_seed17_20261001\job_status.json` | 225任务状态和attempt记录 |
| 各seed目录 `job_logs\` | 每个attempt的 `.log` 和结果 `.json` |
| 各seed目录 `{family}\fold_{0..4}\models\r{0..2}_k{0..4}\` | `training_state.pt`、`model.pt`、`preprocessor.npz`、`provenance.json`、`predictions.csv`、`training_history.csv`、`COMPLETE.json`；仅实际生成的文件可使用 |
| `reports\gaitrite_replication_seed7_20261001\` | 已完整核验的GAITRite seed7结果和原执行源码；只读复用 |
| 上述seed7目录 `REPLICATION_REPORT.md` / `CONCLUSIONS.md` | 已完成科学结果 |
| `primary_metrics.csv` / `secondary_metrics.csv` / `DS_paired_comparisons.csv` | 四方法完整统计及配对差异 |
| `beta_TVR_by_fold.csv` / `beta_TVR_training_subject_bootstrap.csv` | 每fold系数与CI |
| `verification.json` / `checkpoint_replay_verification.csv` / `OUTPUT_SHA256.json` | 历史核验证据 |
| `reports\three_seed_frozen_folds_20260930\` | WearGait三seed结果、`CONCLUSIONS.md`、`THREE_SEED_REPORT.md` 和核验证据 |
| `docs\superpowers\plans\2026-10-01-gaitrite-low-supervision.md` | runner实现计划、既往10项测试和代码复核记录；其“运行中”启动记录是历史信息 |

`CODEX_NEXT_TASK.md` 是过时的早期计划，其中重连pipeline、额外优化或神经adapter等建议不能替代当前冻结协议与用户指令。旧 WearGait外部协议草案中的32-channel建议也不是当前GAITRite64-channel方案。

## 7. 当前确认的主要结果（仅引用已保存结果）

### WearGait 三-seed

已有报告支持 DS 在两个跨条件方向的 velocity、step_length 平均 gain 为正；合并 subject-bootstrap CI 排除0。对应四格 mean gain：SP→HP velocity 3.082 cm/s [0.244,5.910]、step_length 2.101 cm [1.151,3.111]；HP→SP velocity 3.042 cm/s [0.409,5.618]、step_length 1.869 cm [0.606,3.225]。每方向23人。

step_time 的合并 CI 跨0，HP→SP平均gain略负。DS不能被称为全面优于simple bias或full V2.3；只支持将其冻结为简单候选并保留全部比较方法。全部12个DS pair×target都有个体negative transfer。

### GAITRite seed7

两个主方向各18人。DS 相对CP：

| 方向 | Target | DS MAE | Mean gain [95%CI] | Benefit | Negative transfer |
|---|---|---:|---:|---:|---:|
| natural→fast | velocity | 16.4336 | 3.39682 [-3.26228,10.7235] | 55.6% | 38.9% |
| natural→fast | step_length | 7.24956 | 2.10218 [-1.06361,5.47792] | 72.2% | 22.2% |
| natural→fast | step_time | 0.0119462 | 0.00000337 [-0.00016865,0.00017334] | 11.1% | 11.1% |
| fast→natural | velocity | 17.0851 | -0.958429 [-7.64197,4.40221] | 55.6% | 44.4% |
| fast→natural | step_length | 7.37106 | 0.464142 [-1.73407,2.68033] | 61.1% | 38.9% |
| fast→natural | step_time | 0.0222904 | 0.00003247 [-0.00017051,0.00024920] | 11.1% | 11.1% |

单位分别为 cm/s、cm、s。六格平均 gain CI 全部跨0；四个velocity/step_length格子中3个均值同方向改善，fast→natural velocity 未同方向复现。natural→fast step_length 的 median gain 为1.449 cm [0.362,3.656]，但不能据此声称平均收益已经稳定确认。

DS相对simple bias的清晰配对优势仅见natural→fast step_time；相对full V2.3的清晰配对优势仅见fast→natural step_time，后者主要来自一个fold。其余差异CI跨0，不证明等效。完整四方法数值在 seed7 `REPLICATION_REPORT.md`，本次未重算。

Training OOF 中 velocity/step_length 的 β 在两个方向均5/5 folds为正；step_time只1/5为正，β与TVR跨fold中位数为0。正β不保证独立test收益，且β受非负方差约束，不能当作无约束的显著性检验。

**当前只能说GAITRite单seed呈部分方向一致；不能声称已经完成三seed复现，也不能把DS宣布为普遍最佳。**

## 8. NEXT_ACTION（唯一）

**已完成：用户授权下已启动唯一 batch controller（PID 34496）。不读取日志或轮询；等待 runner 自行完成完整阶段、报告 critical error 或停止后，再做一次状态核验并更新本文件。**

历史 ID `v2-3` 已不在磁盘上；实际配置的 ID 为 `automation`，其旧提示词为“继续未完成的任务。”。本次通过 Codex 自动化接口将其状态更新为 `PAUSED`，没有直接改写配置文件或更改其他字段。

后续既定研究链由已启动的 runner 按第5节顺序执行；禁止手动插队运行 seed27、aggregate、bootstrap 或第二个 controller。

## 9. 恢复命令（仅记录，未执行）

以下入口、解释器和目录已在磁盘确认存在。`batch_runner.py` 的 `main()` 明确顺序执行 seeds17、27，跳过已封存模型，恢复 checkpoint，然后运行阶段核验和最终汇总。恢复前须确认没有旧controller/worker仍活跃，且用户已明确撤销本次暂停要求。

在 PowerShell 中，未来获准恢复时可运行：

```powershell
Set-Location -LiteralPath 'D:\codex\insole\SAFE_Gait_V2_3'
Start-Process -FilePath 'D:\codex\insole\APACE_V6_RealFix_PyCharm\APACE_V6_RealFix\.venv\Scripts\python.exe' -ArgumentList @('-u', 'D:\codex\insole\SAFE_Gait_V2_3\reports\gaitrite_training_seeds_17_27_20261001\batch_runner.py') -WorkingDirectory 'D:\codex\insole\SAFE_Gait_V2_3' -WindowStyle Hidden -RedirectStandardOutput 'D:\codex\insole\SAFE_Gait_V2_3\reports\gaitrite_training_seeds_17_27_20261001\controller.resume.stdout.log' -RedirectStandardError 'D:\codex\insole\SAFE_Gait_V2_3\reports\gaitrite_training_seeds_17_27_20261001\controller.resume.stderr.log' -PassThru
```

重定向使用新的 resume 日志名，保留原启动日志；若此名称以后已存在，应换一个新名称而非覆盖旧日志。不要并行手动启动 seed27、aggregate、bootstrap 或另一个controller；不要重新跑 `prepare_batch.py --seal` 以绕过变化的hash。若将来存在 `CRITICAL_STOP.json`，原入口会拒绝启动，须按原技术故障处理流程解决，不能删除标记硬跑。

保存策略由现有 `RUNNER_POLICY.json` 确定：4 workers、每worker4 threads、单job最多2次失败尝试；OOM重试单独运行、不改batch_size；5次连续同类技术失败停止；数据/协议/泄漏/存储环境故障停止。科学MAE/gain/CI/β/TVR不参与停止判定。缺失job保留失败记录，不得缩小15模型ensemble后冒充完整fold。

恢复命令已于 2026-10-01 21:59 +09:00 执行一次；GAITRite三-seed最终结论仍待完整实验和核验完成后形成。
