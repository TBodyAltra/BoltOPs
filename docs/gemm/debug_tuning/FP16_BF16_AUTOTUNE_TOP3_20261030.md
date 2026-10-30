# GEMM autotune 机台信息与逐 MNK Top3 配置

整理日期：2026-10-30。本文包含两份归档的全部 160 条需求、153 个唯一 MNK、960 条 Top3 记录。按唯一 MNK 分组，同一 MNK 的重复需求保留 requirement ID；两轮及不同输出类型分别列出，互不合并排名。

**第一轮是 BF16 输入，第二轮是 FP16 输入；均使用零输入做性能测试，未做数值校验。** Top3 按原始有效计时的最小 ms 排序，保留异常高吞吐和计时波动记录。配置只能作为复测候选，不能视为已验证的数值正确性或峰值性能。

本文包含完整配置字典、有效编译参数、HSACO 哈希、功能激活证据和原始远端日志路径，可单独复制保存。

## 机台和软件环境

| 项目 | 归档确认的信息 |
|---|---|
| 测试机 | node5；SSH 目标 `root@10.60.50.235`，经 `xukang@bedr01.hygon.cn` 跳转 |
| 硬件目标 | `gfx946`；运行日志报告 CU/XCD=48、XCD 数=2、multiProcessorCount=96、LDS/block=131072 B |
| 实际执行 die | die 5、6；`UNITTEST_DIE_EXEC_MASK=3` / `die_exec_mask=0x3`；multi-die |
| 时钟 | 所有 960 条 Top3 记录均为 die5/6 GFX=500 MHz、UCLK=1400 MHz；读取范围为 `before pack` |
| 静态设备属性 | 日志设备属性另报 `sclk=1500MHz`；这是设备属性报告值，不替代上述 GC 500 MHz 实测频率 |
| SMBus | 正式 pack 使用选定的 400 kHz；第一轮早期 setup.log 的 200 kHz 和 600/1000 kHz 探针不能混作最终口径 |
| 测试环境根目录 | `/root/jingyang/opt/unittest_shaobo` |
| 环境加载 | 在 `HIPTest` 下 source `setup_hipmock_env.sh`；`HIP_ENV_NO_SELFHEAL=1`、`HIP_ENV_NO_BUILD=1` |
| 编译机 | SSH 别名 `node60`；归档没有确认其真实 IP |
| 编译容器 | `kxu_pmd_8569_fp16_map_20261020` |
| Triton | 有 compiler_provenance 的编译产物均记为 `3.6.0`；第一轮部分早期产物未附此字段 |
| HCU backend 路径 | `/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py` |
| HCU backend SHA256 | `d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39`（归档中带 provenance 的产物一致） |
| 编译环境脚本 | `/xukang/code/pmd-triton-sessions/gemm-optimization-20261020/env.sh` |
| runner-pack SHA256 | `2ef9ca1bb934f70b5186a810d71dd4b44a0bf5e47ea9184533de5d10526b7c0e`，第二轮 CAMPAIGN.md 记为沿用第一轮版本 |
| OS、驱动、hipmock、VBIOS 精确版本 | 这些归档没有提供足以确认的完整版本清单，本文不填推测值 |

环境信息依据：两轮 `service.py`、`start-pack.sh`、首批 `console.log`，第一轮 `EXPERIMENT.md`、`throughput/RING-SPEED-REPORT.md`，第二轮 `CAMPAIGN.md`，以及逐条 clocks 和编译产物 compiler_provenance。

## 两轮范围和最终归档状态

| 项目 | 第一轮 BF16 | 第二轮 FP16 |
|---|---|---|
| 输入 / 累加 | BF16 / FP32 | FP16 / FP32 |
| 输出 | 133 条 BF16、27 条 FP32 | 160 条 FP16 |
| 需求数 / 唯一 MNK | 160 / 153 | 160 / 153 |
| 粗调 | 1440/1440，9 配置/需求 | 3840/3840，24 配置/需求 |
| 精调 | 1280/1280 | 393/1280 |
| 已记录 / 计划 | 2720/2720 | 4233/5120（82.68%） |
| BENCH_OK | 2622 | 4164 |
| 其他记录 | DEVICE_ERROR=1；RESOURCE_REJECTED=97 | DEVICE_ERROR=3；COMPILE_FAILED=5；RESOURCE_REJECTED=58；REJECTED_SPILL=3 |
| 精调覆盖 | 160 条完成 8 项尝试 | 49 条完成 8 项、1 条完成 1 项、110 条未开始 |
| 完成 / 停止时间 | 2026-10-29 09:07:18 COMPLETE | 2026-10-30 09:46:59 用户停止；最新结果进度时间为 09:01:57 |
| 剩余 | 0 | 887 项，均为精调 |
| 最优模式分布 | nowasp=49；wasp44=27；wdra48=43；wdra88=41 | nowasp=88；wasp44=18；wdra48=40；wdra88=14 |

第一轮归档记载生产 worktree HEAD 为 `42f28687eaef2f35079c3878ca2fa1aa1bbb8439`，使用 dtype-only 实验改动，未改 FP32 累加和流水结构。第二轮走未修改的 `production_fp16.py` 快照 FP16 导出路径；不能仅凭沿用脚本推定第二轮仓库 HEAD 与第一轮相同。

“已记录”包括编译/资源拒绝与设备错误，不等于测量成功。第二轮 `FINAL24.md` 和 `final24-summary.json` 是完成 441 项时的旧快照；这里使用 records.jsonl、最新 progress.json 和 USER-STOP.json。Top3 的 fine_state=planned 是生成器标签，以下精调覆盖直接按记录 phase 计算。

## 计时与配置解读

- MNK 和 leading strides 为运行期参数；row-row、unit inner strides，公共 div16 元数据；配置跨 MNK 复用。
- ms 是最终 GEMM completion-envelope GPU 时间，TFLOPS=`2*M*N*K/(ms*1e9)`；双 die 不额外乘 2。
- 早期 warmup 1 + 计时 1；后期 warmup 0 + 计时 2，差异超过 2% 补第 3 次，取最小值。下表逐项保留实际样本数。
- 加速比基线是同轮同需求“最优粗调 nowasp”，两轮 nowasp 搜索空间不同，不能跨轮解释为优化收益。
- 波动=`(max(samples)-min(samples))/min(samples)`。没有 samples_ms 的历史单次记录显示“未记录”，不视为零波动。
- 第一轮 43 条、第二轮 47 条赢家的样本跨度超过 10%；高 TFLOPS 与极端加速比须复核计时与完整执行。
- tile 表示 BM/BN/BK。配置链接进入文末完整字典：包含 GROUP_SIZE_M、num_stages、num_warps、layout、partition、MLS、BPP、async 等原始有效选项。
- 每个表中状态均为 BENCH_OK；pending 是该需求剩余计划数。按原始配置 ID 保留 Top3，不因 mode/tile 相同而去重。
- 所有当前赢家 spill=0；BPP/async 均没有 verified 激活证据。是否启用功能以配置附录的 IR/ISA evidence 为准。

## MNK 索引

- [1 × 154880 × 6144](#mnk-1-154880-6144)：G237
- [2 × 64640 × 7168](#mnk-2-64640-7168)：G238
- [2 × 129280 × 7168](#mnk-2-129280-7168)：G239
- [3 × 120832 × 4096](#mnk-3-120832-4096)：G240
- [4 × 192 × 4096](#mnk-4-192-4096)：G241
- [4 × 4096 × 8192](#mnk-4-4096-8192)：G242
- [4 × 7168 × 21504](#mnk-4-7168-21504)：G243
- [4 × 120832 × 4096](#mnk-4-120832-4096)：G244
- [4 × 154880 × 6144](#mnk-4-154880-6144)：G245
- [4 × 163840 × 7168](#mnk-4-163840-7168)：G246
- [6 × 32 × 6144](#mnk-6-32-6144)：G247
- [6 × 256 × 6144](#mnk-6-256-6144)：G248
- [6 × 6144 × 12288](#mnk-6-6144-12288)：G249
- [6 × 154880 × 6144](#mnk-6-154880-6144)：G250
- [8 × 16160 × 7168](#mnk-8-16160-7168)：G251
- [8 × 19360 × 6144](#mnk-8-19360-6144)：G252
- [8 × 129280 × 4096](#mnk-8-129280-4096)：G253
- [8 × 248320 × 4096](#mnk-8-248320-4096)：G254
- [12 × 32 × 6144](#mnk-12-32-6144)：G255
- [12 × 64 × 2048](#mnk-12-64-2048)：G256
- [12 × 256 × 2048](#mnk-12-256-2048)：G257
- [12 × 256 × 6144](#mnk-12-256-6144)：G258
- [12 × 1024 × 2048](#mnk-12-1024-2048)：G259
- [12 × 2048 × 512](#mnk-12-2048-512)：G260
- [12 × 2048 × 4096](#mnk-12-2048-4096)：G261
- [12 × 6144 × 12288](#mnk-12-6144-12288)：G262
- [12 × 9216 × 2048](#mnk-12-9216-2048)：G263
- [12 × 12288 × 2048](#mnk-12-12288-2048)：G264
- [12 × 154880 × 6144](#mnk-12-154880-6144)：G265
- [12 × 248320 × 2048](#mnk-12-248320-2048)：G266
- [16 × 192 × 4096](#mnk-16-192-4096)：G267
- [16 × 384 × 7168](#mnk-16-384-7168)：G268
- [16 × 2112 × 7168](#mnk-16-2112-7168)：G269
- [16 × 4096 × 7168](#mnk-16-4096-7168)：G270
- [16 × 7168 × 2048](#mnk-16-7168-2048)：G271
- [16 × 7168 × 8192](#mnk-16-7168-8192)：G272
- [16 × 7168 × 18432](#mnk-16-7168-18432)：G273
- [16 × 12288 × 1536](#mnk-16-12288-1536)：G274
- [16 × 16160 × 7168](#mnk-16-16160-7168)：G275
- [16 × 36864 × 7168](#mnk-16-36864-7168)：G276
- [16 × 120832 × 4096](#mnk-16-120832-4096)：G277
- [16 × 163840 × 7168](#mnk-16-163840-7168)：G278
- [24 × 32 × 6144](#mnk-24-32-6144)：G279
- [24 × 256 × 6144](#mnk-24-256-6144)：G280
- [24 × 15104 × 4096](#mnk-24-15104-4096)：G281
- [24 × 154880 × 6144](#mnk-24-154880-6144)：G282
- [32 × 19360 × 6144](#mnk-32-19360-6144)：G283
- [32 × 20480 × 7168](#mnk-32-20480-7168)：G284
- [48 × 32 × 6144](#mnk-48-32-6144)：G285
- [48 × 256 × 6144](#mnk-48-256-6144)：G286
- [48 × 512 × 4096](#mnk-48-512-4096)：G287
- [48 × 4096 × 8192](#mnk-48-4096-8192)：G288
- [48 × 124160 × 4096](#mnk-48-124160-4096)：G289
- [48 × 154880 × 6144](#mnk-48-154880-6144)：G290
- [64 × 64 × 2048](#mnk-64-64-2048)：G291
- [64 × 256 × 2048](#mnk-64-256-2048)：G292
- [64 × 16160 × 4096](#mnk-64-16160-4096)：G293
- [64 × 31040 × 4096](#mnk-64-31040-4096)：G294
- [64 × 248320 × 2048](#mnk-64-248320-2048)：G295
- [80 × 129280 × 7168](#mnk-80-129280-7168)：G296
- [96 × 64 × 7168](#mnk-96-64-7168)：G297
- [96 × 512 × 7168](#mnk-96-512-7168)：G298, G405
- [96 × 1024 × 7168](#mnk-96-1024-7168)：G406
- [96 × 2048 × 7168](#mnk-96-2048-7168)：G407
- [96 × 7168 × 21504](#mnk-96-7168-21504)：G299
- [96 × 129280 × 7168](#mnk-96-129280-7168)：G300
- [160 × 129280 × 4096](#mnk-160-129280-4096)：G301
- [192 × 64 × 4096](#mnk-192-64-4096)：G302
- [192 × 512 × 4096](#mnk-192-512-4096)：G303, G408
- [192 × 1024 × 4096](#mnk-192-1024-4096)：G409
- [192 × 2048 × 4096](#mnk-192-2048-4096)：G410
- [192 × 4096 × 12288](#mnk-192-4096-12288)：G304
- [192 × 124160 × 4096](#mnk-192-124160-4096)：G305
- [192 × 129280 × 4096](#mnk-192-129280-4096)：G306
- [240 × 129280 × 4096](#mnk-240-129280-4096)：G307
- [288 × 64 × 4096](#mnk-288-64-4096)：G308
- [288 × 512 × 4096](#mnk-288-512-4096)：G309, G411
- [288 × 1024 × 4096](#mnk-288-1024-4096)：G412
- [288 × 2048 × 4096](#mnk-288-2048-4096)：G413
- [288 × 4096 × 12288](#mnk-288-4096-12288)：G310
- [288 × 129280 × 4096](#mnk-288-129280-4096)：G311
- [512 × 192 × 4096](#mnk-512-192-4096)：G312
- [2048 × 64 × 2048](#mnk-2048-64-2048)：G313
- [2048 × 256 × 2048](#mnk-2048-256-2048)：G314
- [2048 × 1024 × 2048](#mnk-2048-1024-2048)：G315
- [2048 × 2048 × 512](#mnk-2048-2048-512)：G316
- [2048 × 2048 × 4096](#mnk-2048-2048-4096)：G317
- [2048 × 9216 × 2048](#mnk-2048-9216-2048)：G318
- [2048 × 12288 × 2048](#mnk-2048-12288-2048)：G319
- [2560 × 32 × 6144](#mnk-2560-32-6144)：G320
- [2560 × 64 × 2048](#mnk-2560-64-2048)：G321
- [2560 × 64 × 7168](#mnk-2560-64-7168)：G322
- [2560 × 256 × 2048](#mnk-2560-256-2048)：G323
- [2560 × 256 × 6144](#mnk-2560-256-6144)：G324
- [2560 × 512 × 7168](#mnk-2560-512-7168)：G414
- [2560 × 1024 × 7168](#mnk-2560-1024-7168)：G415
- [2560 × 2048 × 7168](#mnk-2560-2048-7168)：G416
- [3584 × 64 × 4096](#mnk-3584-64-4096)：G325
- [3584 × 128 × 4096](#mnk-3584-128-4096)：G326
- [3584 × 512 × 4096](#mnk-3584-512-4096)：G327, G417
- [3584 × 1024 × 4096](#mnk-3584-1024-4096)：G418
- [3584 × 2048 × 4096](#mnk-3584-2048-4096)：G419
- [4608 × 32 × 6144](#mnk-4608-32-6144)：G328
- [4608 × 64 × 7168](#mnk-4608-64-7168)：G329
- [4608 × 256 × 6144](#mnk-4608-256-6144)：G330
- [4608 × 512 × 7168](#mnk-4608-512-7168)：G420
- [4608 × 1024 × 7168](#mnk-4608-1024-7168)：G421
- [4608 × 2048 × 7168](#mnk-4608-2048-7168)：G422
- [5120 × 64 × 4096](#mnk-5120-64-4096)：G331
- [5120 × 384 × 7168](#mnk-5120-384-7168)：G332
- [5120 × 512 × 4096](#mnk-5120-512-4096)：G423
- [5120 × 512 × 7168](#mnk-5120-512-7168)：G333
- [5120 × 1024 × 4096](#mnk-5120-1024-4096)：G424
- [5120 × 2048 × 4096](#mnk-5120-2048-4096)：G425
- [5120 × 2112 × 7168](#mnk-5120-2112-7168)：G334
- [5120 × 4608 × 7168](#mnk-5120-4608-7168)：G335
- [5120 × 7168 × 256](#mnk-5120-7168-256)：G336
- [5120 × 7168 × 2304](#mnk-5120-7168-2304)：G337
- [5120 × 7168 × 8192](#mnk-5120-7168-8192)：G338
- [5120 × 12288 × 1536](#mnk-5120-12288-1536)：G339
- [5120 × 16384 × 512](#mnk-5120-16384-512)：G340
- [6656 × 32 × 6144](#mnk-6656-32-6144)：G341
- [6656 × 64 × 2048](#mnk-6656-64-2048)：G342
- [6656 × 64 × 4096](#mnk-6656-64-4096)：G343
- [6656 × 64 × 7168](#mnk-6656-64-7168)：G344
- [6656 × 128 × 4096](#mnk-6656-128-4096)：G345
- [6656 × 192 × 4096](#mnk-6656-192-4096)：G346
- [6656 × 256 × 2048](#mnk-6656-256-2048)：G347
- [6656 × 256 × 6144](#mnk-6656-256-6144)：G348
- [6656 × 384 × 7168](#mnk-6656-384-7168)：G349
- [6656 × 512 × 4096](#mnk-6656-512-4096)：G350, G426
- [6656 × 512 × 7168](#mnk-6656-512-7168)：G351, G427
- [6656 × 1024 × 2048](#mnk-6656-1024-2048)：G352
- [6656 × 1024 × 4096](#mnk-6656-1024-4096)：G428
- [6656 × 1024 × 7168](#mnk-6656-1024-7168)：G429
- [6656 × 2048 × 512](#mnk-6656-2048-512)：G353
- [6656 × 2048 × 4096](#mnk-6656-2048-4096)：G354, G430
- [6656 × 2048 × 7168](#mnk-6656-2048-7168)：G431
- [6656 × 2112 × 7168](#mnk-6656-2112-7168)：G355
- [6656 × 4608 × 7168](#mnk-6656-4608-7168)：G356
- [6656 × 7168 × 256](#mnk-6656-7168-256)：G357
- [6656 × 7168 × 2304](#mnk-6656-7168-2304)：G358
- [6656 × 7168 × 8192](#mnk-6656-7168-8192)：G359
- [6656 × 9216 × 2048](#mnk-6656-9216-2048)：G360
- [6656 × 12288 × 1536](#mnk-6656-12288-1536)：G361
- [6656 × 12288 × 2048](#mnk-6656-12288-2048)：G362
- [6656 × 16384 × 512](#mnk-6656-16384-512)：G363
- [28672 × 16 × 4096](#mnk-28672-16-4096)：G364
- [40960 × 4608 × 7168](#mnk-40960-4608-7168)：G365
- [40960 × 7168 × 2304](#mnk-40960-7168-2304)：G366
- [53248 × 16 × 4096](#mnk-53248-16-4096)：G367
- [53248 × 4608 × 7168](#mnk-53248-4608-7168)：G368
- [53248 × 7168 × 2304](#mnk-53248-7168-2304)：G369

<a id="mnk-1-154880-6144"></a>
## MNK 1 × 154880 × 6144

### 第一轮 BF16 · G237 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.959359884 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.126959994 | 14.9903 | 31.1859× | 0.129999995, 0.126959994, 0.127120003 | 2.39% |
| 2 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 3.866400003 | 0.4922 | 1.0240× | 3.866400003, 3.868400097 | 0.05% |
| 3 | [a8db5435a105c79a8cd1](#r1-a8db5435a105c79a8cd1) | wasp44 · 64/64/64 | fine | 3.868400097 | 0.4920 | 1.0235× | 3.868400097, 3.873440027 | 0.13% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `1x154880x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=2；原始结果 ID `1x154880x6144-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0096`。
- Rank 3：warmup=0，timed=2；原始结果 ID `1x154880x6144-bf16-a8db5435a105c79a8cd1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0096`。
### 第二轮 FP16 · G237 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 3.586479902 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [768b997665c94dc584f7](#r2-768b997665c94dc584f7) | nowasp · 16/128/64 | fine | 3.326319933 | 0.5722 | 1.0782× | 3.349040031, 3.326319933 | 0.68% |
| 2 | [484d307f1978ef3f454e](#r2-484d307f1978ef3f454e) | nowasp · 128/128/64 | fine | 3.542799950 | 0.5372 | 1.0123× | 3.542799950, 3.547600031 | 0.14% |
| 3 | [f89f00abebc2a438c6a0](#r2-f89f00abebc2a438c6a0) | nowasp · 128/128/64 | fine | 3.585599899 | 0.5308 | 1.0002× | 3.585599899, 3.593039989 | 0.21% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `1x154880x6144-fp16-G237-768b997665c94dc584f7`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0194`。
- Rank 2：warmup=0，timed=2；原始结果 ID `1x154880x6144-fp16-G237-484d307f1978ef3f454e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0194`。
- Rank 3：warmup=0，timed=2；原始结果 ID `1x154880x6144-fp16-G237-f89f00abebc2a438c6a0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0194`。

<a id="mnk-2-64640-7168"></a>
## MNK 2 × 64640 × 7168

### 第一轮 BF16 · G238 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.912639976 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.059760001 | 31.0134 | 32.0054× | 0.063280001, 0.059760001, 0.059760001 | 5.89% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.060720000 | 30.5230 | 31.4993× | 0.063919999, 0.060880002, 0.060720000 | 5.27% |
| 3 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 1.912639976 | 0.9690 | 1.0000× | 1.912639976, 1.922000051 | 0.49% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `2x64640x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2x64640x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2x64640x7168-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0008`。
### 第二轮 FP16 · G238 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 1.890720010 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 1.890720010 | 0.9802 | 1.0000× | 1.890720010, 1.891759992 | 0.06% |
| 2 | [9c257c155867db0a649b](#r2-9c257c155867db0a649b) | nowasp · 64/64/128 | fine | 1.892879963 | 0.9791 | 0.9989× | 1.896000028, 1.892879963 | 0.16% |
| 3 | [bdcb5554a17204c8e26a](#r2-bdcb5554a17204c8e26a) | nowasp · 16/64/64 | fine | 1.897359967 | 0.9768 | 0.9965× | 1.897359967, 1.914639950 | 0.91% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2x64640x7168-fp16-G238-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0129`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2x64640x7168-fp16-G238-9c257c155867db0a649b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0194`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2x64640x7168-fp16-G238-bdcb5554a17204c8e26a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0194`。

<a id="mnk-2-129280-7168"></a>
## MNK 2 × 129280 × 7168

### 第一轮 BF16 · G239 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.058240000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.058240000 | 63.6455 | 1.0000× | 未记录 | — |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.108080000 | 34.2960 | 0.5389× | 0.110799998, 0.108240001, 0.108080000 | 2.52% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.118160002 | 31.3703 | 0.4929× | 0.119920000, 0.118160002 | 1.49% |

逐项采样和追溯：

- Rank 1：warmup=1，timed=1；原始结果 ID `2x129280x7168-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0000`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2x129280x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2x129280x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
### 第二轮 FP16 · G239 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 3.554320097 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [768b997665c94dc584f7](#r2-768b997665c94dc584f7) | nowasp · 16/128/64 | fine | 3.322959900 | 1.1155 | 1.0696× | 3.346960068, 3.322959900 | 0.72% |
| 2 | [484d307f1978ef3f454e](#r2-484d307f1978ef3f454e) | nowasp · 128/128/64 | fine | 3.506720066 | 1.0570 | 1.0136× | 3.507440090, 3.506720066 | 0.02% |
| 3 | [f89f00abebc2a438c6a0](#r2-f89f00abebc2a438c6a0) | nowasp · 128/128/64 | fine | 3.549439907 | 1.0443 | 1.0014× | 3.549439907, 3.553359985 | 0.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2x129280x7168-fp16-G239-768b997665c94dc584f7`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0194`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2x129280x7168-fp16-G239-484d307f1978ef3f454e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0195`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2x129280x7168-fp16-G239-f89f00abebc2a438c6a0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0195`。

<a id="mnk-3-120832-4096"></a>
## MNK 3 × 120832 × 4096

### 第一轮 BF16 · G240 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.108880043 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.098959997 | 30.0078 | 21.3104× | 0.101839997, 0.099600002, 0.098959997 | 2.91% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.108719997 | 27.3139 | 19.3974× | 0.111120000, 0.108719997, 0.109040000 | 2.21% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 2.012880087 | 1.4753 | 1.0477× | 2.013679981, 2.012880087 | 0.04% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `3x120832x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=3；原始结果 ID `3x120832x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3x120832x4096-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0025`。
### 第二轮 FP16 · G240 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 1.865520000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f8dc5db3e2c95b32a898](#r2-f8dc5db3e2c95b32a898) | nowasp · 128/128/64 | fine | 1.660640001 | 1.7882 | 1.1234× | 1.660640001, 1.668400049 | 0.47% |
| 2 | [768b997665c94dc584f7](#r2-768b997665c94dc584f7) | nowasp · 16/128/64 | fine | 1.692880034 | 1.7542 | 1.1020× | 1.706560016, 1.692880034 | 0.81% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.865520000 | 1.5918 | 1.0000× | 1.865520000, 1.874160051 | 0.46% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3x120832x4096-fp16-G240-f8dc5db3e2c95b32a898`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0195`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3x120832x4096-fp16-G240-768b997665c94dc584f7`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0195`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3x120832x4096-fp16-G240-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0169`。

<a id="mnk-4-192-4096"></a>
## MNK 4 × 192 × 4096

### 第一轮 BF16 · G241 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.012320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.007920000 | 0.7944 | 1.5556× | 0.010320000, 0.008720000, 0.007920000 | 30.30% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.012320000 | 0.5107 | 1.0000× | 未记录 | — |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013040000 | 0.4825 | 0.9448× | 0.016240001, 0.013040000, 0.013520000 | 24.54% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x192x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=1，timed=1；原始结果 ID `4x192x4096-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0000`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x192x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
### 第二轮 FP16 · G241 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.134480000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.012560000 | 0.5009 | 10.7070× | 0.013920000, 0.012560000, 0.014640000 | 16.56% |
| 2 | [8196eabaa94d5c892734](#r2-8196eabaa94d5c892734) | wdra48 · 64/64/128 | fine | 0.092560001 | 0.0680 | 1.4529× | 0.096079998, 0.092560001, 0.093199998 | 3.80% |
| 3 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.101520002 | 0.0620 | 1.3247× | 0.118720002, 0.105039999, 0.101520002 | 16.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x192x4096-fp16-G241-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0186`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x192x4096-fp16-G241-8196eabaa94d5c892734`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0195`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x192x4096-fp16-G241-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0049`。

<a id="mnk-4-4096-8192"></a>
## MNK 4 × 4096 × 8192

### 第一轮 BF16 · G242 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.436639994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008240000 | 32.5771 | 52.9903× | 0.009840000, 0.008240000, 0.008720000 | 19.42% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013680000 | 19.6225 | 31.9181× | 0.015280000, 0.013680000, 0.013840000 | 11.70% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.132880002 | 2.0201 | 3.2860× | 0.164719999, 0.133520007, 0.132880002 | 23.96% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x4096x8192-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x4096x8192-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x4096x8192-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G242 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.011120000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 28.1970 | 1.1681× | 0.010800000, 0.009680000, 0.009520000 | 13.45% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 27.2800 | 1.1301× | 0.011920000, 0.009840000, 0.010000000 | 21.14% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.011120000 | 24.1399 | 1.0000× | 0.013840000, 0.011120000, 0.012880000 | 24.46% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x4096x8192-fp16-G242-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x4096x8192-fp16-G242-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x4096x8192-fp16-G242-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-4-7168-21504"></a>
## MNK 4 × 7168 × 21504

### 第一轮 BF16 · G243 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.295040011 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.013040000 | 94.5648 | 99.3129× | 0.015120000, 0.013520000, 0.013040000 | 15.95% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013520000 | 91.2075 | 95.7870× | 0.015120000, 0.013680000, 0.013520000 | 11.83% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.258800000 | 4.7648 | 5.0040× | 0.260720015, 0.258800000 | 0.74% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x7168x21504-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x7168x21504-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4x7168x21504-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G243 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010800000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009680000 | 127.3890 | 1.1157× | 0.013680000, 0.009680000, 0.011920000 | 41.32% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010800000 | 114.1783 | 1.0000× | 0.011920000, 0.011120000, 0.010800000 | 10.37% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.011760000 | 104.8576 | 0.9184× | 0.011760000, 0.012720000, 0.012240000 | 8.16% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x7168x21504-fp16-G243-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x7168x21504-fp16-G243-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x7168x21504-fp16-G243-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。

<a id="mnk-4-120832-4096"></a>
## MNK 4 × 120832 × 4096

### 第一轮 BF16 · G244 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.115200043 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.098959997 | 40.0103 | 21.3743× | 0.102320001, 0.099119999, 0.098959997 | 3.40% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.109040000 | 36.3117 | 19.3984× | 0.110639997, 0.109040000 | 1.47% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 2.015759945 | 1.9642 | 1.0493× | 2.015759945, 2.017040014 | 0.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x120832x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4x120832x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4x120832x4096-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0025`。
### 第二轮 FP16 · G244 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.053199999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.053199999 | 74.4252 | 1.0000× | 0.054320000, 0.053199999, 0.055280000 | 3.91% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.058800001 | 67.3371 | 0.9048× | 0.060400002, 0.058800001, 0.058959998 | 2.72% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.069360003 | 57.0851 | 0.7670× | 0.071999997, 0.069360003, 0.069360003 | 3.81% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x120832x4096-fp16-G244-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x120832x4096-fp16-G244-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x120832x4096-fp16-G244-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0074`。

<a id="mnk-4-154880-6144"></a>
## MNK 4 × 154880 × 6144

### 第一轮 BF16 · G245 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.965919971 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.127759993 | 59.5856 | 31.0420× | 0.130160004, 0.127759993 | 1.88% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.139919996 | 54.4072 | 28.3442× | 0.142000005, 0.139919996 | 1.49% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 3.307600021 | 2.3016 | 1.1990× | 3.307600021, 3.308559895 | 0.03% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4x154880x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4x154880x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4x154880x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G245 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.067599997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.067599997 | 112.6133 | 1.0000× | 0.070160002, 0.067599997, 0.067759998 | 3.79% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.071280003 | 106.7994 | 0.9484× | 0.075599998, 0.071599998, 0.071280003 | 6.06% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.090000004 | 84.5851 | 0.7511× | 0.093359999, 0.090319999, 0.090000004 | 3.73% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4x154880x6144-fp16-G245-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x154880x6144-fp16-G245-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x154880x6144-fp16-G245-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0074`。

<a id="mnk-4-163840-7168"></a>
## MNK 4 × 163840 × 7168

### 第一轮 BF16 · G246 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.821199894 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.135279998 | 69.4503 | 35.6387× | 0.137679994, 0.135279998 | 1.77% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.135759994 | 69.2048 | 35.5127× | 0.139280006, 0.136240005, 0.135759994 | 2.59% |
| 3 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 0.136240005 | 68.9610 | 35.3875× | 0.139119998, 0.136240005, 0.136559993 | 2.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4x163840x7168-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0093`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x163840x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x163840x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0093`。
### 第二轮 FP16 · G246 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.072080001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.072080001 | 130.3446 | 1.0000× | 0.073200002, 0.072080001 | 1.55% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.076240003 | 123.2324 | 0.9454× | 0.082319997, 0.076240003, 0.079120003 | 7.97% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.094640002 | 99.2735 | 0.7616× | 0.097680002, 0.094640002, 0.096400000 | 3.21% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4x163840x7168-fp16-G246-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4x163840x7168-fp16-G246-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4x163840x7168-fp16-G246-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0074`。

<a id="mnk-6-32-6144"></a>
## MNK 6 × 32 × 6144

### 第一轮 BF16 · G247 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.236159995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.006240000 | 0.3781 | 37.8462× | 0.008480000, 0.006240000, 0.006720000 | 35.90% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.011680000 | 0.2020 | 20.2192× | 0.013440000, 0.011680000, 0.011680000 | 15.07% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.133440003 | 0.0177 | 1.7698× | 0.138319999, 0.134079993, 0.133440003 | 3.66% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x32x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x32x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6x32x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G247 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.009920000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.008800000 | 0.2681 | 1.1273× | 0.010960000, 0.008800000, 0.009520000 | 24.55% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 0.2478 | 1.0420× | 0.010960000, 0.009840000, 0.009520000 | 15.13% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.009920000 | 0.2378 | 1.0000× | 0.010560000, 0.011440000, 0.009920000 | 15.32% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x32x6144-fp16-G247-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x32x6144-fp16-G247-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6x32x6144-fp16-G247-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-6-256-6144"></a>
## MNK 6 × 256 × 6144

### 第一轮 BF16 · G248 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.242960006 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008080000 | 2.3359 | 30.0693× | 0.010320000, 0.008080000, 0.008720000 | 27.72% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013200000 | 1.4299 | 18.4061× | 0.015600000, 0.013200000, 0.013520000 | 18.18% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.134959996 | 0.1399 | 1.8002× | 0.136879995, 0.134959996 | 1.42% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x256x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x256x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6x256x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G248 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010800000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 1.9826 | 1.1345× | 0.011600000, 0.009520000, 0.009520000 | 21.85% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 1.9181 | 1.0976× | 0.014000000, 0.012240000, 0.009840000 | 42.28% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010800000 | 1.7476 | 1.0000× | 0.011440000, 0.012880000, 0.010800000 | 19.26% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x256x6144-fp16-G248-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x256x6144-fp16-G248-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6x256x6144-fp16-G248-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-6-6144-12288"></a>
## MNK 6 × 6144 × 12288

### 第一轮 BF16 · G249 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.700959980 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.009040000 | 100.2179 | 77.5398× | 0.010800000, 0.009360000, 0.009040000 | 19.47% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.012720000 | 71.2240 | 55.1069× | 0.015600000, 0.014320000, 0.012720000 | 22.64% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.132080004 | 6.8592 | 5.3071× | 0.134639993, 0.132080004 | 1.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x6144x12288-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x6144x12288-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6x6144x12288-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G249 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010800000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009360000 | 96.7916 | 1.1538× | 0.010800000, 0.012080000, 0.009360000 | 29.06% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.010320000 | 87.7878 | 1.0465× | 0.011600000, 0.010320000, 0.012240000 | 18.60% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.010800000 | 83.8861 | 1.0000× | 0.014960000, 0.012240000, 0.010800000 | 38.52% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x6144x12288-fp16-G249-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x6144x12288-fp16-G249-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6x6144x12288-fp16-G249-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。

<a id="mnk-6-154880-6144"></a>
## MNK 6 × 154880 × 6144

### 第一轮 BF16 · G250 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.932240009 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.126959994 | 89.9417 | 30.9723× | 0.129999995, 0.126959994, 0.127759993 | 2.39% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.139760002 | 81.7043 | 28.1357× | 0.141360000, 0.139760002 | 1.14% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 3.308239937 | 3.4517 | 1.1886× | 3.308239937, 3.310080051 | 0.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6x154880x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6x154880x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6x154880x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G250 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.068719998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.068719998 | 166.1670 | 1.0000× | 0.068719998, 0.069839999 | 1.63% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.071280003 | 160.1991 | 0.9641× | 0.077359997, 0.074479997, 0.071280003 | 8.53% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.090000004 | 126.8777 | 0.7636× | 0.091040000, 0.090000004 | 1.16% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6x154880x6144-fp16-G250-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6x154880x6144-fp16-G250-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6x154880x6144-fp16-G250-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0074`。

<a id="mnk-8-16160-7168"></a>
## MNK 8 × 16160 × 7168

### 第一轮 BF16 · G251 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.518480003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.018800000 | 98.5829 | 27.5787× | 0.020880001, 0.018800000, 0.019280000 | 11.06% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.021679999 | 85.4870 | 23.9151× | 0.024720000, 0.021840001, 0.021679999 | 14.02% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.387919992 | 4.7777 | 1.3366× | 0.389039993, 0.387919992 | 0.29% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `8x16160x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `8x16160x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=2；原始结果 ID `8x16160x7168-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G251 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.014320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.014320000 | 129.4244 | 1.0000× | 0.015440000, 0.016559999, 0.014320000 | 15.64% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.016080000 | 115.2586 | 0.8905× | 0.016080000, 0.017680001, 0.017680001 | 9.95% |
| 3 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.016080000 | 115.2586 | 0.8905× | 0.020880001, 0.018959999, 0.016080000 | 29.85% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `8x16160x7168-fp16-G251-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `8x16160x7168-fp16-G251-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=3；原始结果 ID `8x16160x7168-fp16-G251-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。

<a id="mnk-8-19360-6144"></a>
## MNK 8 × 19360 × 6144

### 第一轮 BF16 · G252 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.520079970 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.022480000 | 84.6604 | 23.1352× | 0.025520001, 0.022639999, 0.022480000 | 13.52% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.023440000 | 81.1931 | 22.1877× | 0.025680000, 0.023920000, 0.023440000 | 9.56% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.513119996 | 3.7090 | 1.0136× | 0.514800012, 0.513119996 | 0.33% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `8x19360x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=3；原始结果 ID `8x19360x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `8x19360x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G252 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.014640000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.014640000 | 129.9976 | 1.0000× | 0.015600000, 0.014640000, 0.016240001 | 10.93% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.016000001 | 118.9478 | 0.9150× | 0.016000001, 0.017519999, 0.017680001 | 10.50% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.017200001 | 110.6491 | 0.8512× | 0.019600000, 0.017519999, 0.017200001 | 13.95% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `8x19360x6144-fp16-G252-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `8x19360x6144-fp16-G252-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=3；原始结果 ID `8x19360x6144-fp16-G252-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。

<a id="mnk-8-129280-4096"></a>
## MNK 8 × 129280 × 4096

### 第一轮 BF16 · G253 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.173039913 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.107600003 | 78.7406 | 20.1955× | 0.109999999, 0.107600003, 0.107759997 | 2.23% |
| 2 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 0.107919998 | 78.5072 | 20.1357× | 0.111280002, 0.108240001, 0.107919998 | 3.11% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.108240001 | 78.2751 | 20.0761× | 0.111120000, 0.108240001, 0.108240001 | 2.66% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `8x129280x4096-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
- Rank 2：warmup=0，timed=3；原始结果 ID `8x129280x4096-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
- Rank 3：warmup=0，timed=3；原始结果 ID `8x129280x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
### 第二轮 FP16 · G253 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.058160000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.058160000 | 145.6756 | 1.0000× | 0.059280001, 0.058160000 | 1.93% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.061360002 | 138.0785 | 0.9478× | 0.067440003, 0.061360002, 0.063919999 | 9.91% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.075439997 | 112.3077 | 0.7709× | 0.081200004, 0.075439997, 0.075599998 | 7.64% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `8x129280x4096-fp16-G253-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=3；原始结果 ID `8x129280x4096-fp16-G253-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=3；原始结果 ID `8x129280x4096-fp16-G253-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0074`。

<a id="mnk-8-248320-4096"></a>
## MNK 8 × 248320 × 4096

### 第一轮 BF16 · G254 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.056000233 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.202480003 | 80.3729 | 20.0316× | 0.205679998, 0.202480003 | 1.58% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.218640000 | 74.4324 | 18.5510× | 0.220880002, 0.218640000 | 1.02% |
| 3 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 4.056000233 | 4.0123 | 1.0000× | 4.056000233, 4.072480202 | 0.41% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `8x248320x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 2：warmup=0，timed=2；原始结果 ID `8x248320x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 3：warmup=0，timed=2；原始结果 ID `8x248320x4096-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0009`。
### 第二轮 FP16 · G254 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.106480002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.106480002 | 152.8353 | 1.0000× | 0.106480002, 0.107120000 | 0.60% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.113360003 | 143.5594 | 0.9393× | 0.115440004, 0.113360003 | 1.83% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.141039997 | 115.3850 | 0.7550× | 0.142159998, 0.141039997 | 0.79% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `8x248320x4096-fp16-G254-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 2：warmup=0，timed=2；原始结果 ID `8x248320x4096-fp16-G254-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=2；原始结果 ID `8x248320x4096-fp16-G254-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0074`。

<a id="mnk-12-32-6144"></a>
## MNK 12 × 32 × 6144

### 第一轮 BF16 · G255 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.237920001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.006240000 | 0.7562 | 38.1282× | 0.010800000, 0.006240000, 0.006720000 | 73.08% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013120000 | 0.3596 | 18.1341× | 0.013120000, 0.014000000, 0.013200000 | 6.71% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.134320006 | 0.0351 | 1.7713× | 0.135600001, 0.134320006 | 0.95% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x32x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x32x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0057`。
- Rank 3：warmup=0，timed=2；原始结果 ID `12x32x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0033`。
### 第二轮 FP16 · G255 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.009920000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.008800000 | 0.5362 | 1.1273× | 0.011440000, 0.008800000, 0.009520000 | 30.00% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 0.4957 | 1.0420× | 0.011120000, 0.009520000, 0.009680000 | 16.81% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.009920000 | 0.4757 | 1.0000× | 0.011760000, 0.011280000, 0.009920000 | 18.55% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x32x6144-fp16-G255-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x32x6144-fp16-G255-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x32x6144-fp16-G255-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-12-64-2048"></a>
## MNK 12 × 64 × 2048

### 第一轮 BF16 · G256 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.083999999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008400000 | 0.3745 | 10.0000× | 0.008480000, 0.008400000 | 0.95% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013120000 | 0.2398 | 6.4024× | 0.013120000, 0.013840000, 0.013520000 | 5.49% |
| 3 | [f914f5100c7c0a6581fe](#r1-f914f5100c7c0a6581fe) | wdra48 · 64/64/128 | fine | 0.057599999 | 0.0546 | 1.4583× | 0.059680000, 0.057599999, 0.058079999 | 3.61% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `12x64x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x64x2048-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x64x2048-bf16-f914f5100c7c0a6581fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0120`。
### 第二轮 FP16 · G256 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.009920000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009200000 | 0.3419 | 1.0783× | 0.010560000, 0.009200000, 0.009840000 | 14.78% |
| 2 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 0.3304 | 1.0420× | 0.010240000, 0.009680000, 0.009520000 | 7.56% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.009920000 | 0.3171 | 1.0000× | 0.011760000, 0.010480000, 0.009920000 | 18.55% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x64x2048-fp16-G256-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x64x2048-fp16-G256-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x64x2048-fp16-G256-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-12-256-2048"></a>
## MNK 12 × 256 × 2048

### 第一轮 BF16 · G257 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.085680000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008560000 | 1.4700 | 10.0093× | 0.010320000, 0.008560000, 0.008560000 | 20.56% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013680000 | 0.9198 | 6.2632× | 0.015760001, 0.013680000, 0.013840000 | 15.20% |
| 3 | [9dbcc1f02010f1b928b8](#r1-9dbcc1f02010f1b928b8) | wdra48 · 64/64/128 | fine | 0.056079999 | 0.2244 | 1.5278× | 0.059280001, 0.057680000, 0.056079999 | 5.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x256x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0049`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x256x2048-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x256x2048-bf16-9dbcc1f02010f1b928b8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0099`。
### 第二轮 FP16 · G257 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.011120000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 1.3217 | 1.1681× | 0.011440000, 0.012240000, 0.009520000 | 28.57% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009680000 | 1.2999 | 1.1488× | 0.014000000, 0.009680000, 0.012240000 | 44.63% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.011120000 | 1.1316 | 1.0000× | 0.011600000, 0.012880000, 0.011120000 | 15.83% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x256x2048-fp16-G257-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x256x2048-fp16-G257-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x256x2048-fp16-G257-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-12-256-6144"></a>
## MNK 12 × 256 × 6144

### 第一轮 BF16 · G258 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.250479996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008080000 | 4.6719 | 31.0000× | 0.009680000, 0.008080000, 0.008080000 | 19.80% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.012880000 | 2.9308 | 19.4472× | 0.015760001, 0.013200000, 0.012880000 | 22.36% |
| 3 | [f914f5100c7c0a6581fe](#r1-f914f5100c7c0a6581fe) | wdra48 · 64/64/128 | fine | 0.149360001 | 0.2527 | 1.6770× | 0.169839993, 0.149360001, 0.150000006 | 13.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x256x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x256x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x256x6144-bf16-f914f5100c7c0a6581fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0120`。
### 第二轮 FP16 · G258 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.011120000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 3.9652 | 1.1681× | 0.013840000, 0.012560000, 0.009520000 | 45.38% |
| 2 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 3.9652 | 1.1681× | 0.013840000, 0.012240000, 0.009520000 | 45.38% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.011120000 | 3.3947 | 1.0000× | 0.013360000, 0.011120000, 0.012560000 | 20.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x256x6144-fp16-G258-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x256x6144-fp16-G258-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x256x6144-fp16-G258-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-12-1024-2048"></a>
## MNK 12 × 1024 × 2048

### 第一轮 BF16 · G259 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.097039998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008880000 | 5.6680 | 10.9279× | 0.009680000, 0.008880000, 0.009040000 | 9.01% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013680000 | 3.6792 | 7.0936× | 0.015440000, 0.014000000, 0.013680000 | 12.87% |
| 3 | [e9119b3c019fe173f6ed](#r1-e9119b3c019fe173f6ed) | wdra48 · 64/64/128 | fine | 0.063120000 | 0.7974 | 1.5374× | 0.064560004, 0.063120000, 0.063759997 | 2.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x1024x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x1024x2048-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x1024x2048-bf16-e9119b3c019fe173f6ed`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0099`。
### 第二轮 FP16 · G259 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010800000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 5.2869 | 1.1345× | 0.011280000, 0.012560000, 0.009520000 | 31.93% |
| 2 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 5.2869 | 1.1345× | 0.013520000, 0.009520000, 0.009680000 | 42.02% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010800000 | 4.6603 | 1.0000× | 0.011920000, 0.011440000, 0.010800000 | 10.37% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x1024x2048-fp16-G259-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x1024x2048-fp16-G259-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x1024x2048-fp16-G259-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-12-2048-512"></a>
## MNK 12 × 2048 × 512

### 第一轮 BF16 · G260 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.027600000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.009040000 | 2.7838 | 3.0531× | 0.009680000, 0.009040000, 0.009200000 | 7.08% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013520000 | 1.8614 | 2.0414× | 0.014960000, 0.013680000, 0.013520000 | 10.65% |
| 3 | [9dbcc1f02010f1b928b8](#r1-9dbcc1f02010f1b928b8) | wdra48 · 64/64/128 | fine | 0.023920000 | 1.0521 | 1.1538× | 0.026000001, 0.023920000, 0.024400000 | 8.70% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x2048x512-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x2048x512-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x2048x512-bf16-9dbcc1f02010f1b928b8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0100`。
### 第二轮 FP16 · G260 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.011120000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 2.6435 | 1.1681× | 0.013840000, 0.009840000, 0.009520000 | 45.38% |
| 2 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.009520000 | 2.6435 | 1.1681× | 0.011120000, 0.009520000, 0.012080000 | 26.89% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.011120000 | 2.2631 | 1.0000× | 0.013520000, 0.012880000, 0.011120000 | 21.58% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x2048x512-fp16-G260-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x2048x512-fp16-G260-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0178`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x2048x512-fp16-G260-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。

<a id="mnk-12-2048-4096"></a>
## MNK 12 × 2048 × 4096

### 第一轮 BF16 · G261 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.199760005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.009040000 | 22.2706 | 22.0973× | 0.010800000, 0.009520000, 0.009040000 | 19.47% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013040000 | 15.4392 | 15.3190× | 0.015280000, 0.013040000, 0.013520000 | 17.18% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.114320002 | 1.7611 | 1.7474× | 0.114320002, 0.116880000, 0.116880000 | 2.24% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x2048x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x2048x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x2048x4096-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0066`。
### 第二轮 FP16 · G261 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010960000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 21.1478 | 1.1513× | 0.014160000, 0.009520000, 0.009680000 | 48.74% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010960000 | 18.3692 | 1.0000× | 0.011760000, 0.010960000, 0.012720000 | 16.06% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.012880000 | 15.6309 | 0.8509× | 0.014000000, 0.012880000, 0.014960000 | 16.15% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x2048x4096-fp16-G261-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0138`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x2048x4096-fp16-G261-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0170`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x2048x4096-fp16-G261-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0187`。

<a id="mnk-12-6144-12288"></a>
## MNK 12 × 6144 × 12288

### 第一轮 BF16 · G262 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.711359978 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.009360000 | 193.5833 | 76.0000× | 0.010000000, 0.009360000, 0.009520000 | 6.84% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.012400000 | 146.1241 | 57.3677× | 0.015600000, 0.013840000, 0.012400000 | 25.81% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.347600013 | 5.2127 | 2.0465× | 0.353280008, 0.347600013 | 1.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x6144x12288-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x6144x12288-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=2；原始结果 ID `12x6144x12288-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0066`。
### 第二轮 FP16 · G262 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 190.3298 | 1.0840× | 0.011600000, 0.010480000, 0.009520000 | 21.85% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010320000 | 175.5755 | 1.0000× | 0.012080000, 0.010320000, 0.010320000 | 17.05% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.010960000 | 165.3229 | 0.9416× | 0.015120000, 0.012400000, 0.010960000 | 37.96% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x6144x12288-fp16-G262-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x6144x12288-fp16-G262-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x6144x12288-fp16-G262-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0114`。

<a id="mnk-12-9216-2048"></a>
## MNK 12 × 9216 × 2048

### 第一轮 BF16 · G263 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.148880005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.012880000 | 35.1696 | 11.5590× | 0.015760001, 0.013680000, 0.012880000 | 22.36% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.013520000 | 33.5048 | 11.0118× | 0.015440000, 0.013840000, 0.013520000 | 14.20% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.105999999 | 4.2734 | 1.4045× | 0.105999999, 0.107120000 | 1.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x9216x2048-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x9216x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 3：warmup=0，timed=2；原始结果 ID `12x9216x2048-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0026`。
### 第二轮 FP16 · G263 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010000000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010000000 | 45.2985 | 1.0000× | 0.014320000, 0.010320000, 0.010000000 | 43.20% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.010160000 | 44.5851 | 0.9843× | 0.011600000, 0.010320000, 0.010160000 | 14.17% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.012240000 | 37.0086 | 0.8170× | 0.014480000, 0.012880000, 0.012240000 | 18.30% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x9216x2048-fp16-G263-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x9216x2048-fp16-G263-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x9216x2048-fp16-G263-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0058`。

<a id="mnk-12-12288-2048"></a>
## MNK 12 × 12288 × 2048

### 第一轮 BF16 · G264 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.142959997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013360000 | 45.2081 | 10.7006× | 0.015440000, 0.013360000, 0.014000000 | 15.57% |
| 2 | [f914f5100c7c0a6581fe](#r1-f914f5100c7c0a6581fe) | wdra48 · 64/64/128 | fine | 0.014160000 | 42.6539 | 10.0960× | 0.016240001, 0.014640000, 0.014160000 | 14.69% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.014480000 | 41.7113 | 9.8729× | 0.015920000, 0.014480000, 0.014640000 | 9.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x12288x2048-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x12288x2048-bf16-f914f5100c7c0a6581fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0122`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x12288x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
### 第二轮 FP16 · G264 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010480000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.010000000 | 60.3980 | 1.0480× | 0.011600000, 0.010000000, 0.012880000 | 28.80% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010480000 | 57.6317 | 1.0000× | 0.011920000, 0.013200000, 0.010480000 | 25.95% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.011600000 | 52.0672 | 0.9034× | 0.015120000, 0.011600000, 0.014160000 | 30.34% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x12288x2048-fp16-G264-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x12288x2048-fp16-G264-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x12288x2048-fp16-G264-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。

<a id="mnk-12-154880-6144"></a>
## MNK 12 × 154880 × 6144

### 第一轮 BF16 · G265 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.940720081 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.126479998 | 180.5660 | 31.1569× | 0.129040003, 0.126479998, 0.126959994 | 2.02% |
| 2 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 0.127120003 | 179.6569 | 31.0000× | 0.130160004, 0.127120003, 0.127120003 | 2.39% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.127440006 | 179.2058 | 30.9222× | 0.129999995, 0.127440006, 0.127440006 | 2.01% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `12x154880x6144-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0122`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x154880x6144-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0122`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x154880x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
### 第二轮 FP16 · G265 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.067280002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.067280002 | 339.4469 | 1.0000× | 0.068080001, 0.067280002 | 1.19% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.071439996 | 319.6807 | 0.9418× | 0.077519998, 0.071439996, 0.071439996 | 8.51% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.090480000 | 252.4092 | 0.7436× | 0.090960003, 0.090480000 | 0.53% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `12x154880x6144-fp16-G265-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x154880x6144-fp16-G265-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=2；原始结果 ID `12x154880x6144-fp16-G265-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0075`。

<a id="mnk-12-248320-2048"></a>
## MNK 12 × 248320 × 2048

### 第一轮 BF16 · G266 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.035919905 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.201040000 | 60.7114 | 10.1269× | 0.204080001, 0.201040000 | 1.51% |
| 2 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 0.202319995 | 60.3273 | 10.0629× | 0.205520004, 0.202319995 | 1.58% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.202800006 | 60.1845 | 10.0391× | 0.205359995, 0.202800006 | 1.26% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `12x248320x2048-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0093`。
- Rank 2：warmup=0，timed=2；原始结果 ID `12x248320x2048-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0093`。
- Rank 3：warmup=0，timed=2；原始结果 ID `12x248320x2048-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
### 第二轮 FP16 · G266 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.107600003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.107600003 | 113.4333 | 1.0000× | 0.108560003, 0.107600003 | 0.89% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.111440003 | 109.5246 | 0.9655× | 0.116719998, 0.111440003, 0.111599997 | 4.74% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.139119998 | 87.7331 | 0.7734× | 0.144879997, 0.139119998, 0.139119998 | 4.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `12x248320x2048-fp16-G266-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `12x248320x2048-fp16-G266-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `12x248320x2048-fp16-G266-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0075`。

<a id="mnk-16-192-4096"></a>
## MNK 16 × 192 × 4096

### 第一轮 BF16 · G267 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.161200002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [e9119b3c019fe173f6ed](#r1-e9119b3c019fe173f6ed) | wdra48 · 64/64/128 | fine | 0.007600000 | 3.3113 | 21.2105× | 0.010480000, 0.008560000, 0.007600000 | 37.89% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.007920000 | 3.1775 | 20.3535× | 0.009840000, 0.009040000, 0.007920000 | 24.24% |
| 3 | [9dbcc1f02010f1b928b8](#r1-9dbcc1f02010f1b928b8) | wdra48 · 64/64/128 | fine | 0.008080000 | 3.1146 | 19.9505× | 0.010160000, 0.008080000, 0.008720000 | 25.74% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x192x4096-bf16-e9119b3c019fe173f6ed`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0122`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x192x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x192x4096-bf16-9dbcc1f02010f1b928b8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0122`。
### 第二轮 FP16 · G267 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010000000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009680000 | 2.5998 | 1.0331× | 0.011120000, 0.011920000, 0.009680000 | 23.14% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010000000 | 2.5166 | 1.0000× | 0.011280000, 0.010640000, 0.010000000 | 12.80% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.012080000 | 2.0833 | 0.8278× | 0.015440000, 0.012080000, 0.014800000 | 27.81% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x192x4096-fp16-G267-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x192x4096-fp16-G267-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x192x4096-fp16-G267-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。

<a id="mnk-16-384-7168"></a>
## MNK 16 × 384 × 7168

### 第一轮 BF16 · G268 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.337199986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f914f5100c7c0a6581fe](#r1-f914f5100c7c0a6581fe) | wdra48 · 64/64/128 | fine | 0.008080000 | 10.9010 | 41.7327× | 0.010000000, 0.008400000, 0.008080000 | 23.76% |
| 2 | [9dbcc1f02010f1b928b8](#r1-9dbcc1f02010f1b928b8) | wdra48 · 64/64/128 | fine | 0.008240000 | 10.6894 | 40.9223× | 0.010000000, 0.008240000, 0.008400000 | 21.36% |
| 3 | [e9119b3c019fe173f6ed](#r1-e9119b3c019fe173f6ed) | wdra48 · 64/64/128 | fine | 0.008240000 | 10.6894 | 40.9223× | 0.010000000, 0.008240000, 0.008240000 | 21.36% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x384x7168-bf16-f914f5100c7c0a6581fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0122`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x384x7168-bf16-9dbcc1f02010f1b928b8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x384x7168-bf16-e9119b3c019fe173f6ed`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
### 第二轮 FP16 · G268 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010480000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 9.2521 | 1.1008× | 0.011280000, 0.009520000, 0.009840000 | 18.49% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.010480000 | 8.4046 | 1.0000× | 0.017680001, 0.012560000, 0.010480000 | 68.70% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010640000 | 8.2782 | 0.9850× | 0.011600000, 0.013200000, 0.010640000 | 24.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x384x7168-fp16-G268-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x384x7168-fp16-G268-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x384x7168-fp16-G268-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。

<a id="mnk-16-2112-7168"></a>
## MNK 16 × 2112 × 7168

### 第一轮 BF16 · G269 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.356480002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008720000 | 55.5553 | 40.8807× | 0.010000000, 0.008720000, 0.009040000 | 14.68% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013680000 | 35.4124 | 26.0585× | 0.014800000, 0.013840000, 0.013680000 | 8.19% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.189520001 | 2.5562 | 1.8810× | 0.189520001, 0.190239996 | 0.38% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x2112x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x2112x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=2；原始结果 ID `16x2112x7168-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0066`。
### 第二轮 FP16 · G269 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010640000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 49.2319 | 1.0813× | 0.011440000, 0.009840000, 0.009840000 | 16.26% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010640000 | 45.5303 | 1.0000× | 0.013840000, 0.013360000, 0.010640000 | 30.08% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.012880000 | 37.6120 | 0.8261× | 0.014320000, 0.012880000, 0.012880000 | 11.18% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x2112x7168-fp16-G269-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x2112x7168-fp16-G269-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x2112x7168-fp16-G269-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0187`。

<a id="mnk-16-4096-7168"></a>
## MNK 16 × 4096 × 7168

### 第一轮 BF16 · G270 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.380400002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9dbcc1f02010f1b928b8](#r1-9dbcc1f02010f1b928b8) | wdra48 · 64/64/128 | fine | 0.008560000 | 109.7575 | 44.4393× | 0.010480000, 0.008560000, 0.008560000 | 22.43% |
| 2 | [e9119b3c019fe173f6ed](#r1-e9119b3c019fe173f6ed) | wdra48 · 64/64/128 | fine | 0.008560000 | 109.7575 | 44.4393× | 0.010480000, 0.008560000, 0.009200000 | 22.43% |
| 3 | [f914f5100c7c0a6581fe](#r1-f914f5100c7c0a6581fe) | wdra48 · 64/64/128 | fine | 0.008720000 | 107.7436 | 43.6239× | 0.010000000, 0.009040000, 0.008720000 | 14.68% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x4096x7168-bf16-9dbcc1f02010f1b928b8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x4096x7168-bf16-e9119b3c019fe173f6ed`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x4096x7168-bf16-f914f5100c7c0a6581fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
### 第二轮 FP16 · G270 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010640000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 95.4801 | 1.0813× | 0.014160000, 0.009840000, 0.012400000 | 43.90% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010640000 | 88.3011 | 1.0000× | 0.011280000, 0.013040000, 0.010640000 | 22.56% |
| 3 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.012400000 | 75.7681 | 0.8581× | 0.013680000, 0.012560000, 0.012400000 | 10.32% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x4096x7168-fp16-G270-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x4096x7168-fp16-G270-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x4096x7168-fp16-G270-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0099`。

<a id="mnk-16-7168-2048"></a>
## MNK 16 × 7168 × 2048

### 第一轮 BF16 · G271 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.130799994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.012720000 | 36.9310 | 10.2830× | 0.015120000, 0.013840000, 0.012720000 | 18.87% |
| 2 | [f914f5100c7c0a6581fe](#r1-f914f5100c7c0a6581fe) | wdra48 · 64/64/128 | fine | 0.012880000 | 36.4722 | 10.1553× | 0.015120000, 0.013200000, 0.012880000 | 17.39% |
| 3 | [3b67ff7fe683b39aa55a](#r1-3b67ff7fe683b39aa55a) | wdra88 · 128/128/64 | fine | 0.013040000 | 36.0247 | 10.0307× | 0.015440000, 0.013360000, 0.013040000 | 18.40% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x7168x2048-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x7168x2048-bf16-f914f5100c7c0a6581fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x7168x2048-bf16-3b67ff7fe683b39aa55a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0123`。
### 第二轮 FP16 · G271 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010960000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 49.3448 | 1.1513× | 0.011760000, 0.012560000, 0.009520000 | 31.93% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.010960000 | 42.8615 | 1.0000× | 0.015120000, 0.012240000, 0.010960000 | 37.96% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.011120000 | 42.2448 | 0.9856× | 0.011120000, 0.013200000, 0.012880000 | 18.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x7168x2048-fp16-G271-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x7168x2048-fp16-G271-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x7168x2048-fp16-G271-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。

<a id="mnk-16-7168-8192"></a>
## MNK 16 × 7168 × 8192

### 第一轮 BF16 · G272 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.508639991 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.012880000 | 145.8888 | 39.4907× | 0.015600000, 0.012880000, 0.013360000 | 21.12% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.012880000 | 145.8888 | 39.4907× | 0.015600000, 0.014160000, 0.012880000 | 21.12% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.355760008 | 5.2818 | 1.4297× | 0.357360005, 0.355760008 | 0.45% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x7168x8192-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x7168x8192-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 3：warmup=0，timed=2；原始结果 ID `16x7168x8192-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0026`。
### 第二轮 FP16 · G272 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010320000 | 182.0783 | 1.0000× | 0.011600000, 0.013360000, 0.010320000 | 29.46% |
| 2 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.011920000 | 157.6383 | 0.8658× | 0.014320000, 0.014800000, 0.011920000 | 24.16% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.012080000 | 155.5503 | 0.8543× | 0.013200000, 0.012400000, 0.012080000 | 9.27% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x7168x8192-fp16-G272-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x7168x8192-fp16-G272-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0059`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x7168x8192-fp16-G272-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。

<a id="mnk-16-7168-18432"></a>
## MNK 16 × 7168 × 18432

### 第一轮 BF16 · G273 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.137599945 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.012560000 | 336.6129 | 90.5732× | 0.015600000, 0.013680000, 0.012560000 | 24.20% |
| 2 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.012560000 | 336.6129 | 90.5732× | 0.014960000, 0.013200000, 0.012560000 | 19.11% |
| 3 | [4906f7e92d0963838f87](#r1-4906f7e92d0963838f87) | wdra48 · 64/64/128 | fine | 0.012720000 | 332.3788 | 89.4340× | 0.015120000, 0.012880000, 0.012720000 | 18.87% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x7168x18432-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x7168x18432-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0124`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x7168x18432-bf16-4906f7e92d0963838f87`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0124`。
### 第二轮 FP16 · G273 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.013200000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 444.1028 | 1.3866× | 0.011600000, 0.009840000, 0.009520000 | 21.85% |
| 2 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.012080000 | 349.9883 | 1.0927× | 0.013520000, 0.014960000, 0.012080000 | 23.84% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.012080000 | 349.9883 | 1.0927× | 0.014000000, 0.013040000, 0.012080000 | 15.89% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x7168x18432-fp16-G273-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x7168x18432-fp16-G273-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0099`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x7168x18432-fp16-G273-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0187`。

<a id="mnk-16-12288-1536"></a>
## MNK 16 × 12288 × 1536

### 第一轮 BF16 · G274 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.118639998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.013840000 | 43.6402 | 8.5723× | 0.015440000, 0.013840000, 0.014000000 | 11.56% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.014160000 | 42.6539 | 8.3785× | 0.016080000, 0.014640000, 0.014160000 | 13.56% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.090159997 | 6.6990 | 1.3159× | 0.091440000, 0.090159997 | 1.42% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x12288x1536-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0058`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x12288x1536-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 3：warmup=0，timed=2；原始结果 ID `16x12288x1536-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0026`。
### 第二轮 FP16 · G274 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010080000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 61.3801 | 1.0244× | 0.013840000, 0.012880000, 0.009840000 | 40.65% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010080000 | 59.9186 | 1.0000× | 0.014160000, 0.010160000, 0.010080000 | 40.48% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.010960000 | 55.1076 | 0.9197× | 0.017200001, 0.010960000, 0.010960000 | 56.93% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x12288x1536-fp16-G274-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x12288x1536-fp16-G274-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x12288x1536-fp16-G274-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。

<a id="mnk-16-16160-7168"></a>
## MNK 16 × 16160 × 7168

### 第一轮 BF16 · G275 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.516879976 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.019280000 | 192.2571 | 26.8091× | 0.020880001, 0.019280000, 0.019440001 | 8.30% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.022159999 | 167.2706 | 23.3249× | 0.024879999, 0.022159999, 0.022480000 | 12.27% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.473360002 | 7.8306 | 1.0919× | 0.480239987, 0.473360002 | 1.45% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x16160x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0050`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x16160x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
- Rank 3：warmup=0，timed=2；原始结果 ID `16x16160x7168-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0026`。
### 第二轮 FP16 · G275 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.014560000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.014560000 | 254.5822 | 1.0000× | 0.014560000, 0.016559999, 0.016240001 | 13.74% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.014800000 | 250.4538 | 0.9838× | 0.019120000, 0.017519999, 0.014800000 | 29.19% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.017039999 | 217.5303 | 0.8545× | 0.019440001, 0.017039999, 0.017039999 | 14.08% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x16160x7168-fp16-G275-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x16160x7168-fp16-G275-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x16160x7168-fp16-G275-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。

<a id="mnk-16-36864-7168"></a>
## MNK 16 × 36864 × 7168

### 第一轮 BF16 · G276 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.095280051 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 0.032400001 | 260.9789 | 33.8049× | 0.035440002, 0.032400001, 0.032880001 | 9.38% |
| 2 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.032400001 | 260.9789 | 33.8049× | 0.035440002, 0.032400001, 0.032720000 | 9.38% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.032720000 | 258.4266 | 33.4743× | 0.035760000, 0.032720000, 0.033199999 | 9.29% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x36864x7168-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0124`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x36864x7168-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0124`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x36864x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
### 第二轮 FP16 · G276 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.019760000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.019760000 | 427.9209 | 1.0000× | 0.021199999, 0.019760000, 0.020080000 | 7.29% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.021199999 | 398.8546 | 0.9321× | 0.030640000, 0.021199999, 0.023920000 | 44.53% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.025520001 | 331.3369 | 0.7743× | 0.029840000, 0.028240001, 0.025520001 | 16.93% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x36864x7168-fp16-G276-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x36864x7168-fp16-G276-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x36864x7168-fp16-G276-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。

<a id="mnk-16-120832-4096"></a>
## MNK 16 × 120832 × 4096

### 第一轮 BF16 · G277 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.073359966 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.098480001 | 160.8214 | 21.0536× | 0.101360001, 0.098480001, 0.098480001 | 2.92% |
| 2 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 0.098800004 | 160.3005 | 20.9854× | 0.101839997, 0.098800004, 0.099119999 | 3.08% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.099119999 | 159.7830 | 20.9177× | 0.101999998, 0.099119999, 0.099760003 | 2.91% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x120832x4096-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0125`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x120832x4096-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0124`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x120832x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
### 第二轮 FP16 · G277 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.052880000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.052880000 | 299.5025 | 1.0000× | 0.053840000, 0.052880000 | 1.82% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.056240000 | 281.6090 | 0.9403× | 0.060240000, 0.056240000, 0.056400001 | 7.11% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.069360003 | 228.3404 | 0.7624× | 0.071920000, 0.069360003, 0.071280003 | 3.69% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `16x120832x4096-fp16-G277-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x120832x4096-fp16-G277-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x120832x4096-fp16-G277-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0075`。

<a id="mnk-16-163840-7168"></a>
## MNK 16 × 163840 × 7168

### 第一轮 BF16 · G278 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.811759949 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.134639993 | 279.1218 | 35.7380× | 0.137999997, 0.134639993, 0.135440007 | 2.50% |
| 2 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 0.136079997 | 276.1682 | 35.3598× | 0.139760002, 0.136079997, 0.136559993 | 2.70% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.136240005 | 275.8438 | 35.3183× | 0.139280006, 0.136399999, 0.136240005 | 2.23% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `16x163840x7168-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0125`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x163840x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0125`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x163840x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
### 第二轮 FP16 · G278 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.072719999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.072719999 | 516.7899 | 1.0000× | 0.072719999, 0.074160002 | 1.98% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.076080002 | 493.9664 | 0.9558× | 0.080559999, 0.076080002, 0.079120003 | 5.89% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.094319999 | 398.4411 | 0.7710× | 0.100400001, 0.094319999, 0.096239999 | 6.45% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `16x163840x7168-fp16-G278-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `16x163840x7168-fp16-G278-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `16x163840x7168-fp16-G278-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0075`。

<a id="mnk-24-32-6144"></a>
## MNK 24 × 32 × 6144

### 第一轮 BF16 · G279 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.238240004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.006720000 | 1.4043 | 35.4524× | 0.008800000, 0.006720000, 0.006720000 | 30.95% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008080000 | 1.1680 | 29.4851× | 0.010320000, 0.008560000, 0.008080000 | 27.72% |
| 3 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.010240000 | 0.9216 | 23.2656× | 0.010240000, 0.011440000, 0.011440000 | 11.72% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x32x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x32x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x32x6144-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
### 第二轮 FP16 · G279 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.009600000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 0.9913 | 1.0084× | 0.011360000, 0.009680000, 0.009520000 | 19.33% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.009600000 | 0.9830 | 1.0000× | 0.010560000, 0.009600000, 0.010640000 | 10.83% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.011200000 | 0.8426 | 0.8571× | 0.014640000, 0.012560000, 0.011200000 | 30.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x32x6144-fp16-G279-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x32x6144-fp16-G279-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x32x6144-fp16-G279-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0187`。

<a id="mnk-24-256-6144"></a>
## MNK 24 × 256 × 6144

### 第一轮 BF16 · G280 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.257360011 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [e9119b3c019fe173f6ed](#r1-e9119b3c019fe173f6ed) | wdra48 · 64/64/128 | fine | 0.007760000 | 9.7291 | 33.1650× | 0.010640000, 0.008080000, 0.007760000 | 37.11% |
| 2 | [f1a574149008cfa85abd](#r1-f1a574149008cfa85abd) | wdra48 · 64/64/64 | fine | 0.007760000 | 9.7291 | 33.1650× | 0.010160000, 0.008240000, 0.007760000 | 30.93% |
| 3 | [b8391187ee0536d6c14a](#r1-b8391187ee0536d6c14a) | wdra48 · 64/64/128 | fine | 0.008080000 | 9.3437 | 31.8515× | 0.009840000, 0.008080000, 0.008880000 | 21.78% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x256x6144-bf16-e9119b3c019fe173f6ed`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0125`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x256x6144-bf16-f1a574149008cfa85abd`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0125`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x256x6144-bf16-b8391187ee0536d6c14a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
### 第二轮 FP16 · G280 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010000000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 7.9304 | 1.0504× | 0.013520000, 0.012720000, 0.009520000 | 42.02% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010000000 | 7.5497 | 1.0000× | 0.013680000, 0.010960000, 0.010000000 | 36.80% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.012080000 | 6.2498 | 0.8278× | 0.017200001, 0.012400000, 0.012080000 | 42.38% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x256x6144-fp16-G280-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0139`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x256x6144-fp16-G280-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x256x6144-fp16-G280-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0059`。

<a id="mnk-24-15104-4096"></a>
## MNK 24 × 15104 × 4096

### 第一轮 BF16 · G281 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.288879991 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.017999999 | 164.9760 | 16.0489× | 0.020400001, 0.018320000, 0.017999999 | 13.33% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.018640000 | 159.3115 | 15.4979× | 0.021360001, 0.018800000, 0.018640000 | 14.59% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.019280000 | 154.0232 | 14.9834× | 0.021360001, 0.019440001, 0.019280000 | 10.79% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x15104x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x15104x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x15104x4096-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
### 第二轮 FP16 · G281 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.014400000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.014400000 | 206.2200 | 1.0000× | 0.014400000, 0.016559999, 0.016559999 | 15.00% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.014960000 | 198.5005 | 0.9626× | 0.019120000, 0.017680001, 0.014960000 | 27.81% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.016880000 | 175.9222 | 0.8531× | 0.019280000, 0.016880000, 0.017039999 | 14.22% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x15104x4096-fp16-G281-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0171`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x15104x4096-fp16-G281-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x15104x4096-fp16-G281-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0075`。

<a id="mnk-24-154880-6144"></a>
## MNK 24 × 154880 × 6144

### 第一轮 BF16 · G282 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.929120064 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.101680003 | 449.2129 | 38.6420× | 0.104240000, 0.101839997, 0.101680003 | 2.52% |
| 2 | [37d075650464a6ce6f43](#r1-37d075650464a6ce6f43) | wasp44 · 128/128/64 | fine | 0.101680003 | 449.2129 | 38.6420× | 0.103919998, 0.101680003, 0.102159999 | 2.20% |
| 3 | [be5aa35e5bf6b2f23e8e](#r1-be5aa35e5bf6b2f23e8e) | wasp44 · 128/128/64 | fine | 0.101839997 | 448.5072 | 38.5813× | 0.104240000, 0.101839997, 0.101839997 | 2.36% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `24x154880x6144-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `24x154880x6144-bf16-37d075650464a6ce6f43`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
- Rank 3：warmup=0，timed=3；原始结果 ID `24x154880x6144-bf16-be5aa35e5bf6b2f23e8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
### 第二轮 FP16 · G282 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.067120001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.067120001 | 680.5121 | 1.0000× | 0.067919999, 0.067120001 | 1.19% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.074160002 | 615.9111 | 0.9051× | 0.075120002, 0.074160002 | 1.29% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.090159997 | 506.6102 | 0.7445× | 0.091279998, 0.090159997 | 1.24% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `24x154880x6144-fp16-G282-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=2；原始结果 ID `24x154880x6144-fp16-G282-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=2；原始结果 ID `24x154880x6144-fp16-G282-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0075`。

<a id="mnk-32-19360-6144"></a>
## MNK 32 × 19360 × 6144

### 第一轮 BF16 · G283 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.529600024 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.018320000 | 415.5383 | 28.9083× | 0.020560000, 0.018479999, 0.018320000 | 12.23% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.022320000 | 341.0691 | 23.7276× | 0.025520001, 0.022320000, 0.022320000 | 14.34% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.023920000 | 318.2551 | 22.1405× | 0.025680000, 0.023920000, 0.023920000 | 7.36% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `32x19360x6144-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `32x19360x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
- Rank 3：warmup=0，timed=3；原始结果 ID `32x19360x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
### 第二轮 FP16 · G283 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.014480000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.014480000 | 525.7363 | 1.0000× | 0.019920001, 0.014480000, 0.017360000 | 37.57% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.015120000 | 503.4829 | 0.9577× | 0.015120000, 0.016720001, 0.016880000 | 11.64% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.017519999 | 434.5127 | 0.8265× | 0.019280000, 0.017519999, 0.019920001 | 13.70% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `32x19360x6144-fp16-G283-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 2：warmup=0，timed=3；原始结果 ID `32x19360x6144-fp16-G283-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 3：warmup=0，timed=3；原始结果 ID `32x19360x6144-fp16-G283-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。

<a id="mnk-32-20480-7168"></a>
## MNK 32 × 20480 × 7168

### 第一轮 BF16 · G284 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.625039995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.018320000 | 512.8407 | 34.1179× | 0.020560000, 0.018959999, 0.018320000 | 12.23% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.022480000 | 417.9378 | 27.8043× | 0.025520001, 0.023119999, 0.022480000 | 13.52% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.023760000 | 395.4226 | 26.3064× | 0.025680000, 0.023760000, 0.024240000 | 8.08% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `32x20480x7168-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `32x20480x7168-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
- Rank 3：warmup=0，timed=3；原始结果 ID `32x20480x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
### 第二轮 FP16 · G284 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.013840000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.013840000 | 678.8469 | 1.0000× | 0.014960000, 0.013840000, 0.014640000 | 8.09% |
| 2 | [f89f00abebc2a438c6a0](#r2-f89f00abebc2a438c6a0) | nowasp · 128/128/64 | fine | 0.014000000 | 671.0886 | 0.9886× | 0.014880000, 0.014000000, 0.016559999 | 18.29% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.014800000 | 634.8136 | 0.9351× | 0.016559999, 0.014800000, 0.014960000 | 11.89% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `32x20480x7168-fp16-G284-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `32x20480x7168-fp16-G284-f89f00abebc2a438c6a0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0213`。
- Rank 3：warmup=0，timed=3；原始结果 ID `32x20480x7168-fp16-G284-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。

<a id="mnk-48-32-6144"></a>
## MNK 48 × 32 × 6144

### 第一轮 BF16 · G285 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.244959995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.006400000 | 2.9491 | 38.2750× | 0.010160000, 0.006720000, 0.006400000 | 58.75% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008240000 | 2.2906 | 29.7282× | 0.008320000, 0.008240000 | 0.97% |
| 3 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.009280000 | 2.0339 | 26.3966× | 0.010400000, 0.009280000, 0.009280000 | 12.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x32x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
- Rank 2：warmup=0，timed=2；原始结果 ID `48x32x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x32x6144-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
### 第二轮 FP16 · G285 · 输出 FP16

已记录 32 项，剩余 0 项；最优粗调 nowasp 基线 0.010480000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.008800000 | 2.1448 | 1.1909× | 0.011600000, 0.008800000, 0.010160000 | 31.82% |
| 2 | [cd3eb736d84b1dffa424](#r2-cd3eb736d84b1dffa424) | wdra48 · 128/128/64 | fine | 0.008960000 | 2.1065 | 1.1696× | 0.011760000, 0.008960000, 0.009520000 | 31.25% |
| 3 | [2102b4428dc21ca7e8dc](#r2-2102b4428dc21ca7e8dc) | wdra48 · 128/128/64 | fine | 0.008960000 | 2.1065 | 1.1696× | 0.011600000, 0.008960000, 0.010000000 | 29.46% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x32x6144-fp16-G285-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x32x6144-fp16-G285-cd3eb736d84b1dffa424`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0213`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x32x6144-fp16-G285-2102b4428dc21ca7e8dc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0213`。

<a id="mnk-48-256-6144"></a>
## MNK 48 × 256 × 6144

### 第一轮 BF16 · G286 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.273360014 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [ab18f5bc5035fd03b9d3](#r1-ab18f5bc5035fd03b9d3) | wdra48 · 64/64/128 | fine | 0.007760000 | 19.4581 | 35.2268× | 0.010000000, 0.008560000, 0.007760000 | 28.87% |
| 2 | [4906f7e92d0963838f87](#r1-4906f7e92d0963838f87) | wdra48 · 64/64/128 | fine | 0.008080000 | 18.6875 | 33.8317× | 0.009840000, 0.008560000, 0.008080000 | 21.78% |
| 3 | [b8391187ee0536d6c14a](#r1-b8391187ee0536d6c14a) | wdra48 · 64/64/128 | fine | 0.008080000 | 18.6875 | 33.8317× | 0.010160000, 0.008080000, 0.008400000 | 25.74% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x256x6144-bf16-ab18f5bc5035fd03b9d3`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x256x6144-bf16-4906f7e92d0963838f87`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x256x6144-bf16-b8391187ee0536d6c14a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
### 第二轮 FP16 · G286 · 输出 FP16

已记录 25 项，剩余 7 项；最优粗调 nowasp 基线 0.010320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.009840000 | 15.3450 | 1.0488× | 0.013840000, 0.011920000, 0.009840000 | 40.65% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 15.3450 | 1.0488× | 0.013200000, 0.012240000, 0.009840000 | 34.15% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010320000 | 14.6313 | 1.0000× | 0.011440000, 0.010320000, 0.010320000 | 10.85% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x256x6144-fp16-G286-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x256x6144-fp16-G286-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x256x6144-fp16-G286-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。

<a id="mnk-48-512-4096"></a>
## MNK 48 × 512 × 4096

### 第一轮 BF16 · G287 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.187199995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [05c5c97400044a95f5e6](#r1-05c5c97400044a95f5e6) | wdra48 · 64/64/64 | fine | 0.008080000 | 24.9167 | 23.1683× | 0.011120000, 0.008080000, 0.009360000 | 37.62% |
| 2 | [46a1fbc6df36ffed1a49](#r1-46a1fbc6df36ffed1a49) | wdra48 · 64/64/64 | fine | 0.008080000 | 24.9167 | 23.1683× | 0.010320000, 0.008400000, 0.008080000 | 27.72% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008240000 | 24.4328 | 22.7184× | 0.010160000, 0.008560000, 0.008240000 | 23.30% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x512x4096-bf16-05c5c97400044a95f5e6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0126`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x512x4096-bf16-46a1fbc6df36ffed1a49`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0127`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x512x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
### 第二轮 FP16 · G287 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.010640000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009840000 | 20.4600 | 1.0813× | 0.011760000, 0.009840000, 0.009840000 | 19.51% |
| 2 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.010000000 | 20.1327 | 1.0640× | 0.011120000, 0.012080000, 0.010000000 | 20.80% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010640000 | 18.9217 | 1.0000× | 0.011120000, 0.010640000, 0.010640000 | 4.51% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x512x4096-fp16-G287-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x512x4096-fp16-G287-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x512x4096-fp16-G287-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。

<a id="mnk-48-4096-8192"></a>
## MNK 48 × 4096 × 8192

### 第一轮 BF16 · G288 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.448720008 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.008240000 | 390.9254 | 54.4563× | 0.010160000, 0.009200000, 0.008240000 | 23.30% |
| 2 | [ab18f5bc5035fd03b9d3](#r1-ab18f5bc5035fd03b9d3) | wdra48 · 64/64/128 | fine | 0.008240000 | 390.9254 | 54.4563× | 0.010160000, 0.009200000, 0.008240000 | 23.30% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008560000 | 376.3114 | 52.4206× | 0.010640000, 0.008560000, 0.009040000 | 24.30% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x4096x8192-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x4096x8192-bf16-ab18f5bc5035fd03b9d3`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0127`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x4096x8192-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
### 第二轮 FP16 · G288 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.010320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.010000000 | 322.1226 | 1.0320× | 0.011440000, 0.010000000, 0.011440000 | 14.40% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010320000 | 312.1342 | 1.0000× | 0.011280000, 0.010320000, 0.013360000 | 29.46% |
| 3 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.012400000 | 259.7762 | 0.8323× | 0.014320000, 0.012400000, 0.015120000 | 21.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x4096x8192-fp16-G288-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x4096x8192-fp16-G288-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x4096x8192-fp16-G288-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0100`。

<a id="mnk-48-124160-4096"></a>
## MNK 48 × 124160 × 4096

### 第一轮 BF16 · G289 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.078560114 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.085680000 | 569.8144 | 24.2596× | 0.088399999, 0.086479999, 0.085680000 | 3.17% |
| 2 | [37d075650464a6ce6f43](#r1-37d075650464a6ce6f43) | wasp44 · 128/128/64 | fine | 0.085840002 | 568.7523 | 24.2144× | 0.088079996, 0.085840002, 0.085840002 | 2.61% |
| 3 | [be5aa35e5bf6b2f23e8e](#r1-be5aa35e5bf6b2f23e8e) | wasp44 · 128/128/64 | fine | 0.086159997 | 566.6400 | 24.1244× | 0.087920003, 0.086159997, 0.086159997 | 2.04% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x124160x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x124160x4096-bf16-37d075650464a6ce6f43`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0127`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x124160x4096-bf16-be5aa35e5bf6b2f23e8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0127`。
### 第二轮 FP16 · G289 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.056400001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.056400001 | 865.6329 | 1.0000× | 0.058479998, 0.056400001, 0.057200000 | 3.69% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.063919999 | 763.7938 | 0.8824× | 0.064879999, 0.063919999 | 1.50% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.082960002 | 588.4968 | 0.6798× | 0.085280001, 0.083439998, 0.082960002 | 2.80% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x124160x4096-fp16-G289-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=2；原始结果 ID `48x124160x4096-fp16-G289-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=3；原始结果 ID `48x124160x4096-fp16-G289-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-48-154880-6144"></a>
## MNK 48 × 154880 × 6144

### 第一轮 BF16 · G290 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.918720007 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.101999998 | 895.6073 | 38.4188× | 0.104400001, 0.101999998, 0.101999998 | 2.35% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.127120003 | 718.6276 | 30.8269× | 0.130480006, 0.127120003, 0.127120003 | 2.64% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.139760002 | 653.6344 | 28.0389× | 0.141680002, 0.139760002 | 1.37% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x154880x6144-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x154880x6144-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0059`。
- Rank 3：warmup=0，timed=2；原始结果 ID `48x154880x6144-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
### 第二轮 FP16 · G290 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.067120001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.067120001 | 1361.0241 | 1.0000× | 0.068240002, 0.070000000, 0.067120001 | 4.29% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.071120001 | 1284.4761 | 0.9438× | 0.077200003, 0.071120001, 0.074000001 | 8.55% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.100879997 | 905.5506 | 0.6653× | 0.100960001, 0.100879997 | 0.08% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `48x154880x6144-fp16-G290-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `48x154880x6144-fp16-G290-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=2；原始结果 ID `48x154880x6144-fp16-G290-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-64-64-2048"></a>
## MNK 64 × 64 × 2048

### 第一轮 BF16 · G291 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.086879998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.006400000 | 2.6214 | 13.5750× | 0.008960000, 0.006400000, 0.008080000 | 40.00% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.006400000 | 2.6214 | 13.5750× | 0.007680000, 0.006720000, 0.006400000 | 20.00% |
| 3 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.010240000 | 1.6384 | 8.4844× | 0.010240000, 0.010960000, 0.011600000 | 13.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x64x2048-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x64x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x64x2048-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
### 第二轮 FP16 · G291 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.009440000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.008800000 | 1.9065 | 1.0727× | 0.011760000, 0.010160000, 0.008800000 | 33.64% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.008960000 | 1.8725 | 1.0536× | 0.010560000, 0.009680000, 0.008960000 | 17.86% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.009440000 | 1.7772 | 1.0000× | 0.010960000, 0.009760000, 0.009440000 | 16.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x64x2048-fp16-G291-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x64x2048-fp16-G291-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x64x2048-fp16-G291-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。

<a id="mnk-64-256-2048"></a>
## MNK 64 × 256 × 2048

### 第一轮 BF16 · G292 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.089199997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.008240000 | 8.1443 | 10.8252× | 0.010000000, 0.008240000, 0.008240000 | 21.36% |
| 2 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.008560000 | 7.8398 | 10.4206× | 0.009840000, 0.008560000, 0.008560000 | 14.95% |
| 3 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.010800000 | 6.2138 | 8.2593× | 0.012560000, 0.011120000, 0.010800000 | 16.30% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x256x2048-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x256x2048-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x256x2048-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
### 第二轮 FP16 · G292 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.010160000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 7.0493 | 1.0672× | 0.011280000, 0.009520000, 0.009520000 | 18.49% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010160000 | 6.6052 | 1.0000× | 0.011440000, 0.010160000, 0.010960000 | 12.60% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.011600000 | 5.7852 | 0.8759× | 0.011760000, 0.012240000, 0.011600000 | 5.52% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x256x2048-fp16-G292-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x256x2048-fp16-G292-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x256x2048-fp16-G292-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-64-16160-4096"></a>
## MNK 64 × 16160 × 4096

### 第一轮 BF16 · G293 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.309280008 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.017999999 | 470.6941 | 17.1822× | 0.020400001, 0.018640000, 0.017999999 | 13.33% |
| 2 | [37d075650464a6ce6f43](#r1-37d075650464a6ce6f43) | wasp44 · 128/128/64 | fine | 0.017999999 | 470.6941 | 17.1822× | 0.020400001, 0.017999999, 0.018479999 | 13.33% |
| 3 | [be5aa35e5bf6b2f23e8e](#r1-be5aa35e5bf6b2f23e8e) | wasp44 · 128/128/64 | fine | 0.018160000 | 466.5470 | 17.0308× | 0.020560000, 0.018320000, 0.018160000 | 13.22% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x16160x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x16160x4096-bf16-37d075650464a6ce6f43`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x16160x4096-bf16-be5aa35e5bf6b2f23e8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
### 第二轮 FP16 · G293 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.014560000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.014560000 | 581.9021 | 1.0000× | 0.014560000, 0.016559999, 0.016559999 | 13.74% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.014960000 | 566.3432 | 0.9733× | 0.019600000, 0.017680001, 0.014960000 | 31.02% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.016880000 | 501.9250 | 0.8626× | 0.021520000, 0.016880000, 0.016880000 | 27.49% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x16160x4096-fp16-G293-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x16160x4096-fp16-G293-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x16160x4096-fp16-G293-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-64-31040-4096"></a>
## MNK 64 × 31040 × 4096

### 第一轮 BF16 · G294 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.570959985 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.026319999 | 618.3093 | 21.6930× | 0.027920000, 0.026319999, 0.026319999 | 6.08% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.034000002 | 478.6441 | 16.7929× | 0.036400001, 0.034000002, 0.034000002 | 7.06% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.034639999 | 469.8008 | 16.4827× | 0.035920002, 0.034639999, 0.034960002 | 3.70% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x31040x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x31040x4096-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x31040x4096-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
### 第二轮 FP16 · G294 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.019440001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.019440001 | 837.1347 | 1.0000× | 0.021040000, 0.022320000, 0.019440001 | 14.81% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.020560000 | 791.5321 | 0.9455× | 0.026960000, 0.021360001, 0.020560000 | 31.13% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.024720000 | 658.3293 | 0.7864× | 0.027120000, 0.024720000, 0.027600000 | 11.65% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `64x31040x4096-fp16-G294-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x31040x4096-fp16-G294-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=3；原始结果 ID `64x31040x4096-fp16-G294-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。

<a id="mnk-64-248320-2048"></a>
## MNK 64 × 248320 × 2048

### 第一轮 BF16 · G295 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.042399883 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [be5aa35e5bf6b2f23e8e](#r1-be5aa35e5bf6b2f23e8e) | wasp44 · 128/128/64 | fine | 0.161520004 | 403.0188 | 12.6449× | 0.163920000, 0.161520004 | 1.49% |
| 2 | [37d075650464a6ce6f43](#r1-37d075650464a6ce6f43) | wasp44 · 128/128/64 | fine | 0.161840007 | 402.2219 | 12.6199× | 0.164079994, 0.161840007 | 1.38% |
| 3 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.162000000 | 401.8247 | 12.6074× | 0.164079994, 0.162000000 | 1.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `64x248320x2048-bf16-be5aa35e5bf6b2f23e8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0093`。
- Rank 2：warmup=0，timed=2；原始结果 ID `64x248320x2048-bf16-37d075650464a6ce6f43`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0093`。
- Rank 3：warmup=0，timed=2；原始结果 ID `64x248320x2048-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
### 第二轮 FP16 · G295 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.104319997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.104319997 | 623.9992 | 1.0000× | 0.105520003, 0.104319997 | 1.15% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.111120000 | 585.8135 | 0.9388× | 0.115120001, 0.111120000, 0.113839999 | 3.60% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.157839999 | 412.4151 | 0.6609× | 0.159679994, 0.157839999 | 1.17% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `64x248320x2048-fp16-G295-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `64x248320x2048-fp16-G295-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=2；原始结果 ID `64x248320x2048-fp16-G295-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-80-129280-7168"></a>
## MNK 80 × 129280 × 7168

### 第一轮 BF16 · G296 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.611199856 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.086479999 | 1714.4848 | 53.3210× | 0.088560000, 0.086479999, 0.086800002 | 2.41% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.228239998 | 649.6173 | 20.2033× | 0.231120005, 0.228239998 | 1.26% |
| 3 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.230000004 | 644.6463 | 20.0487× | 0.231920004, 0.230000004 | 0.83% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `80x129280x7168-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 2：warmup=0，timed=2；原始结果 ID `80x129280x7168-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0051`。
- Rank 3：warmup=0，timed=2；原始结果 ID `80x129280x7168-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
### 第二轮 FP16 · G296 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.057519998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.057519998 | 2577.6886 | 1.0000× | 0.058800001, 0.060400002, 0.057519998 | 5.01% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.061360002 | 2416.3729 | 0.9374× | 0.065360002, 0.061360002, 0.064080000 | 6.52% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.084080003 | 1763.4234 | 0.6841× | 0.086000003, 0.084080003, 0.086159997 | 2.47% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `80x129280x7168-fp16-G296-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `80x129280x7168-fp16-G296-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=3；原始结果 ID `80x129280x7168-fp16-G296-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-96-64-7168"></a>
## MNK 96 × 64 × 7168

### 第一轮 BF16 · G297 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.299519986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.008080000 | 10.9010 | 37.0693× | 0.010320000, 0.008560000, 0.008080000 | 27.72% |
| 2 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.009280000 | 9.4914 | 32.2759× | 0.010240000, 0.009280000, 0.011280000 | 21.55% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.184719995 | 0.4768 | 1.6215× | 0.197999999, 0.185360000, 0.184719995 | 7.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x64x7168-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0035`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x64x7168-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0019`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x64x7168-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0068`。
### 第二轮 FP16 · G297 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.009600000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.008800000 | 10.0091 | 1.0909× | 0.010560000, 0.010000000, 0.008800000 | 20.00% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.008800000 | 10.0091 | 1.0909× | 0.011920000, 0.009520000, 0.008800000 | 35.45% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.009600000 | 9.1750 | 1.0000× | 0.011280000, 0.010800000, 0.009600000 | 17.50% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x64x7168-fp16-G297-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x64x7168-fp16-G297-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x64x7168-fp16-G297-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。

<a id="mnk-96-512-7168"></a>
## MNK 96 × 512 × 7168

### 第一轮 BF16 · G298 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.347840011 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.186320007 | 3.7819 | 1.8669× | 0.189040005, 0.186320007 | 1.46% |
| 2 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.188879997 | 3.7306 | 1.8416× | 0.188879997, 0.191119999 | 1.19% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.189359993 | 3.7212 | 1.8369× | 0.189359993, 0.190479994 | 0.59% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `96x512x7168-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0083`。
- Rank 2：warmup=0，timed=2；原始结果 ID `96x512x7168-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0082`。
- Rank 3：warmup=0，timed=2；原始结果 ID `96x512x7168-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0068`。
### 第二轮 FP16 · G298 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.010640000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009520000 | 74.0171 | 1.1176× | 0.011440000, 0.012240000, 0.009520000 | 28.57% |
| 2 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.010000000 | 70.4643 | 1.0640× | 0.013840000, 0.010160000, 0.010000000 | 38.40% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010640000 | 66.2259 | 1.0000× | 0.014000000, 0.013040000, 0.010640000 | 31.58% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x512x7168-fp16-G298-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x512x7168-fp16-G298-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x512x7168-fp16-G298-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
### 第一轮 BF16 · G405 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.356480002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3f9697ffe28602a898a4](#r1-3f9697ffe28602a898a4) | wdra48 · 64/64/128 | fine | 0.008720000 | 80.8077 | 40.8807× | 0.010800000, 0.008720000, 0.008880000 | 23.85% |
| 2 | [455ec4bcfd3552d92371](#r1-455ec4bcfd3552d92371) | wdra48 · 64/64/128 | fine | 0.008880000 | 79.3517 | 40.1441× | 0.010640000, 0.008880000, 0.008880000 | 19.82% |
| 3 | [3349c6b9c585549f7448](#r1-3349c6b9c585549f7448) | wdra48 · 64/64/128 | fine | 0.009360000 | 75.2824 | 38.0855× | 0.010640000, 0.009360000, 0.009520000 | 13.68% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x512x7168-fp32-3f9697ffe28602a898a4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x512x7168-fp32-455ec4bcfd3552d92371`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x512x7168-fp32-3349c6b9c585549f7448`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
### 第二轮 FP16 · G405 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.270080000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.189520001 | 3.7180 | 1.4251× | 0.189679995, 0.189520001 | 0.08% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.197840005 | 3.5617 | 1.3651× | 0.207120001, 0.197840005, 0.197919995 | 4.69% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.270080000 | 2.6090 | 1.0000× | 0.276879996, 0.270080000, 0.274960011 | 2.52% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `96x512x7168-fp16-G405-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0072`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x512x7168-fp16-G405-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x512x7168-fp16-G405-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0136`。

<a id="mnk-96-1024-7168"></a>
## MNK 96 × 1024 × 7168

### 第一轮 BF16 · G406 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.357520014 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [455ec4bcfd3552d92371](#r1-455ec4bcfd3552d92371) | wdra48 · 64/64/128 | fine | 0.008880000 | 158.7034 | 40.2613× | 0.010640000, 0.009680000, 0.008880000 | 19.82% |
| 2 | [3f9697ffe28602a898a4](#r1-3f9697ffe28602a898a4) | wdra48 · 64/64/128 | fine | 0.009360000 | 150.5648 | 38.1966× | 0.010640000, 0.009520000, 0.009360000 | 13.68% |
| 3 | [3349c6b9c585549f7448](#r1-3349c6b9c585549f7448) | wdra48 · 64/64/128 | fine | 0.009360000 | 150.5648 | 38.1966× | 0.010960000, 0.009360000, 0.009360000 | 17.09% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x1024x7168-fp32-455ec4bcfd3552d92371`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x1024x7168-fp32-3f9697ffe28602a898a4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x1024x7168-fp32-3349c6b9c585549f7448`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
### 第二轮 FP16 · G406 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.271519989 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.189840004 | 7.4235 | 1.4303× | 0.190479994, 0.189840004 | 0.34% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.204640001 | 6.8867 | 1.3268× | 0.204640001, 0.206959993 | 1.13% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.271519989 | 5.1904 | 1.0000× | 0.275440007, 0.271519989 | 1.44% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `96x1024x7168-fp16-G406-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0072`。
- Rank 2：warmup=0，timed=2；原始结果 ID `96x1024x7168-fp16-G406-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 3：warmup=0，timed=2；原始结果 ID `96x1024x7168-fp16-G406-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0136`。

<a id="mnk-96-2048-7168"></a>
## MNK 96 × 2048 × 7168

### 第一轮 BF16 · G407 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.364879996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3ed295572d5209a7ecdd](#r1-3ed295572d5209a7ecdd) | wdra48 · 64/64/128 | fine | 0.008880000 | 317.4068 | 41.0901× | 0.010800000, 0.009040000, 0.008880000 | 21.62% |
| 2 | [3f9697ffe28602a898a4](#r1-3f9697ffe28602a898a4) | wdra48 · 64/64/128 | fine | 0.009200000 | 306.3665 | 39.6609× | 0.010960000, 0.009360000, 0.009200000 | 19.13% |
| 3 | [3349c6b9c585549f7448](#r1-3349c6b9c585549f7448) | wdra48 · 64/64/128 | fine | 0.009200000 | 306.3665 | 39.6609× | 0.010960000, 0.009680000, 0.009200000 | 19.13% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x2048x7168-fp32-3ed295572d5209a7ecdd`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x2048x7168-fp32-3f9697ffe28602a898a4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x2048x7168-fp32-3349c6b9c585549f7448`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0091`。
### 第二轮 FP16 · G407 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.279920012 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.191919997 | 14.6862 | 1.4585× | 0.191919997, 0.191919997 | 0.00% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.206320003 | 13.6612 | 1.3567× | 0.207279995, 0.206320003 | 0.47% |
| 3 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.279439986 | 10.0865 | 1.0017× | 0.281199992, 0.279439986 | 0.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `96x2048x7168-fp16-G407-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0072`。
- Rank 2：warmup=0，timed=2；原始结果 ID `96x2048x7168-fp16-G407-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 3：warmup=0，timed=2；原始结果 ID `96x2048x7168-fp16-G407-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。

<a id="mnk-96-7168-21504"></a>
## MNK 96 × 7168 × 21504

### 第一轮 BF16 · G299 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.303679943 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.378639996 | 78.1613 | 3.4431× | 0.378639996, 1.141600013, 1.137359977 | 201.50% |
| 2 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.572960019 | 51.6528 | 2.2753× | 0.576560020, 0.572960019 | 0.63% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 1.036800027 | 28.5446 | 1.2574× | 1.036800027, 1.036880016 | 0.01% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x7168x21504-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `96x7168x21504-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0052`。
- Rank 3：warmup=0，timed=2；原始结果 ID `96x7168x21504-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0028`。
### 第二轮 FP16 · G299 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.010640000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.009360000 | 3161.8601 | 1.1368× | 0.011920000, 0.009840000, 0.009360000 | 27.35% |
| 2 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.010480000 | 2823.9513 | 1.0153× | 0.011440000, 0.010480000, 0.011760000 | 12.21% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.010640000 | 2781.4858 | 1.0000× | 0.014320000, 0.010640000, 0.013200000 | 34.59% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `96x7168x21504-fp16-G299-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0140`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x7168x21504-fp16-G299-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x7168x21504-fp16-G299-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。

<a id="mnk-96-129280-7168"></a>
## MNK 96 × 129280 × 7168

### 第一轮 BF16 · G300 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.637199879 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 4.116000175 | 43.2270 | 1.1266× | 4.118720055, 4.116000175 | 0.07% |
| 2 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 4.159679890 | 42.7731 | 1.1148× | 4.159679890, 4.169839859 | 0.24% |
| 3 | [2943f23873d8087254e4](#r1-2943f23873d8087254e4) | wdra88 · 128/128/64 | fine | 4.161920071 | 42.7501 | 1.1142× | 4.166160107, 4.161920071 | 0.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `96x129280x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0106`。
- Rank 2：warmup=0，timed=2；原始结果 ID `96x129280x7168-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0106`。
- Rank 3：warmup=0，timed=2；原始结果 ID `96x129280x7168-bf16-2943f23873d8087254e4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0106`。
### 第二轮 FP16 · G300 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.057680000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.057680000 | 3084.6459 | 1.0000× | 0.058320001, 0.057680000 | 1.11% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.063919999 | 2783.5166 | 0.9024× | 0.067120001, 0.064240001, 0.063919999 | 5.01% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.084080003 | 2116.1081 | 0.6860× | 0.088880002, 0.084080003, 0.084080003 | 5.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `96x129280x7168-fp16-G300-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=3；原始结果 ID `96x129280x7168-fp16-G300-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=3；原始结果 ID `96x129280x7168-fp16-G300-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-160-129280-4096"></a>
## MNK 160 × 129280 × 4096

### 第一轮 BF16 · G301 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.915599823 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 4.726560116 | 35.8506 | 1.0400× | 4.726560116, 4.758560181 | 0.68% |
| 2 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 4.787759781 | 35.3923 | 1.0267× | 4.787759781, 4.799680233 | 0.25% |
| 3 | [2943f23873d8087254e4](#r1-2943f23873d8087254e4) | wdra88 · 128/128/64 | fine | 4.868879795 | 34.8026 | 1.0096× | 4.869359970, 4.868879795 | 0.01% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `160x129280x4096-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0128`。
- Rank 2：warmup=0，timed=2；原始结果 ID `160x129280x4096-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0129`。
- Rank 3：warmup=0，timed=2；原始结果 ID `160x129280x4096-bf16-2943f23873d8087254e4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0129`。
### 第二轮 FP16 · G301 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.109999999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.109999999 | 1540.4535 | 1.0000× | 0.109999999, 0.111759998 | 1.60% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.118479997 | 1430.1982 | 0.9284× | 0.120080002, 0.118479997 | 1.35% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.164719999 | 1028.7147 | 0.6678× | 0.168559998, 0.164719999, 0.164719999 | 2.33% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `160x129280x4096-fp16-G301-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0172`。
- Rank 2：warmup=0，timed=2；原始结果 ID `160x129280x4096-fp16-G301-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=3；原始结果 ID `160x129280x4096-fp16-G301-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。

<a id="mnk-192-64-4096"></a>
## MNK 192 × 64 × 4096

### 第一轮 BF16 · G302 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.167760000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.103759997 | 0.9702 | 1.6168× | 0.107600003, 0.105360001, 0.103759997 | 3.70% |
| 2 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.111599997 | 0.9020 | 1.5032× | 0.119120002, 0.111599997, 0.114639997 | 6.74% |
| 3 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.113760002 | 0.8849 | 1.4747× | 0.114480004, 0.113760002 | 0.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `192x64x4096-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0094`。
- Rank 2：warmup=0，timed=3；原始结果 ID `192x64x4096-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0068`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x64x4096-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0094`。
### 第二轮 FP16 · G302 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.141200006 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.010000000 | 10.0663 | 14.1200× | 0.011440000, 0.010000000, 0.011600000 | 16.00% |
| 2 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.011920000 | 8.4449 | 11.8456× | 0.013360000, 0.011920000, 0.014640000 | 22.82% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.013040000 | 7.7196 | 10.8282× | 0.014640000, 0.013040000, 0.014320000 | 12.27% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `192x64x4096-fp16-G302-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=3；原始结果 ID `192x64x4096-fp16-G302-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0100`。
- Rank 3：warmup=0，timed=3；原始结果 ID `192x64x4096-fp16-G302-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0060`。

<a id="mnk-192-512-4096"></a>
## MNK 192 × 512 × 4096

### 第一轮 BF16 · G303 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.194240004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.111280002 | 7.2368 | 1.7455× | 0.112400003, 0.111280002 | 1.01% |
| 2 | [247c182993fcb65dda68](#r1-247c182993fcb65dda68) | wdra88 · 64/64/128 | fine | 0.114480004 | 7.0345 | 1.6967× | 0.118960001, 0.116240002, 0.114480004 | 3.91% |
| 3 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.116719998 | 6.8995 | 1.6642× | 0.117040001, 0.116719998 | 0.27% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x512x4096-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0084`。
- Rank 2：warmup=0，timed=3；原始结果 ID `192x512x4096-bf16-247c182993fcb65dda68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0084`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x512x4096-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0084`。
### 第二轮 FP16 · G303 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.152559996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.011200000 | 71.9024 | 13.6214× | 0.014240000, 0.011200000, 0.014320000 | 27.86% |
| 2 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.011760000 | 68.4784 | 12.9728× | 0.011760000, 0.011760000 | 0.00% |
| 3 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.113839999 | 7.0740 | 1.3401× | 0.116080001, 0.113839999 | 1.97% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `192x512x4096-fp16-G303-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0060`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x512x4096-fp16-G303-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x512x4096-fp16-G303-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0068`。
### 第一轮 BF16 · G408 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.195600003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [64ee2c9768bc4fbee283](#r1-64ee2c9768bc4fbee283) | wdra88 · 64/64/128 | fine | 0.110480003 | 7.2892 | 1.7705× | 0.111759998, 0.110480003 | 1.16% |
| 2 | [0eccce44229d587e140d](#r1-0eccce44229d587e140d) | wdra88 · 64/64/128 | fine | 0.114799999 | 7.0149 | 1.7038× | 0.118160002, 0.114799999, 0.115759999 | 2.93% |
| 3 | [b0fb3701e2e7d92ec37f](#r1-b0fb3701e2e7d92ec37f) | wdra88 · 64/64/128 | coarse | 0.115280002 | 6.9857 | 1.6967× | 0.115759999, 0.115280002 | 0.42% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x512x4096-fp32-64ee2c9768bc4fbee283`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0114`。
- Rank 2：warmup=0，timed=3；原始结果 ID `192x512x4096-fp32-0eccce44229d587e140d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0114`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x512x4096-fp32-b0fb3701e2e7d92ec37f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0071`。
### 第二轮 FP16 · G408 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.201440006 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.119039997 | 6.7650 | 1.6922× | 0.120559998, 0.119039997 | 1.28% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.164560005 | 4.8937 | 1.2241× | 0.166800007, 0.164560005 | 1.36% |
| 3 | [e2469f47dd001539f214](#r2-e2469f47dd001539f214) | wdra48 · 64/64/64 | coarse | 0.165840000 | 4.8559 | 1.2147× | 0.168080002, 0.165840000 | 1.35% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x512x4096-fp16-G408-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x512x4096-fp16-G408-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x512x4096-fp16-G408-e2469f47dd001539f214`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0040`。

<a id="mnk-192-1024-4096"></a>
## MNK 192 × 1024 × 4096

### 第一轮 BF16 · G409 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.202319995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [64ee2c9768bc4fbee283](#r1-64ee2c9768bc4fbee283) | wdra88 · 64/64/128 | fine | 0.113200001 | 14.2280 | 1.7873× | 0.113200001, 0.113440000 | 0.21% |
| 2 | [b0fb3701e2e7d92ec37f](#r1-b0fb3701e2e7d92ec37f) | wdra88 · 64/64/128 | coarse | 0.116400003 | 13.8369 | 1.7381× | 0.116559997, 0.116400003 | 0.14% |
| 3 | [0eccce44229d587e140d](#r1-0eccce44229d587e140d) | wdra88 · 64/64/128 | fine | 0.116959997 | 13.7706 | 1.7298× | 0.116959997, 0.117519997 | 0.48% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x1024x4096-fp32-64ee2c9768bc4fbee283`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x1024x4096-fp32-b0fb3701e2e7d92ec37f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0072`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x1024x4096-fp32-0eccce44229d587e140d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
### 第二轮 FP16 · G409 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.207519993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.119280003 | 13.5028 | 1.7398× | 0.119280003, 0.119840004 | 0.47% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.163599998 | 9.8448 | 1.2685× | 0.164240003, 0.163599998 | 0.39% |
| 3 | [e2469f47dd001539f214](#r2-e2469f47dd001539f214) | wdra48 · 64/64/64 | coarse | 0.164879993 | 9.7684 | 1.2586× | 0.165040001, 0.164879993 | 0.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x1024x4096-fp16-G409-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x1024x4096-fp16-G409-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x1024x4096-fp16-G409-e2469f47dd001539f214`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0040`。

<a id="mnk-192-2048-4096"></a>
## MNK 192 × 2048 × 4096

### 第一轮 BF16 · G410 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.213679999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b0fb3701e2e7d92ec37f](#r1-b0fb3701e2e7d92ec37f) | wdra88 · 64/64/128 | coarse | 0.116880000 | 27.5601 | 1.8282× | 0.119280003, 0.116880000, 0.117200002 | 2.05% |
| 2 | [64ee2c9768bc4fbee283](#r1-64ee2c9768bc4fbee283) | wdra88 · 64/64/128 | fine | 0.116880000 | 27.5601 | 1.8282× | 0.116880000, 0.117040001 | 0.14% |
| 3 | [0eccce44229d587e140d](#r1-0eccce44229d587e140d) | wdra88 · 64/64/128 | fine | 0.117839999 | 27.3356 | 1.8133× | 0.118639998, 0.117839999 | 0.68% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `192x2048x4096-fp32-b0fb3701e2e7d92ec37f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0072`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x2048x4096-fp32-64ee2c9768bc4fbee283`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x2048x4096-fp32-0eccce44229d587e140d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0114`。
### 第二轮 FP16 · G410 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.217280000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.124559999 | 25.8608 | 1.7444× | 0.124559999, 0.125200003 | 0.51% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.165759996 | 19.4331 | 1.3108× | 0.165759996, 0.168080002 | 1.40% |
| 3 | [e2469f47dd001539f214](#r2-e2469f47dd001539f214) | wdra48 · 64/64/64 | coarse | 0.166319996 | 19.3676 | 1.3064× | 0.171599999, 0.166319996, 0.169679999 | 3.17% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x2048x4096-fp16-G410-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x2048x4096-fp16-G410-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 3：warmup=0，timed=3；原始结果 ID `192x2048x4096-fp16-G410-e2469f47dd001539f214`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0040`。

<a id="mnk-192-4096-12288"></a>
## MNK 192 × 4096 × 12288

### 第一轮 BF16 · G304 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.635280013 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.201839998 | 95.7558 | 3.1474× | 0.204400003, 0.201839998 | 1.27% |
| 2 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.288879991 | 66.9044 | 2.1991× | 0.289519995, 0.288879991 | 0.22% |
| 3 | [7539c91fdab289ff899b](#r1-7539c91fdab289ff899b) | wdra48 · 64/64/128 | coarse | 0.398880005 | 48.4541 | 1.5927× | 0.401760012, 0.398880005 | 0.72% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x4096x12288-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x4096x12288-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0036`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x4096x12288-bf16-7539c91fdab289ff899b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0052`。
### 第二轮 FP16 · G304 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.235839993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.010000000 | 1932.7353 | 23.5840× | 0.014320000, 0.010320000, 0.010000000 | 43.20% |
| 2 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.197679996 | 97.7709 | 1.1930× | 0.200560004, 0.197679996 | 1.46% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.235839993 | 81.9511 | 1.0000× | 0.238639995, 0.235839993 | 1.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `192x4096x12288-fp16-G304-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0044`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x4096x12288-fp16-G304-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x4096x12288-fp16-G304-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0173`。

<a id="mnk-192-124160-4096"></a>
## MNK 192 × 124160 × 4096

### 第一轮 BF16 · G305 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.879360199 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 4.071599960 | 47.9632 | 1.1984× | 4.073999882, 4.071599960 | 0.06% |
| 2 | [2943f23873d8087254e4](#r1-2943f23873d8087254e4) | wdra88 · 128/128/64 | fine | 4.665760040 | 41.8553 | 1.0458× | 4.667759895, 4.665760040 | 0.04% |
| 3 | [3b67ff7fe683b39aa55a](#r1-3b67ff7fe683b39aa55a) | wdra88 · 128/128/64 | fine | 4.665999889 | 41.8532 | 1.0457× | 4.672080040, 4.665999889 | 0.13% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x124160x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x124160x4096-bf16-2943f23873d8087254e4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0129`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x124160x4096-bf16-3b67ff7fe683b39aa55a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0129`。
### 第二轮 FP16 · G305 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.362479925 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.362479925 | 58.0782 | 1.0000× | 3.362479925, 3.364079952 | 0.05% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.747839928 | 52.1065 | 0.8972× | 3.748480082, 3.747839928 | 0.02% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 4.049119949 | 48.2294 | 0.8304× | 4.049119949, 4.050960064 | 0.05% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x124160x4096-fp16-G305-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0117`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x124160x4096-fp16-G305-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0173`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x124160x4096-fp16-G305-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。

<a id="mnk-192-129280-4096"></a>
## MNK 192 × 129280 × 4096

### 第一轮 BF16 · G306 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.935520172 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 4.248079777 | 47.8663 | 1.1618× | 4.248079777, 4.250879765 | 0.07% |
| 2 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 4.869679928 | 41.7563 | 1.0135× | 4.876480103, 4.869679928 | 0.14% |
| 3 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 4.871679783 | 41.7392 | 1.0131× | 4.871679783, 4.872879982 | 0.02% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x129280x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x129280x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0060`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x129280x4096-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0130`。
### 第二轮 FP16 · G306 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.383120060 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.383120060 | 60.1042 | 1.0000× | 3.383120060, 3.411279917 | 0.83% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.920959949 | 51.8597 | 0.8628× | 3.921679974, 3.920959949 | 0.02% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 4.231040001 | 48.0591 | 0.7996× | 4.240880013, 4.231040001 | 0.23% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `192x129280x4096-fp16-G306-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0117`。
- Rank 2：warmup=0，timed=2；原始结果 ID `192x129280x4096-fp16-G306-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0173`。
- Rank 3：warmup=0，timed=2；原始结果 ID `192x129280x4096-fp16-G306-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。

<a id="mnk-240-129280-4096"></a>
## MNK 240 × 129280 × 4096

### 第一轮 BF16 · G307 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.934160233 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 4.255440235 | 59.7294 | 1.1595× | 4.255440235, 4.256559849 | 0.03% |
| 2 | [3b67ff7fe683b39aa55a](#r1-3b67ff7fe683b39aa55a) | wdra88 · 128/128/64 | fine | 4.875760078 | 52.1303 | 1.0120× | 4.875760078, 4.885200024 | 0.19% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 4.882959843 | 52.0534 | 1.0105× | 4.886479855, 4.882959843 | 0.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `240x129280x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `240x129280x4096-bf16-3b67ff7fe683b39aa55a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0094`。
- Rank 3：warmup=0，timed=2；原始结果 ID `240x129280x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0060`。
### 第二轮 FP16 · G307 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.393759966 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.393759966 | 74.8948 | 1.0000× | 3.393759966, 3.405200005 | 0.34% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.916160107 | 64.9041 | 0.8666× | 3.916160107, 3.922159910 | 0.15% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 4.234320164 | 60.0273 | 0.8015× | 4.237599850, 4.234320164 | 0.08% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `240x129280x4096-fp16-G307-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0117`。
- Rank 2：warmup=0，timed=2；原始结果 ID `240x129280x4096-fp16-G307-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0173`。
- Rank 3：warmup=0，timed=2；原始结果 ID `240x129280x4096-fp16-G307-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。

<a id="mnk-288-64-4096"></a>
## MNK 288 × 64 × 4096

### 第一轮 BF16 · G308 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.167439997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.103600003 | 1.4575 | 1.6162× | 0.106160000, 0.103600003, 0.104400001 | 2.47% |
| 2 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.112400003 | 1.3434 | 1.4897× | 0.120399997, 0.112400003, 0.115280002 | 7.12% |
| 3 | [b8657961b1124e881dd5](#r1-b8657961b1124e881dd5) | wdra88 · 64/64/128 | fine | 0.113200001 | 1.3339 | 1.4792× | 0.117040001, 0.113760002, 0.113200001 | 3.39% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `288x64x4096-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0083`。
- Rank 2：warmup=0，timed=3；原始结果 ID `288x64x4096-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0068`。
- Rank 3：warmup=0，timed=3；原始结果 ID `288x64x4096-bf16-b8657961b1124e881dd5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0083`。
### 第二轮 FP16 · G308 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.142480001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.108879998 | 1.3868 | 1.3086× | 0.130160004, 0.110799998, 0.108879998 | 19.54% |
| 2 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.111440003 | 1.3549 | 1.2785× | 0.120399997, 0.111440003, 0.114480004 | 8.04% |
| 3 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.141279995 | 1.0688 | 1.0085× | 0.169039994, 0.141279995, 0.144400001 | 19.65% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `288x64x4096-fp16-G308-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0053`。
- Rank 2：warmup=0，timed=3；原始结果 ID `288x64x4096-fp16-G308-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 3：warmup=0，timed=3；原始结果 ID `288x64x4096-fp16-G308-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0029`。

<a id="mnk-288-512-4096"></a>
## MNK 288 × 512 × 4096

### 第一轮 BF16 · G309 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.196559995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f1a574149008cfa85abd](#r1-f1a574149008cfa85abd) | wdra48 · 64/64/64 | fine | 0.008400000 | 143.8047 | 23.4000× | 0.010640000, 0.008560000, 0.008400000 | 26.67% |
| 2 | [46a1fbc6df36ffed1a49](#r1-46a1fbc6df36ffed1a49) | wdra48 · 64/64/64 | fine | 0.008560000 | 141.1168 | 22.9626× | 0.010640000, 0.008560000, 0.008560000 | 24.30% |
| 3 | [247c182993fcb65dda68](#r1-247c182993fcb65dda68) | wdra88 · 64/64/128 | fine | 0.011120000 | 108.6295 | 17.6763× | 0.012080000, 0.011440000, 0.011120000 | 8.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `288x512x4096-bf16-f1a574149008cfa85abd`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
- Rank 2：warmup=0，timed=3；原始结果 ID `288x512x4096-bf16-46a1fbc6df36ffed1a49`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
- Rank 3：warmup=0，timed=3；原始结果 ID `288x512x4096-bf16-247c182993fcb65dda68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0077`。
### 第二轮 FP16 · G309 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.154640004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.116880000 | 10.3350 | 1.3231× | 0.121200003, 0.116880000, 0.118960001 | 3.70% |
| 2 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.117519997 | 10.2788 | 1.3159× | 0.118000001, 0.117519997 | 0.41% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.154640004 | 7.8114 | 1.0000× | 0.154640004, 0.155599996 | 0.62% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `288x512x4096-fp16-G309-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0053`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x512x4096-fp16-G309-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x512x4096-fp16-G309-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0133`。
### 第一轮 BF16 · G411 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.202319995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [64ee2c9768bc4fbee283](#r1-64ee2c9768bc4fbee283) | wdra88 · 64/64/128 | fine | 0.111599997 | 10.8240 | 1.8129× | 0.114000000, 0.111599997, 0.111759998 | 2.15% |
| 2 | [880a05c17a0b9653b61c](#r1-880a05c17a0b9653b61c) | wdra88 · 64/64/128 | fine | 0.114639997 | 10.5370 | 1.7648× | 0.118160002, 0.114960000, 0.114639997 | 3.07% |
| 3 | [b0fb3701e2e7d92ec37f](#r1-b0fb3701e2e7d92ec37f) | wdra88 · 64/64/128 | coarse | 0.115120001 | 10.4930 | 1.7575× | 0.117040001, 0.115120001 | 1.67% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `288x512x4096-fp32-64ee2c9768bc4fbee283`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0083`。
- Rank 2：warmup=0，timed=3；原始结果 ID `288x512x4096-fp32-880a05c17a0b9653b61c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0083`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x512x4096-fp32-b0fb3701e2e7d92ec37f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0072`。
### 第二轮 FP16 · G411 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.206080005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.119280003 | 10.1271 | 1.7277× | 0.120559998, 0.119280003 | 1.07% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.165199995 | 7.3121 | 1.2475× | 0.167439997, 0.165199995 | 1.36% |
| 3 | [e2469f47dd001539f214](#r2-e2469f47dd001539f214) | wdra48 · 64/64/64 | coarse | 0.166960001 | 7.2350 | 1.2343× | 0.168720007, 0.166960001 | 1.05% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x512x4096-fp16-G411-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x512x4096-fp16-G411-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x512x4096-fp16-G411-e2469f47dd001539f214`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0040`。

<a id="mnk-288-1024-4096"></a>
## MNK 288 × 1024 × 4096

### 第一轮 BF16 · G412 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.208159998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [64ee2c9768bc4fbee283](#r1-64ee2c9768bc4fbee283) | wdra88 · 64/64/128 | fine | 0.113360003 | 21.3119 | 1.8363× | 0.115280002, 0.113360003 | 1.69% |
| 2 | [0eccce44229d587e140d](#r1-0eccce44229d587e140d) | wdra88 · 64/64/128 | fine | 0.115680002 | 20.8845 | 1.7994× | 0.118639998, 0.115680002, 0.117519997 | 2.56% |
| 3 | [b0fb3701e2e7d92ec37f](#r1-b0fb3701e2e7d92ec37f) | wdra88 · 64/64/128 | coarse | 0.115920000 | 20.8413 | 1.7957× | 0.118479997, 0.115920000, 0.116400003 | 2.21% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x1024x4096-fp32-64ee2c9768bc4fbee283`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0095`。
- Rank 2：warmup=0，timed=3；原始结果 ID `288x1024x4096-fp32-0eccce44229d587e140d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0095`。
- Rank 3：warmup=0，timed=3；原始结果 ID `288x1024x4096-fp32-b0fb3701e2e7d92ec37f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0072`。
### 第二轮 FP16 · G412 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.213599995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.122160003 | 19.7767 | 1.7485× | 0.122960001, 0.122160003 | 0.65% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.165680006 | 14.5818 | 1.2892× | 0.165840000, 0.165680006 | 0.10% |
| 3 | [e2469f47dd001539f214](#r2-e2469f47dd001539f214) | wdra48 · 64/64/64 | coarse | 0.166160002 | 14.5397 | 1.2855× | 0.166960001, 0.166160002 | 0.48% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x1024x4096-fp16-G412-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0056`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x1024x4096-fp16-G412-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x1024x4096-fp16-G412-e2469f47dd001539f214`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0040`。

<a id="mnk-288-2048-4096"></a>
## MNK 288 × 2048 × 4096

### 第一轮 BF16 · G413 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.216720000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [04f5799b1bd3f13ca9ca](#r1-04f5799b1bd3f13ca9ca) | wasp44 · 64/64/64 | fine | 0.168239996 | 28.7199 | 1.2882× | 0.171279997, 0.168239996 | 1.81% |
| 2 | [4d692db862aea00189bd](#r1-4d692db862aea00189bd) | wasp44 · 64/64/64 | fine | 0.169359997 | 28.5300 | 1.2796× | 0.171440005, 0.169359997 | 1.23% |
| 3 | [06df5002ca9f92ab59cd](#r1-06df5002ca9f92ab59cd) | wasp44 · 64/64/64 | coarse | 0.170320004 | 28.3692 | 1.2724× | 0.171440005, 0.170320004 | 0.66% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x2048x4096-fp32-04f5799b1bd3f13ca9ca`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0084`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x2048x4096-fp32-4d692db862aea00189bd`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0084`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x2048x4096-fp32-06df5002ca9f92ab59cd`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0032`。
### 第二轮 FP16 · G413 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.219679996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.168400005 | 28.6926 | 1.3045× | 0.168400005, 0.169839993 | 0.86% |
| 2 | [b51d8b6af4744a99ab8e](#r2-b51d8b6af4744a99ab8e) | wdra48 · 128/128/64 | coarse | 0.216800004 | 22.2871 | 1.0133× | 0.219439998, 0.216800004 | 1.22% |
| 3 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.219679996 | 21.9949 | 1.0000× | 0.221760005, 0.219679996 | 0.95% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x2048x4096-fp16-G413-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x2048x4096-fp16-G413-b51d8b6af4744a99ab8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0152`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x2048x4096-fp16-G413-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。

<a id="mnk-288-4096-12288"></a>
## MNK 288 × 4096 × 12288

### 第一轮 BF16 · G310 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.833280027 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.203119993 | 142.7286 | 4.1024× | 0.206479996, 0.203119993 | 1.65% |
| 2 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.559599996 | 51.8067 | 1.4891× | 0.563679993, 0.559599996 | 0.73% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.627919972 | 46.1699 | 1.3270× | 0.644159973, 0.627919972, 0.630479991 | 2.59% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x4096x12288-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x4096x12288-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0036`。
- Rank 3：warmup=0，timed=3；原始结果 ID `288x4096x12288-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0060`。
### 第二轮 FP16 · G310 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.238639995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.200880006 | 144.3201 | 1.1880× | 0.202319995, 0.200880006 | 0.72% |
| 2 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.202800006 | 142.9538 | 1.1767× | 0.206799999, 0.202800006 | 1.97% |
| 3 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.217680007 | 133.1819 | 1.0963× | 0.220880002, 0.217680007 | 1.47% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x4096x12288-fp16-G310-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x4096x12288-fp16-G310-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0045`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x4096x12288-fp16-G310-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0101`。

<a id="mnk-288-129280-4096"></a>
## MNK 288 × 129280 × 4096

### 第一轮 BF16 · G311 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 7.201920033 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 6.229839802 | 48.9595 | 1.1560× | 6.235280037, 6.229839802 | 0.09% |
| 2 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 7.082880020 | 43.0630 | 1.0168× | 7.082880020, 7.083280087 | 0.01% |
| 3 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 7.148880005 | 42.6654 | 1.0074× | 7.148880005, 7.149280071 | 0.01% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x129280x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x129280x4096-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0095`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x129280x4096-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0095`。
### 第二轮 FP16 · G311 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 4.952559948 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 4.952559948 | 61.5863 | 1.0000× | 4.952559948, 4.973919868 | 0.43% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 5.714079857 | 53.3786 | 0.8667× | 5.714079857, 5.717279911 | 0.06% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 6.188079834 | 49.2899 | 0.8003× | 6.192240238, 6.188079834 | 0.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `288x129280x4096-fp16-G311-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0117`。
- Rank 2：warmup=0，timed=2；原始结果 ID `288x129280x4096-fp16-G311-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0173`。
- Rank 3：warmup=0，timed=2；原始结果 ID `288x129280x4096-fp16-G311-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。

<a id="mnk-512-192-4096"></a>
## MNK 512 × 192 × 4096

### 第一轮 BF16 · G312 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.200399995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.109520003 | 7.3531 | 1.8298× | 0.112719998, 0.110480003, 0.109520003 | 2.92% |
| 2 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.115199998 | 6.9905 | 1.7396× | 0.118160002, 0.115199998, 0.116080001 | 2.57% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.115920000 | 6.9471 | 1.7288× | 0.117679998, 0.115920000 | 1.52% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `512x192x4096-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0106`。
- Rank 2：warmup=0，timed=3；原始结果 ID `512x192x4096-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0106`。
- Rank 3：warmup=0，timed=2；原始结果 ID `512x192x4096-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0068`。
### 第二轮 FP16 · G312 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.154080003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.115120001 | 6.9954 | 1.3384× | 0.116880000, 0.115120001 | 1.53% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.126000002 | 6.3913 | 1.2229× | 0.128240004, 0.126000002 | 1.78% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.154080003 | 5.2265 | 1.0000× | 0.157839999, 0.154080003, 0.155599996 | 2.44% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `512x192x4096-fp16-G312-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 2：warmup=0，timed=2；原始结果 ID `512x192x4096-fp16-G312-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0053`。
- Rank 3：warmup=0，timed=3；原始结果 ID `512x192x4096-fp16-G312-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0133`。

<a id="mnk-2048-64-2048"></a>
## MNK 2048 × 64 × 2048

### 第一轮 BF16 · G313 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.098640002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.063440003 | 8.4627 | 1.5549× | 0.063600004, 0.063440003 | 0.25% |
| 2 | [b8657961b1124e881dd5](#r1-b8657961b1124e881dd5) | wdra88 · 64/64/128 | fine | 0.066000000 | 8.1344 | 1.4945× | 0.068080001, 0.066639997, 0.066000000 | 3.15% |
| 3 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.066160001 | 8.1147 | 1.4909× | 0.067280002, 0.066160001 | 1.69% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x64x2048-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0130`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2048x64x2048-bf16-b8657961b1124e881dd5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0130`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x64x2048-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0130`。
### 第二轮 FP16 · G313 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.080720000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.066079997 | 8.1246 | 1.2215× | 0.072880000, 0.066079997, 0.069360003 | 10.29% |
| 2 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.066960000 | 8.0178 | 1.2055× | 0.067599997, 0.066960000 | 0.96% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.080720000 | 6.6510 | 1.0000× | 0.083120003, 0.080720000, 0.081840001 | 2.97% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `2048x64x2048-fp16-G313-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0053`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x64x2048-fp16-G313-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2048x64x2048-fp16-G313-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0133`。

<a id="mnk-2048-256-2048"></a>
## MNK 2048 × 256 × 2048

### 第一轮 BF16 · G314 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.113040000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.089680001 | 23.9461 | 1.2605× | 0.090319999, 0.089680001 | 0.71% |
| 2 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.091600001 | 23.4441 | 1.2341× | 0.091600001, 0.094159998, 0.093680002 | 2.79% |
| 3 | [3717d3df609be657fa32](#r1-3717d3df609be657fa32) | wasp44 · 64/64/64 | fine | 0.093599997 | 22.9432 | 1.2077× | 0.093599997, 0.094640002 | 1.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x256x2048-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0107`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2048x256x2048-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0028`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x256x2048-bf16-3717d3df609be657fa32`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0106`。
### 第二轮 FP16 · G314 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.099280000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.092399999 | 23.2412 | 1.0745× | 0.092399999, 0.092720002 | 0.35% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.099280000 | 21.6306 | 1.0000× | 0.100240000, 0.099280000 | 0.97% |
| 3 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.114079997 | 18.8244 | 0.8703× | 0.114079997, 0.119360000, 0.118400000 | 4.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x256x2048-fp16-G314-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0029`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x256x2048-fp16-G314-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0133`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2048x256x2048-fp16-G314-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0011`。

<a id="mnk-2048-1024-2048"></a>
## MNK 2048 × 1024 × 2048

### 第一轮 BF16 · G315 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.191760004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 0.188240007 | 45.6329 | 1.0187× | 0.188240007, 0.190960005 | 1.44% |
| 2 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.190640002 | 45.0584 | 1.0059× | 0.192240000, 0.190640002 | 0.84% |
| 3 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.191760004 | 44.7952 | 1.0000× | 0.192719996, 0.191760004 | 0.50% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x1024x2048-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0107`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x1024x2048-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0107`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x1024x2048-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0012`。
### 第二轮 FP16 · G315 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.190880001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.190880001 | 45.0018 | 1.0000× | 0.193279997, 0.190880001 | 1.26% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.193279997 | 44.4430 | 0.9876× | 0.193279997, 0.193599999 | 0.17% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.200639993 | 42.8127 | 0.9514× | 0.200639993, 0.204879999, 0.204559997 | 2.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x1024x2048-fp16-G315-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0011`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x1024x2048-fp16-G315-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0003`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2048x1024x2048-fp16-G315-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0077`。

<a id="mnk-2048-2048-512"></a>
## MNK 2048 × 2048 × 512

### 第一轮 BF16 · G316 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.099919997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.099919997 | 42.9841 | 1.0000× | 0.101199999, 0.099919997 | 1.28% |
| 2 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.099919997 | 42.9841 | 1.0000× | 0.100720003, 0.099919997 | 0.80% |
| 3 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.100560002 | 42.7105 | 0.9936× | 0.100879997, 0.100560002 | 0.32% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x2048x512-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0012`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x2048x512-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0107`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x2048x512-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0107`。
### 第二轮 FP16 · G316 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.100479998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.100479998 | 42.7445 | 1.0000× | 0.102240004, 0.100479998 | 1.75% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.106080003 | 40.4880 | 0.9472× | 0.106080003, 0.110239998, 0.108319998 | 3.92% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.109200001 | 39.3312 | 0.9201× | 0.113360003, 0.109200001, 0.109999999 | 3.81% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x2048x512-fp16-G316-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0011`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2048x2048x512-fp16-G316-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0003`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2048x2048x512-fp16-G316-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0077`。

<a id="mnk-2048-2048-4096"></a>
## MNK 2048 × 2048 × 4096

### 第一轮 BF16 · G317 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.666559994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [a2ec7ee991efd60bd336](#r1-a2ec7ee991efd60bd336) | wasp44 · 128/128/64 | coarse | 0.587040007 | 58.5305 | 1.1355× | 0.592639983, 0.587040007 | 0.95% |
| 2 | [05df19f7638b5be06b68](#r1-05df19f7638b5be06b68) | wdra88 · 128/128/64 | fine | 0.657599986 | 52.2502 | 1.0136× | 0.663439989, 0.657599986 | 0.89% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 0.657999992 | 52.2184 | 1.0130× | 0.657999992, 0.661840022 | 0.58% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x2048x4096-bf16-a2ec7ee991efd60bd336`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0020`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x2048x4096-bf16-05df19f7638b5be06b68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0108`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x2048x4096-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0061`。
### 第二轮 FP16 · G317 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.582480013 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.346320003 | 99.2138 | 1.6819× | 0.348879993, 0.346320003 | 0.74% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.582480013 | 58.9887 | 1.0000× | 0.582480013, 0.583760023 | 0.22% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.583679974 | 58.8674 | 0.9979× | 0.590960026, 0.583679974 | 1.25% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x2048x4096-fp16-G317-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0189`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x2048x4096-fp16-G317-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0173`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x2048x4096-fp16-G317-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0141`。

<a id="mnk-2048-9216-2048"></a>
## MNK 2048 × 9216 × 2048

### 第一轮 BF16 · G318 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.320639968 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 0.981760025 | 78.7457 | 1.3452× | 0.984160006, 0.981760025 | 0.24% |
| 2 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 1.320320010 | 58.5535 | 1.0002× | 1.320320010, 1.326959968 | 0.50% |
| 3 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 1.320639968 | 58.5394 | 1.0000× | 1.320639968, 1.335999966 | 1.16% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x9216x2048-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x9216x2048-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0130`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x9216x2048-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0004`。
### 第二轮 FP16 · G318 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.247120023 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 1.247120023 | 61.9904 | 1.0000× | 1.248800039, 1.247120023 | 0.13% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.323840022 | 58.3979 | 0.9420× | 1.323840022, 1.332960010 | 0.69% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 1.382079959 | 55.9370 | 0.9024× | 1.382079959, 1.384400010 | 0.17% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x9216x2048-fp16-G318-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0077`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x9216x2048-fp16-G318-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0004`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x9216x2048-fp16-G318-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0189`。

<a id="mnk-2048-12288-2048"></a>
## MNK 2048 × 12288 × 2048

### 第一轮 BF16 · G319 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.675119996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 1.288959980 | 79.9708 | 1.2996× | 1.290799975, 1.288959980 | 0.14% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 1.584159970 | 65.0687 | 1.0574× | 1.584159970, 1.584959984 | 0.05% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 1.652240038 | 62.3876 | 1.0138× | 1.652240038, 1.733520031, 1.682559967 | 4.92% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x12288x2048-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2048x12288x2048-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2048x12288x2048-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
### 第二轮 FP16 · G319 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.625759959 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 1.625759959 | 63.4037 | 1.0000× | 1.626880050, 1.625759959 | 0.07% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.696640015 | 60.7549 | 0.9582× | 1.753280044, 1.703199983, 1.696640015 | 3.34% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 1.839599967 | 56.0335 | 0.8838× | 1.844560027, 1.839599967 | 0.27% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2048x12288x2048-fp16-G319-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0077`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2048x12288x2048-fp16-G319-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0004`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2048x12288x2048-fp16-G319-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0189`。

<a id="mnk-2560-32-6144"></a>
## MNK 2560 × 32 × 6144

### 第一轮 BF16 · G320 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.306400001 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.161840007 | 6.2199 | 1.8932× | 0.162799999, 0.161840007 | 0.59% |
| 2 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.166960001 | 6.0292 | 1.8352× | 0.167520002, 0.166960001 | 0.34% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.167040005 | 6.0263 | 1.8343× | 0.167280003, 0.172240004, 0.167040005 | 3.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x32x6144-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0085`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x32x6144-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0085`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2560x32x6144-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0069`。
### 第二轮 FP16 · G320 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.233999997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.117760003 | 8.5482 | 1.9871× | 0.121679999, 0.117760003, 0.120399997 | 3.33% |
| 2 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.168160006 | 5.9862 | 1.3915× | 0.175119996, 0.169520006, 0.168160006 | 4.14% |
| 3 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.179839998 | 5.5974 | 1.3012× | 0.179839998, 0.181360006 | 0.85% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `2560x32x6144-fp16-G320-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0189`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2560x32x6144-fp16-G320-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x32x6144-fp16-G320-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0053`。

<a id="mnk-2560-64-2048"></a>
## MNK 2560 × 64 × 2048

### 第一轮 BF16 · G321 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.103440002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [17b66ad620da24d6f82b](#r1-17b66ad620da24d6f82b) | wdra88 · 64/64/128 | fine | 0.064080000 | 10.4727 | 1.6142× | 0.065200001, 0.064080000 | 1.75% |
| 2 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.067280002 | 9.9746 | 1.5375× | 0.068080001, 0.067280002 | 1.19% |
| 3 | [247c182993fcb65dda68](#r1-247c182993fcb65dda68) | wdra88 · 64/64/128 | fine | 0.067919999 | 9.8806 | 1.5230× | 0.068240002, 0.067919999 | 0.47% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x64x2048-bf16-17b66ad620da24d6f82b`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x64x2048-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x64x2048-bf16-247c182993fcb65dda68`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0131`。
### 第二轮 FP16 · G321 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.081680000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.066480003 | 10.0946 | 1.2286× | 0.069360003, 0.066480003, 0.069519997 | 4.57% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.068000004 | 9.8690 | 1.2012× | 0.074960001, 0.071280003, 0.068000004 | 10.24% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.081680000 | 8.2161 | 1.0000× | 0.084399998, 0.081840001, 0.081680000 | 3.33% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `2560x64x2048-fp16-G321-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2560x64x2048-fp16-G321-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0053`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2560x64x2048-fp16-G321-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0133`。

<a id="mnk-2560-64-7168"></a>
## MNK 2560 × 64 × 7168

### 第一轮 BF16 · G322 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.364239991 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.087600000 | 26.8129 | 4.1580× | 0.087600000, 0.087920003 | 0.37% |
| 2 | [b8657961b1124e881dd5](#r1-b8657961b1124e881dd5) | wdra88 · 64/64/128 | fine | 0.199440002 | 11.7770 | 1.8263× | 0.199440002, 0.201519996 | 1.04% |
| 3 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.199680001 | 11.7629 | 1.8241× | 0.199680001, 0.200399995 | 0.36% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x64x7168-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x64x7168-bf16-b8657961b1124e881dd5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x64x7168-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0078`。
### 第二轮 FP16 · G322 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.281199992 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.199279994 | 11.7865 | 1.4111× | 0.200880006, 0.199279994 | 0.80% |
| 2 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.201839998 | 11.6370 | 1.3932× | 0.203439996, 0.201839998 | 0.79% |
| 3 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.218640000 | 10.7428 | 1.2861× | 0.221839994, 0.218640000 | 1.46% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x64x7168-fp16-G322-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0069`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x64x7168-fp16-G322-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0045`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x64x7168-fp16-G322-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0101`。

<a id="mnk-2560-256-2048"></a>
## MNK 2560 × 256 × 2048

### 第一轮 BF16 · G323 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.121200003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.089040004 | 30.1477 | 1.3612× | 0.090800002, 0.089040004 | 1.98% |
| 2 | [eee8f4403014e9f7aacb](#r1-eee8f4403014e9f7aacb) | wasp44 · 64/64/64 | fine | 0.093680002 | 28.6545 | 1.2938× | 0.095440000, 0.093680002 | 1.88% |
| 3 | [3717d3df609be657fa32](#r1-3717d3df609be657fa32) | wasp44 · 64/64/64 | fine | 0.094159998 | 28.5084 | 1.2872× | 0.095279999, 0.094159998 | 1.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x256x2048-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0108`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x256x2048-bf16-eee8f4403014e9f7aacb`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0108`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x256x2048-bf16-3717d3df609be657fa32`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0108`。
### 第二轮 FP16 · G323 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.098959997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.093359999 | 28.7527 | 1.0600× | 0.095600002, 0.093359999, 0.093840003 | 2.40% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.098959997 | 27.1257 | 1.0000× | 0.102959998, 0.098959997, 0.098959997 | 4.04% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.117279999 | 22.8884 | 0.8438× | 0.122960001, 0.117279999, 0.119920000 | 4.84% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `2560x256x2048-fp16-G323-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0029`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2560x256x2048-fp16-G323-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2560x256x2048-fp16-G323-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0142`。

<a id="mnk-2560-256-6144"></a>
## MNK 2560 × 256 × 6144

### 第一轮 BF16 · G324 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.345440000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.166160002 | 48.4657 | 2.0790× | 0.166160002, 0.166160002 | 0.00% |
| 2 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.241840005 | 33.2991 | 1.4284× | 0.242799997, 0.241840005 | 0.40% |
| 3 | [51908aa291df6bf1d852](#r1-51908aa291df6bf1d852) | wasp44 · 64/64/64 | fine | 0.243919998 | 33.0152 | 1.4162× | 0.243919998, 0.244240001 | 0.13% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x256x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x256x6144-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0108`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x256x6144-bf16-51908aa291df6bf1d852`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0109`。
### 第二轮 FP16 · G324 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.275920004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.202160001 | 39.8351 | 1.3649× | 0.204720005, 0.202160001 | 1.27% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.255679995 | 31.4967 | 1.0792× | 0.258560002, 0.255679995 | 1.13% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.275920004 | 29.1862 | 1.0000× | 0.277520001, 0.275920004 | 0.58% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x256x6144-fp16-G324-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0045`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x256x6144-fp16-G324-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0029`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x256x6144-fp16-G324-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。

<a id="mnk-2560-512-7168"></a>
## MNK 2560 × 512 × 7168

### 第一轮 BF16 · G414 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.467599988 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [8f86441f7995d544ff0e](#r1-8f86441f7995d544ff0e) | wdra88 · 128/128/64 | fine | 0.377200007 | 49.8157 | 1.2397× | 0.378639996, 0.377200007 | 0.38% |
| 2 | [7559008785dcec902504](#r1-7559008785dcec902504) | wdra88 · 128/128/64 | coarse | 0.381760001 | 49.2207 | 1.2249× | 0.384240001, 0.381760001 | 0.65% |
| 3 | [dbddcf203270db2ec3f7](#r1-dbddcf203270db2ec3f7) | wdra88 · 128/128/64 | fine | 0.381839991 | 49.2104 | 1.2246× | 0.385600001, 0.381839991 | 0.98% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x512x7168-fp32-8f86441f7995d544ff0e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0079`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x512x7168-fp32-7559008785dcec902504`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0064`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x512x7168-fp32-dbddcf203270db2ec3f7`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0079`。
### 第二轮 FP16 · G414 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.467680007 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b51d8b6af4744a99ab8e](#r2-b51d8b6af4744a99ab8e) | wdra48 · 128/128/64 | coarse | 0.218239993 | 86.1001 | 2.1430× | 0.218960002, 0.218239993 | 0.33% |
| 2 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.225040004 | 83.4984 | 2.0782× | 0.226480007, 0.225040004 | 0.64% |
| 3 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.324160010 | 57.9667 | 1.4427× | 0.324400008, 0.324160010 | 0.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x512x7168-fp16-G414-b51d8b6af4744a99ab8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0152`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x512x7168-fp16-G414-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0192`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x512x7168-fp16-G414-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0184`。

<a id="mnk-2560-1024-7168"></a>
## MNK 2560 × 1024 × 7168

### 第一轮 BF16 · G415 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.658240020 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 0.543280005 | 69.1742 | 1.2116× | 0.551280022, 0.543280005 | 1.47% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 0.642400026 | 58.5009 | 1.0247× | 0.644959986, 0.642400026 | 0.40% |
| 3 | [249d0e074b4086e8aa18](#r1-249d0e074b4086e8aa18) | nowasp · 128/128/64 | fine | 0.656000018 | 57.2881 | 1.0034× | 0.660080016, 0.656000018 | 0.62% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x1024x7168-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0080`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x1024x7168-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0080`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x1024x7168-fp32-249d0e074b4086e8aa18`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0080`。
### 第二轮 FP16 · G415 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.631839991 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b51d8b6af4744a99ab8e](#r2-b51d8b6af4744a99ab8e) | wdra48 · 128/128/64 | coarse | 0.418559998 | 89.7863 | 1.5096× | 0.430319995, 0.421840012, 0.418559998 | 2.81% |
| 2 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 0.436720014 | 86.0528 | 1.4468× | 0.439280003, 0.436720014 | 0.59% |
| 3 | [a184436d067a2e030558](#r2-a184436d067a2e030558) | wasp44 · 128/128/64 | coarse | 0.621599972 | 60.4584 | 1.0165× | 0.634800017, 0.621599972, 0.625840008 | 2.12% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `2560x1024x7168-fp16-G415-b51d8b6af4744a99ab8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0152`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x1024x7168-fp16-G415-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0192`。
- Rank 3：warmup=0，timed=3；原始结果 ID `2560x1024x7168-fp16-G415-a184436d067a2e030558`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0184`。

<a id="mnk-2560-2048-7168"></a>
## MNK 2560 × 2048 × 7168

### 第一轮 BF16 · G416 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.275760055 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 1.057039976 | 71.1060 | 1.2069× | 1.073359966, 1.057039976 | 1.54% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 1.240399957 | 60.5949 | 1.0285× | 1.240399957, 1.248080015 | 0.62% |
| 3 | [5e627b894b84cb1bc438](#r1-5e627b894b84cb1bc438) | nowasp · 128/128/64 | coarse | 1.275760055 | 58.9154 | 1.0000× | 1.281919956, 1.275760055 | 0.48% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x2048x7168-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0080`。
- Rank 2：warmup=0，timed=2；原始结果 ID `2560x2048x7168-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0080`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x2048x7168-fp32-5e627b894b84cb1bc438`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0008`。
### 第二轮 FP16 · G416 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.162879944 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b51d8b6af4744a99ab8e](#r2-b51d8b6af4744a99ab8e) | wdra48 · 128/128/64 | coarse | 0.809520006 | 92.8475 | 1.4365× | 0.809920013, 0.809520006 | 0.05% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.162879944 | 64.6343 | 1.0000× | 1.221840024, 1.162879944, 1.166479945 | 5.07% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.270400047 | 59.1640 | 0.9154× | 1.271199942, 1.270400047 | 0.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `2560x2048x7168-fp16-G416-b51d8b6af4744a99ab8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0152`。
- Rank 2：warmup=0，timed=3；原始结果 ID `2560x2048x7168-fp16-G416-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0120`。
- Rank 3：warmup=0，timed=2；原始结果 ID `2560x2048x7168-fp16-G416-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。

<a id="mnk-3584-64-4096"></a>
## MNK 3584 × 64 × 4096

### 第一轮 BF16 · G325 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.218319997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.084880002 | 22.1377 | 2.5721× | 0.088720001, 0.086960003, 0.084880002 | 4.52% |
| 2 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.119120002 | 15.7744 | 1.8328× | 0.120080002, 0.119120002 | 0.81% |
| 3 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.119280003 | 15.7533 | 1.8303× | 0.120399997, 0.119280003 | 0.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `3584x64x4096-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x64x4096-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x64x4096-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0069`。
### 第二轮 FP16 · G325 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.159759998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.118639998 | 15.8382 | 1.3466× | 0.119039997, 0.118639998 | 0.34% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.129199997 | 14.5437 | 1.2365× | 0.131119996, 0.129199997 | 1.49% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.159759998 | 11.7617 | 1.0000× | 0.161200002, 0.159759998 | 0.90% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x64x4096-fp16-G325-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0070`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x64x4096-fp16-G325-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0054`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x64x4096-fp16-G325-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。

<a id="mnk-3584-128-4096"></a>
## MNK 3584 × 128 × 4096

### 第一轮 BF16 · G326 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.251760006 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.156399995 | 24.0288 | 1.6097× | 0.162799999, 0.156399995, 0.156880006 | 4.09% |
| 2 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.166480005 | 22.5739 | 1.5123× | 0.166480005, 0.169679999 | 1.92% |
| 3 | [51908aa291df6bf1d852](#r1-51908aa291df6bf1d852) | wasp44 · 64/64/64 | fine | 0.168880001 | 22.2531 | 1.4908× | 0.168880001, 0.170080006 | 0.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `3584x128x4096-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x128x4096-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0132`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x128x4096-bf16-51908aa291df6bf1d852`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0132`。
### 第二轮 FP16 · G326 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.188480005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.177520007 | 21.1700 | 1.0617× | 0.177760005, 0.177520007 | 0.14% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.188480005 | 19.9390 | 1.0000× | 0.191280007, 0.188480005 | 1.49% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.224879995 | 16.7116 | 0.8381× | 0.225199997, 0.224879995 | 0.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x128x4096-fp16-G326-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x128x4096-fp16-G326-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x128x4096-fp16-G326-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0062`。

<a id="mnk-3584-512-4096"></a>
## MNK 3584 × 512 × 4096

### 第一轮 BF16 · G327 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.318960011 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.316560000 | 47.4867 | 1.0076× | 0.316720009, 0.316560000 | 0.05% |
| 2 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.317999989 | 47.2717 | 1.0030× | 0.320560008, 0.317999989 | 0.81% |
| 3 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.318960011 | 47.1294 | 1.0000× | 0.318960011, 0.318960011 | 0.00% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x512x4096-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0085`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x512x4096-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0085`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x512x4096-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0013`。
### 第二轮 FP16 · G327 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.318720013 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.318720013 | 47.1649 | 1.0000× | 0.319359988, 0.318720013 | 0.20% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.363759995 | 41.3250 | 0.8762× | 0.364719987, 0.363759995 | 0.26% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.381040007 | 39.4509 | 0.8364× | 0.381040007, 0.381920010 | 0.23% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp16-G327-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0012`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp16-G327-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp16-G327-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。
### 第一轮 BF16 · G417 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.319599986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9fbc98f298852c6db1bc](#r1-9fbc98f298852c6db1bc) | nowasp · 64/64/64 | fine | 0.317840010 | 47.2954 | 1.0055× | 0.317840010, 0.318480015 | 0.20% |
| 2 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 0.319599986 | 47.0350 | 1.0000× | 0.319759995, 0.319599986 | 0.05% |
| 3 | [39eb19a1c9076aa7945a](#r1-39eb19a1c9076aa7945a) | nowasp · 64/64/64 | fine | 0.320719987 | 46.8707 | 0.9965× | 0.321040004, 0.320719987 | 0.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp32-9fbc98f298852c6db1bc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp32-39eb19a1c9076aa7945a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0080`。
### 第二轮 FP16 · G417 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.318560004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.318560004 | 47.1886 | 1.0000× | 0.318560004, 0.320959985 | 0.75% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.364719987 | 41.2162 | 0.8734× | 0.365359992, 0.364719987 | 0.18% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.388559997 | 38.6874 | 0.8198× | 0.390799999, 0.388559997 | 0.58% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp16-G417-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp16-G417-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x512x4096-fp16-G417-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0137`。

<a id="mnk-3584-1024-4096"></a>
## MNK 3584 × 1024 × 4096

### 第一轮 BF16 · G418 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.604399979 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9fbc98f298852c6db1bc](#r1-9fbc98f298852c6db1bc) | nowasp · 64/64/64 | fine | 0.602240026 | 49.9216 | 1.0036× | 0.602240026, 0.602959991 | 0.12% |
| 2 | [39eb19a1c9076aa7945a](#r1-39eb19a1c9076aa7945a) | nowasp · 64/64/64 | fine | 0.604080021 | 49.7695 | 1.0005× | 0.604240000, 0.604080021 | 0.03% |
| 3 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 0.604399979 | 49.7432 | 1.0000× | 0.607439995, 0.604399979 | 0.50% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x1024x4096-fp32-9fbc98f298852c6db1bc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0092`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x1024x4096-fp32-39eb19a1c9076aa7945a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0092`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x1024x4096-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
### 第二轮 FP16 · G418 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.601920009 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.601920009 | 49.9481 | 1.0000× | 0.601920009, 0.603200018 | 0.21% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.621919990 | 48.3419 | 0.9678× | 0.623120010, 0.621919990 | 0.19% |
| 3 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.652639985 | 46.0664 | 0.9223× | 0.656239986, 0.652639985 | 0.55% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x1024x4096-fp16-G418-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x1024x4096-fp16-G418-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 3：warmup=0，timed=2；原始结果 ID `3584x1024x4096-fp16-G418-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0105`。

<a id="mnk-3584-2048-4096"></a>
## MNK 3584 × 2048 × 4096

### 第一轮 BF16 · G419 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.078799963 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 0.916159987 | 65.6321 | 1.1775× | 0.920239985, 0.916159987 | 0.45% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 1.054640055 | 57.0143 | 1.0229× | 1.060240030, 1.054640055 | 0.53% |
| 3 | [249d0e074b4086e8aa18](#r1-249d0e074b4086e8aa18) | nowasp · 128/128/64 | fine | 1.076079965 | 55.8783 | 1.0025× | 1.140640020, 1.076079965, 1.098080039 | 6.00% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x2048x4096-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x2048x4096-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
- Rank 3：warmup=0，timed=3；原始结果 ID `3584x2048x4096-fp32-249d0e074b4086e8aa18`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
### 第二轮 FP16 · G419 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.960479975 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.960479975 | 62.6036 | 1.0000× | 0.960479975, 0.962639987 | 0.22% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.055520058 | 56.9667 | 0.9100× | 1.055520058, 1.066640019 | 1.05% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.064000010 | 56.5127 | 0.9027× | 1.097120047, 1.064000010, 1.066239953 | 3.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `3584x2048x4096-fp16-G419-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `3584x2048x4096-fp16-G419-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 3：warmup=0，timed=3；原始结果 ID `3584x2048x4096-fp16-G419-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。

<a id="mnk-4608-32-6144"></a>
## MNK 4608 × 32 × 6144

### 第一轮 BF16 · G328 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.336400002 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.087760001 | 20.6465 | 3.8332× | 0.088560000, 0.087760001 | 0.91% |
| 2 | [b8657961b1124e881dd5](#r1-b8657961b1124e881dd5) | wdra88 · 64/64/128 | fine | 0.166960001 | 10.8525 | 2.0149× | 0.168239996, 0.166960001 | 0.77% |
| 3 | [623dfda53d317222ceaf](#r1-623dfda53d317222ceaf) | wdra88 · 64/64/128 | fine | 0.169200003 | 10.7089 | 1.9882× | 0.169839993, 0.169200003 | 0.38% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x32x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x32x6144-bf16-b8657961b1124e881dd5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0086`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x32x6144-bf16-623dfda53d317222ceaf`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0086`。
### 第二轮 FP16 · G328 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.233919993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.168880001 | 10.7292 | 1.3851× | 0.170640007, 0.175280005, 0.168880001 | 3.79% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.183760002 | 9.8604 | 1.2730× | 0.187360004, 0.183760002 | 1.96% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.233919993 | 7.7460 | 1.0000× | 0.233919993, 0.236880004 | 1.27% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4608x32x6144-fp16-G328-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0070`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x32x6144-fp16-G328-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0054`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x32x6144-fp16-G328-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。

<a id="mnk-4608-64-7168"></a>
## MNK 4608 × 64 × 7168

### 第一轮 BF16 · G329 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.394959986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.086159997 | 49.0699 | 4.5840× | 0.089840002, 0.086159997, 0.087600000 | 4.27% |
| 2 | [a1f7c7d7b8b31636bb20](#r1-a1f7c7d7b8b31636bb20) | wdra88 · 64/64/128 | coarse | 0.198799998 | 21.2669 | 1.9867× | 0.203119993, 0.198799998, 0.198960006 | 2.17% |
| 3 | [b8657961b1124e881dd5](#r1-b8657961b1124e881dd5) | wdra88 · 64/64/128 | fine | 0.198960006 | 21.2498 | 1.9851× | 0.201040000, 0.198960006 | 1.05% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `4608x64x7168-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4608x64x7168-bf16-a1f7c7d7b8b31636bb20`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0069`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x64x7168-bf16-b8657961b1124e881dd5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0109`。
### 第二轮 FP16 · G329 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.283760011 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.199760005 | 21.1647 | 1.4205× | 0.199760005, 0.200719997 | 0.48% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.226480007 | 18.6677 | 1.2529× | 0.234320000, 0.228080004, 0.226480007 | 3.46% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.283760011 | 14.8994 | 1.0000× | 0.284720004, 0.283760011 | 0.34% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x64x7168-fp16-G329-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0070`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4608x64x7168-fp16-G329-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0054`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x64x7168-fp16-G329-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。

<a id="mnk-4608-256-6144"></a>
## MNK 4608 × 256 × 6144

### 第一轮 BF16 · G330 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.349839985 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.246800005 | 58.7339 | 1.4175× | 0.246800005, 0.250640005 | 1.56% |
| 2 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.287120014 | 50.4859 | 1.2184× | 0.287120014, 0.289359987 | 0.78% |
| 3 | [51908aa291df6bf1d852](#r1-51908aa291df6bf1d852) | wasp44 · 64/64/64 | fine | 0.290160000 | 49.9570 | 1.2057× | 0.291280001, 0.290160000 | 0.39% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x256x6144-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x256x6144-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0109`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x256x6144-bf16-51908aa291df6bf1d852`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0110`。
### 第二轮 FP16 · G330 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.338719994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.297520012 | 48.7211 | 1.1385× | 0.299760014, 0.297520012 | 0.75% |
| 2 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 0.328640014 | 44.1076 | 1.0307× | 0.342480004, 0.334399998, 0.328640014 | 4.21% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.329439998 | 44.0005 | 1.0282× | 0.342640013, 0.329439998, 0.331440002 | 4.01% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x256x6144-fp16-G330-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=3；原始结果 ID `4608x256x6144-fp16-G330-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0102`。
- Rank 3：warmup=0，timed=3；原始结果 ID `4608x256x6144-fp16-G330-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0062`。

<a id="mnk-4608-512-7168"></a>
## MNK 4608 × 512 × 7168

### 第一轮 BF16 · G420 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.640399992 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [39eb19a1c9076aa7945a](#r1-39eb19a1c9076aa7945a) | nowasp · 64/64/64 | fine | 0.639440000 | 52.8945 | 1.0015× | 0.639440000, 0.639599979 | 0.03% |
| 2 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 0.640399992 | 52.8152 | 1.0000× | 0.640879989, 0.640399992 | 0.07% |
| 3 | [9fbc98f298852c6db1bc](#r1-9fbc98f298852c6db1bc) | nowasp · 64/64/64 | fine | 0.641439974 | 52.7296 | 0.9984× | 0.642799973, 0.641439974 | 0.21% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x512x7168-fp32-39eb19a1c9076aa7945a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0115`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x512x7168-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x512x7168-fp32-9fbc98f298852c6db1bc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0115`。
### 第二轮 FP16 · G420 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.640479982 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.640479982 | 52.8086 | 1.0000× | 0.640479982, 0.640799999 | 0.05% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.670080006 | 50.4759 | 0.9558× | 0.678240001, 0.670080006 | 1.22% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.671360016 | 50.3796 | 0.9540× | 0.672160029, 0.671360016 | 0.12% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x512x7168-fp16-G420-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x512x7168-fp16-G420-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x512x7168-fp16-G420-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。

<a id="mnk-4608-1024-7168"></a>
## MNK 4608 × 1024 × 7168

### 第一轮 BF16 · G421 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.255599976 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [498436a13632e968d1a9](#r1-498436a13632e968d1a9) | wasp44 · 128/128/64 | fine | 1.126880050 | 60.0292 | 1.1142× | 1.126880050, 1.132320046 | 0.48% |
| 2 | [32d663c1fff0030fd159](#r1-32d663c1fff0030fd159) | wdra88 · 128/128/64 | fine | 1.134960055 | 59.6019 | 1.1063× | 1.134960055, 1.144880056 | 0.87% |
| 3 | [dbddcf203270db2ec3f7](#r1-dbddcf203270db2ec3f7) | wdra88 · 128/128/64 | fine | 1.142560005 | 59.2054 | 1.0989× | 1.152799964, 1.142560005 | 0.90% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x1024x7168-fp32-498436a13632e968d1a9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0115`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x1024x7168-fp32-32d663c1fff0030fd159`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0115`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x1024x7168-fp32-dbddcf203270db2ec3f7`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0115`。
### 第二轮 FP16 · G421 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.112239957 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.112239957 | 60.8194 | 1.0000× | 1.112239957, 1.115759969 | 0.32% |
| 2 | [3d4c2441dd5696bafd6f](#r2-3d4c2441dd5696bafd6f) | wdra88 · 128/128/64 | coarse | 1.122480035 | 60.2645 | 0.9909× | 1.131279945, 1.122480035 | 0.78% |
| 3 | [fc1f34eeb0619026b2a6](#r2-fc1f34eeb0619026b2a6) | wdra88 · 128/128/64 | coarse | 1.127760053 | 59.9824 | 0.9862× | 1.145840049, 1.127760053 | 1.60% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x1024x7168-fp16-G421-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x1024x7168-fp16-G421-3d4c2441dd5696bafd6f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0113`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x1024x7168-fp16-G421-fc1f34eeb0619026b2a6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0169`。

<a id="mnk-4608-2048-7168"></a>
## MNK 4608 × 2048 × 7168

### 第一轮 BF16 · G422 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.913439989 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 1.601600051 | 84.4727 | 1.1947× | 1.601600051, 1.603360057 | 0.11% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 1.874559999 | 72.1724 | 1.0207× | 1.875679970, 1.874559999 | 0.06% |
| 3 | [5e627b894b84cb1bc438](#r1-5e627b894b84cb1bc438) | nowasp · 128/128/64 | coarse | 1.913439989 | 70.7059 | 1.0000× | 1.927199960, 1.913439989 | 0.72% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x2048x7168-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0116`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x2048x7168-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0116`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x2048x7168-fp32-5e627b894b84cb1bc438`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0008`。
### 第二轮 FP16 · G422 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.896559954 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.896559954 | 71.3352 | 1.0000× | 1.896720052, 1.896559954 | 0.01% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.908959985 | 70.8718 | 0.9935× | 1.908959985, 1.911839962 | 0.15% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 1.969359994 | 68.6982 | 0.9630× | 1.969359994, 1.970800042 | 0.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `4608x2048x7168-fp16-G422-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 2：warmup=0，timed=2；原始结果 ID `4608x2048x7168-fp16-G422-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。
- Rank 3：warmup=0，timed=2；原始结果 ID `4608x2048x7168-fp16-G422-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0081`。

<a id="mnk-5120-64-4096"></a>
## MNK 5120 × 64 × 4096

### 第一轮 BF16 · G331 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.233360007 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.086159997 | 31.1555 | 2.7084× | 0.087439999, 0.086159997 | 1.49% |
| 2 | [c0cddd92b5bb88b9f96e](#r1-c0cddd92b5bb88b9f96e) | wdra88 · 64/64/128 | fine | 0.119439997 | 22.4745 | 1.9538× | 0.122960001, 0.119920000, 0.119439997 | 2.95% |
| 3 | [b8657961b1124e881dd5](#r1-b8657961b1124e881dd5) | wdra88 · 64/64/128 | fine | 0.119759999 | 22.4145 | 1.9486× | 0.120880000, 0.119759999 | 0.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x64x4096-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=3；原始结果 ID `5120x64x4096-bf16-c0cddd92b5bb88b9f96e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x64x4096-bf16-b8657961b1124e881dd5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0079`。
### 第二轮 FP16 · G331 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.157759994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b63491ed6531d89a40d](#r2-3b63491ed6531d89a40d) | wdra88 · 64/64/128 | coarse | 0.119920000 | 22.3845 | 1.3155× | 0.121359996, 0.119920000 | 1.20% |
| 2 | [3cf95baae0ff329949b1](#r2-3cf95baae0ff329949b1) | wdra48 · 64/64/128 | coarse | 0.130960003 | 20.4975 | 1.2046× | 0.135920003, 0.132560000, 0.130960003 | 3.79% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.157759994 | 17.0154 | 1.0000× | 0.162479997, 0.157759994, 0.163200006 | 3.45% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x64x4096-fp16-G331-3b63491ed6531d89a40d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0070`。
- Rank 2：warmup=0，timed=3；原始结果 ID `5120x64x4096-fp16-G331-3cf95baae0ff329949b1`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0054`。
- Rank 3：warmup=0，timed=3；原始结果 ID `5120x64x4096-fp16-G331-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。

<a id="mnk-5120-384-7168"></a>
## MNK 5120 × 384 × 7168

### 第一轮 BF16 · G332 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.552959979 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [4c7c4f499206cb9b1221](#r1-4c7c4f499206cb9b1221) | wdra48 · 64/64/64 | coarse | 0.407440007 | 69.1776 | 1.3572× | 0.407440007, 1.376559973, 1.381199956 | 238.99% |
| 2 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.547760010 | 51.4563 | 1.0095× | 0.550239980, 0.547760010 | 0.45% |
| 3 | [834aa4489131137d480c](#r1-834aa4489131137d480c) | nowasp · 64/64/64 | fine | 0.552240014 | 51.0389 | 1.0013× | 0.552800000, 0.552240014 | 0.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `5120x384x7168-bf16-4c7c4f499206cb9b1221`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0037`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x384x7168-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0110`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x384x7168-bf16-834aa4489131137d480c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0110`。
### 第二轮 FP16 · G332 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.551999986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.551999986 | 51.0611 | 1.0000× | 0.551999986, 0.552640021 | 0.12% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.602880001 | 46.7518 | 0.9156× | 0.602880001, 0.604879975 | 0.33% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.627040029 | 44.9504 | 0.8803× | 0.627040029, 0.645600021, 0.647199988 | 3.22% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x384x7168-fp16-G332-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0012`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x384x7168-fp16-G332-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 3：warmup=0，timed=3；原始结果 ID `5120x384x7168-fp16-G332-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0004`。

<a id="mnk-5120-512-4096"></a>
## MNK 5120 × 512 × 4096

### 第一轮 BF16 · G423 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.391999990 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 0.309040010 | 69.4889 | 1.2684× | 0.345039994, 0.312799990, 0.309040010 | 11.65% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 0.372799993 | 57.6042 | 1.0515× | 0.375840008, 0.372799993 | 0.82% |
| 3 | [249d0e074b4086e8aa18](#r1-249d0e074b4086e8aa18) | nowasp · 128/128/64 | fine | 0.391519994 | 54.8499 | 1.0012× | 0.400559992, 0.391519994, 0.393759996 | 2.31% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `5120x512x4096-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x512x4096-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
- Rank 3：warmup=0，timed=3；原始结果 ID `5120x512x4096-fp32-249d0e074b4086e8aa18`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0081`。
### 第二轮 FP16 · G423 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.378160000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.378160000 | 56.7877 | 1.0000× | 0.391039997, 0.378160000, 0.383599997 | 3.41% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.385919988 | 55.6458 | 0.9799× | 0.391999990, 0.385919988 | 1.58% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.396560013 | 54.1528 | 0.9536× | 0.399040014, 0.396560013 | 0.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `5120x512x4096-fp16-G423-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x512x4096-fp16-G423-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x512x4096-fp16-G423-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0081`。

<a id="mnk-5120-512-7168"></a>
## MNK 5120 × 512 × 7168

### 第一轮 BF16 · G333 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.676479995 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 0.544640005 | 69.0015 | 1.2421× | 0.583920002, 0.544640005, 0.545440018 | 7.21% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 0.644640028 | 58.2976 | 1.0494× | 0.644640028, 0.647040009 | 0.37% |
| 3 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.676479995 | 55.5537 | 1.0000× | 0.684000015, 0.676479995 | 1.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `5120x512x7168-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0110`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x512x7168-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0110`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x512x7168-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0005`。
### 第二轮 FP16 · G333 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.657280028 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.657280028 | 57.1765 | 1.0000× | 0.660160005, 0.657280028 | 0.44% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.671039999 | 56.0041 | 0.9795× | 0.671039999, 0.679040015 | 1.19% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.700559974 | 53.6442 | 0.9382× | 0.700559974, 0.701919973 | 0.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x512x7168-fp16-G333-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x512x7168-fp16-G333-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0004`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x512x7168-fp16-G333-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0078`。

<a id="mnk-5120-1024-4096"></a>
## MNK 5120 × 1024 × 4096

### 第一轮 BF16 · G424 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.752879977 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 0.650720000 | 66.0033 | 1.1570× | 0.653840005, 0.650720000 | 0.48% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 0.730719984 | 58.7772 | 1.0303× | 0.734879971, 0.730719984 | 0.57% |
| 3 | [249d0e074b4086e8aa18](#r1-249d0e074b4086e8aa18) | nowasp · 128/128/64 | fine | 0.742240012 | 57.8649 | 1.0143× | 0.757279992, 0.742240012, 0.753120005 | 2.03% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x1024x4096-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0092`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x1024x4096-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0092`。
- Rank 3：warmup=0，timed=3；原始结果 ID `5120x1024x4096-fp32-249d0e074b4086e8aa18`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0092`。
### 第二轮 FP16 · G424 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.720480025 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.720480025 | 59.6126 | 1.0000× | 0.747520030, 0.720480025, 0.734719992 | 3.75% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.738640010 | 58.1470 | 0.9754× | 0.748239994, 0.738640010 | 1.30% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.764479995 | 56.1816 | 0.9424× | 0.764479995, 0.775839984 | 1.49% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `5120x1024x4096-fp16-G424-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x1024x4096-fp16-G424-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x1024x4096-fp16-G424-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。

<a id="mnk-5120-2048-4096"></a>
## MNK 5120 × 2048 × 4096

### 第一轮 BF16 · G425 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.466160059 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 1.214800000 | 70.7107 | 1.2069× | 1.217200041, 1.214800000 | 0.20% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 1.402959943 | 61.2272 | 1.0450× | 1.406559944, 1.402959943 | 0.26% |
| 3 | [5e627b894b84cb1bc438](#r1-5e627b894b84cb1bc438) | nowasp · 128/128/64 | coarse | 1.466160059 | 58.5880 | 1.0000× | 1.467839956, 1.466160059 | 0.11% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x2048x4096-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0082`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x2048x4096-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0082`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x2048x4096-fp32-5e627b894b84cb1bc438`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0008`。
### 第二轮 FP16 · G425 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.313279986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.313279986 | 65.4083 | 1.0000× | 1.313279986, 1.323119998 | 0.75% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.414080024 | 60.7457 | 0.9287× | 1.420160055, 1.414080024 | 0.43% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.414399981 | 60.7320 | 0.9285× | 1.475839972, 1.446720004, 1.414399981 | 4.34% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x2048x4096-fp16-G425-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x2048x4096-fp16-G425-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 3：warmup=0，timed=3；原始结果 ID `5120x2048x4096-fp16-G425-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。

<a id="mnk-5120-2112-7168"></a>
## MNK 5120 × 2112 × 7168

### 第一轮 BF16 · G334 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.454879999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 2.088479996 | 74.2269 | 1.1754× | 2.088799953, 2.088479996 | 0.02% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 2.454879999 | 63.1483 | 1.0000× | 2.454879999, 2.455919981 | 0.04% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 2.462640047 | 62.9493 | 0.9968× | 2.462879896, 2.462640047 | 0.01% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x2112x7168-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0086`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x2112x7168-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0005`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x2112x7168-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0086`。
### 第二轮 FP16 · G334 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 2.448319912 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 2.448319912 | 63.3175 | 1.0000× | 2.459919930, 2.448319912 | 0.47% |
| 2 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 2.455840111 | 63.1236 | 0.9969× | 2.455840111, 2.457119942 | 0.05% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 2.541039944 | 61.0071 | 0.9635× | 2.541039944, 2.570080042 | 1.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x2112x7168-fp16-G334-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x2112x7168-fp16-G334-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0004`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x2112x7168-fp16-G334-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。

<a id="mnk-5120-4608-7168"></a>
## MNK 5120 × 4608 × 7168

### 第一轮 BF16 · G335 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 6.097519875 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 5.485040188 | 61.6638 | 1.1117× | 5.491040230, 5.485040188 | 0.11% |
| 2 | [f9a975d65680b8ccd8fe](#r1-f9a975d65680b8ccd8fe) | wasp44 · 128/128/64 | fine | 5.541840076 | 61.0318 | 1.1003× | 5.541840076, 5.571199894 | 0.53% |
| 3 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 5.566400051 | 60.7626 | 1.0954× | 5.581759930, 5.566400051 | 0.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x4608x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0132`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x4608x7168-bf16-f9a975d65680b8ccd8fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0132`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x4608x7168-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0132`。
### 第二轮 FP16 · G335 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 4.775119781 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 4.775119781 | 70.8315 | 1.0000× | 4.775119781, 4.850080013 | 1.57% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 5.274799824 | 64.1216 | 0.9053× | 5.274799824, 5.281839848 | 0.13% |
| 3 | [fc1f34eeb0619026b2a6](#r2-fc1f34eeb0619026b2a6) | wdra88 · 128/128/64 | coarse | 5.573760033 | 60.6823 | 0.8567× | 5.577439785, 5.573760033 | 0.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x4608x7168-fp16-G335-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x4608x7168-fp16-G335-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x4608x7168-fp16-G335-fc1f34eeb0619026b2a6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0166`。

<a id="mnk-5120-7168-256"></a>
## MNK 5120 × 7168 × 256

### 第一轮 BF16 · G336 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.398319989 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 0.377519995 | 49.7735 | 1.0551× | 0.379119992, 0.377519995 | 0.42% |
| 2 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 0.398160011 | 47.1933 | 1.0004× | 0.398160011, 0.399439991 | 0.32% |
| 3 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.398319989 | 47.1743 | 1.0000× | 0.400240004, 0.398319989 | 0.48% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x7168x256-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0111`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x7168x256-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0110`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x7168x256-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0005`。
### 第二轮 FP16 · G336 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.392879993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.392879993 | 47.8275 | 1.0000× | 0.398000002, 0.392879993 | 1.30% |
| 2 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.392879993 | 47.8275 | 1.0000× | 0.393680006, 0.392879993 | 0.20% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.400799990 | 46.8824 | 0.9802× | 0.400799990, 0.401919991 | 0.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x7168x256-fp16-G336-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x7168x256-fp16-G336-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x7168x256-fp16-G336-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0004`。

<a id="mnk-5120-7168-2304"></a>
## MNK 5120 × 7168 × 2304

### 第一轮 BF16 · G337 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.807120085 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 2.267040014 | 74.5970 | 1.2382× | 2.271840096, 2.267040014 | 0.21% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 2.666640043 | 63.4185 | 1.0527× | 2.666640043, 2.671920061 | 0.20% |
| 3 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 2.807120085 | 60.2448 | 1.0000× | 2.807120085, 2.900320053, 2.840640068 | 3.32% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x7168x2304-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x7168x2304-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
- Rank 3：warmup=0，timed=3；原始结果 ID `5120x7168x2304-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0005`。
### 第二轮 FP16 · G337 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 2.504159927 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 2.504159927 | 67.5334 | 1.0000× | 2.513600111, 2.504159927 | 0.38% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 2.705280066 | 62.5127 | 0.9257× | 2.705280066, 2.708080053 | 0.10% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 2.800479889 | 60.3876 | 0.8942× | 2.800479889, 2.833920002 | 1.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x7168x2304-fp16-G337-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x7168x2304-fp16-G337-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x7168x2304-fp16-G337-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。

<a id="mnk-5120-7168-8192"></a>
## MNK 5120 × 7168 × 8192

### 第一轮 BF16 · G338 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 10.182319641 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 9.770719528 | 61.5405 | 1.0421× | 9.770719528, 9.890399933 | 1.22% |
| 2 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 10.002559662 | 60.1142 | 1.0180× | 10.008319855, 10.002559662 | 0.06% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 10.085439682 | 59.6201 | 1.0096× | 10.085439682, 10.119359970 | 0.34% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x7168x8192-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0086`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x7168x8192-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0087`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x7168x8192-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0062`。
### 第二轮 FP16 · G338 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 8.373200417 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 8.373200417 | 71.8119 | 1.0000× | 8.373200417, 8.404319763 | 0.37% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 9.780400276 | 61.4796 | 0.8561× | 9.780400276, 9.824000359 | 0.45% |
| 3 | [467010bd09efdd1d21de](#r2-467010bd09efdd1d21de) | wdra88 · 128/128/64 | coarse | 10.085040092 | 59.6225 | 0.8303× | 10.098159790, 10.085040092 | 0.13% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x7168x8192-fp16-G338-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x7168x8192-fp16-G338-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x7168x8192-fp16-G338-467010bd09efdd1d21de`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0190`。

<a id="mnk-5120-12288-1536"></a>
## MNK 5120 × 12288 × 1536

### 第一轮 BF16 · G339 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.190639973 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 2.533839941 | 76.2769 | 1.2592× | 2.533839941, 2.538160086 | 0.17% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 2.926800013 | 66.0358 | 1.0901× | 2.926800013, 2.947360039 | 0.70% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 3.183120012 | 60.7183 | 1.0024× | 3.193279982, 3.183120012 | 0.32% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x12288x1536-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0087`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x12288x1536-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0087`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x12288x1536-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0087`。
### 第二轮 FP16 · G339 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 2.791440010 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 2.791440010 | 69.2379 | 1.0000× | 2.791440010, 2.801039934 | 0.34% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 3.136240005 | 61.6259 | 0.8901× | 3.136240005, 3.142960072 | 0.21% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.173759937 | 60.8973 | 0.8795× | 3.188719988, 3.173759937 | 0.47% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x12288x1536-fp16-G339-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x12288x1536-fp16-G339-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0078`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x12288x1536-fp16-G339-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。

<a id="mnk-5120-16384-512"></a>
## MNK 5120 × 16384 × 512

### 第一轮 BF16 · G340 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.574480057 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 1.388479948 | 61.8657 | 1.1340× | 1.388479948, 1.390640020 | 0.16% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 1.574480057 | 54.5573 | 1.0000× | 1.574480057, 1.578879952 | 0.28% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 1.578639984 | 54.4135 | 0.9974× | 1.578639984, 1.581680059 | 0.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x16384x512-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0087`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x16384x512-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0006`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x16384x512-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0087`。
### 第二轮 FP16 · G340 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.463039994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.463039994 | 58.7129 | 1.0000× | 1.463039994, 1.470800042 | 0.53% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.563359976 | 54.9453 | 0.9358× | 1.563359976, 1.572800040 | 0.60% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.574239969 | 54.5656 | 0.9294× | 1.574239969, 1.585919976 | 0.74% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `5120x16384x512-fp16-G340-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0174`。
- Rank 2：warmup=0，timed=2；原始结果 ID `5120x16384x512-fp16-G340-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0118`。
- Rank 3：warmup=0，timed=2；原始结果 ID `5120x16384x512-fp16-G340-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0005`。

<a id="mnk-6656-32-6144"></a>
## MNK 6656 × 32 × 6144

### 第一轮 BF16 · G341 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.362159997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.244800001 | 10.6914 | 1.4794× | 0.252240002, 0.244800001, 0.249359995 | 3.04% |
| 2 | [3717d3df609be657fa32](#r1-3717d3df609be657fa32) | wasp44 · 64/64/64 | fine | 0.248160005 | 10.5466 | 1.4594× | 0.248160005, 0.250479996 | 0.93% |
| 3 | [eee8f4403014e9f7aacb](#r1-eee8f4403014e9f7aacb) | wasp44 · 64/64/64 | fine | 0.250319988 | 10.4556 | 1.4468× | 0.250319988, 0.251439989 | 0.45% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x32x6144-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0079`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x32x6144-bf16-3717d3df609be657fa32`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x32x6144-bf16-eee8f4403014e9f7aacb`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0079`。
### 第二轮 FP16 · G341 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.275999993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.249679998 | 10.4824 | 1.1054× | 0.249679998, 0.249840006 | 0.06% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.275999993 | 9.4828 | 1.0000× | 0.278320014, 0.275999993 | 0.84% |
| 3 | [09b0e5995723b60e5a09](#r2-09b0e5995723b60e5a09) | wdra48 · 128/128/64 | coarse | 0.314799994 | 8.3140 | 0.8767× | 0.329840004, 0.314799994, 0.342640013 | 8.84% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x32x6144-fp16-G341-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x32x6144-fp16-G341-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0134`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x32x6144-fp16-G341-09b0e5995723b60e5a09`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0142`。

<a id="mnk-6656-64-2048"></a>
## MNK 6656 × 64 × 2048

### 第一轮 BF16 · G342 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.128079996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.093359999 | 18.6893 | 1.3719× | 0.096560001, 0.093359999, 0.095600002 | 3.43% |
| 2 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.094959997 | 18.3744 | 1.3488× | 0.097360000, 0.094959997, 0.096079998 | 2.53% |
| 3 | [eee8f4403014e9f7aacb](#r1-eee8f4403014e9f7aacb) | wasp44 · 64/64/64 | fine | 0.095440000 | 18.2820 | 1.3420× | 0.099600002, 0.095760003, 0.095440000 | 4.36% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x64x2048-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0111`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x64x2048-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0030`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x64x2048-bf16-eee8f4403014e9f7aacb`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0111`。
### 第二轮 FP16 · G342 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.103119999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.095600002 | 18.2514 | 1.0787× | 0.099440001, 0.095600002, 0.097360000 | 4.02% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.103119999 | 16.9204 | 1.0000× | 0.103119999, 0.104079999 | 0.93% |
| 3 | [b51d8b6af4744a99ab8e](#r2-b51d8b6af4744a99ab8e) | wdra48 · 128/128/64 | coarse | 0.119920000 | 14.5500 | 0.8599× | 0.133039996, 0.119920000, 0.124720000 | 10.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x64x2048-fp16-G342-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x64x2048-fp16-G342-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x64x2048-fp16-G342-b51d8b6af4744a99ab8e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0151`。

<a id="mnk-6656-64-4096"></a>
## MNK 6656 × 64 × 4096

### 第一轮 BF16 · G343 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.246480003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.176400006 | 19.7827 | 1.3973× | 0.176640004, 0.176400006 | 0.14% |
| 2 | [eee8f4403014e9f7aacb](#r1-eee8f4403014e9f7aacb) | wasp44 · 64/64/64 | fine | 0.177039996 | 19.7111 | 1.3922× | 0.177039996, 0.177039996 | 0.00% |
| 3 | [3717d3df609be657fa32](#r1-3717d3df609be657fa32) | wasp44 · 64/64/64 | fine | 0.177200004 | 19.6933 | 1.3910× | 0.178479999, 0.177200004 | 0.72% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x64x4096-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x64x4096-bf16-eee8f4403014e9f7aacb`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0074`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x64x4096-bf16-3717d3df609be657fa32`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0074`。
### 第二轮 FP16 · G343 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.190320000 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.175919995 | 19.8366 | 1.0819× | 0.175919995, 0.177440003 | 0.86% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.190320000 | 18.3358 | 1.0000× | 0.194800004, 0.190799996, 0.190320000 | 2.35% |
| 3 | [49eeff2882f3db7855ce](#r2-49eeff2882f3db7855ce) | wdra48 · 128/128/64 | coarse | 0.222159997 | 15.7079 | 0.8567× | 0.243279994, 0.222159997, 0.238319993 | 9.51% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x64x4096-fp16-G343-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x64x4096-fp16-G343-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x64x4096-fp16-G343-49eeff2882f3db7855ce`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0046`。

<a id="mnk-6656-64-7168"></a>
## MNK 6656 × 64 × 7168

### 第一轮 BF16 · G344 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.428799987 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.294719994 | 20.7210 | 1.4549× | 0.294719994, 0.297679991 | 1.00% |
| 2 | [3717d3df609be657fa32](#r1-3717d3df609be657fa32) | wasp44 · 64/64/64 | fine | 0.294959992 | 20.7042 | 1.4538× | 0.300240010, 0.294959992 | 1.79% |
| 3 | [eee8f4403014e9f7aacb](#r1-eee8f4403014e9f7aacb) | wasp44 · 64/64/64 | fine | 0.298159987 | 20.4820 | 1.4382× | 0.298319995, 0.298159987 | 0.05% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x64x7168-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x64x7168-bf16-3717d3df609be657fa32`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x64x7168-bf16-eee8f4403014e9f7aacb`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
### 第二轮 FP16 · G344 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.337839991 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.297360003 | 20.5371 | 1.1361× | 0.297360003, 0.297600001 | 0.08% |
| 2 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.337839991 | 18.0763 | 1.0000× | 0.340000004, 0.337839991 | 0.64% |
| 3 | [fc1f34eeb0619026b2a6](#r2-fc1f34eeb0619026b2a6) | wdra88 · 128/128/64 | coarse | 0.378320009 | 16.1422 | 0.8930× | 0.381199986, 0.378320009 | 0.76% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x64x7168-fp16-G344-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0030`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x64x7168-fp16-G344-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x64x7168-fp16-G344-fc1f34eeb0619026b2a6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0167`。

<a id="mnk-6656-128-4096"></a>
## MNK 6656 × 128 × 4096

### 第一轮 BF16 · G345 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.259119987 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.196400002 | 35.5363 | 1.3193× | 0.198320001, 0.196400002 | 0.98% |
| 2 | [eee8f4403014e9f7aacb](#r1-eee8f4403014e9f7aacb) | wasp44 · 64/64/64 | fine | 0.211040005 | 33.0711 | 1.2278× | 0.211999997, 0.211040005 | 0.45% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.213520005 | 32.6870 | 1.2136× | 0.216399997, 0.213520005 | 1.35% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x128x4096-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x128x4096-bf16-eee8f4403014e9f7aacb`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x128x4096-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0030`。
### 第二轮 FP16 · G345 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.231439993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.211439997 | 33.0085 | 1.0946× | 0.213440001, 0.211439997 | 0.95% |
| 2 | [fc1f34eeb0619026b2a6](#r2-fc1f34eeb0619026b2a6) | wdra88 · 128/128/64 | coarse | 0.225520000 | 30.9477 | 1.0263× | 0.225520000, 0.225520000 | 0.00% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 0.226160005 | 30.8601 | 1.0233× | 0.226160005, 0.230000004 | 1.70% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x128x4096-fp16-G345-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0031`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x128x4096-fp16-G345-fc1f34eeb0619026b2a6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0167`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x128x4096-fp16-G345-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0063`。

<a id="mnk-6656-192-4096"></a>
## MNK 6656 × 192 × 4096

### 第一轮 BF16 · G346 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.281679988 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.268400013 | 39.0051 | 1.0495× | 0.269039989, 0.268400013 | 0.24% |
| 2 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.278640002 | 37.5717 | 1.0109× | 0.283679992, 0.278640002 | 1.81% |
| 3 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.281679988 | 37.1662 | 1.0000× | 0.283439994, 0.281679988 | 0.62% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x192x4096-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0111`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x192x4096-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0111`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x192x4096-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0014`。
### 第二轮 FP16 · G346 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.277280003 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.277280003 | 37.7560 | 1.0000× | 0.279599994, 0.277280003 | 0.84% |
| 2 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.281439990 | 37.1979 | 0.9852× | 0.283360004, 0.281439990 | 0.68% |
| 3 | [d8809ef3e165e6edd726](#r2-d8809ef3e165e6edd726) | nowasp · 64/64/64 | coarse | 0.292400002 | 35.8036 | 0.9483× | 0.294000000, 0.292400002 | 0.55% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x192x4096-fp16-G346-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x192x4096-fp16-G346-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0013`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x192x4096-fp16-G346-d8809ef3e165e6edd726`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0095`。

<a id="mnk-6656-256-2048"></a>
## MNK 6656 × 256 × 2048

### 第一轮 BF16 · G347 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.167919993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.013840000 | 504.2863 | 12.1329× | 0.014640000, 0.014320000, 0.013840000 | 5.78% |
| 2 | [8fb190809a17bfcaff1a](#r1-8fb190809a17bfcaff1a) | nowasp · 64/64/64 | fine | 0.013840000 | 504.2863 | 12.1329× | 0.015280000, 0.014000000, 0.013840000 | 10.40% |
| 3 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.013840000 | 504.2863 | 12.1329× | 0.014640000, 0.013840000, 0.014800000 | 6.94% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x256x2048-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x256x2048-bf16-8fb190809a17bfcaff1a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x256x2048-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
### 第二轮 FP16 · G347 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.169119999 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.169119999 | 41.2685 | 1.0000× | 0.169599995, 0.169119999 | 0.28% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.190960005 | 36.5486 | 0.8856× | 0.191760004, 0.190960005 | 0.42% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.205679998 | 33.9329 | 0.8222× | 0.205840006, 0.205679998 | 0.08% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x256x2048-fp16-G347-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0013`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x256x2048-fp16-G347-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0031`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x256x2048-fp16-G347-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。

<a id="mnk-6656-256-6144"></a>
## MNK 6656 × 256 × 6144

### 第一轮 BF16 · G348 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.474159986 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [8fb190809a17bfcaff1a](#r1-8fb190809a17bfcaff1a) | nowasp · 64/64/64 | fine | 0.013840000 | 1512.8588 | 34.2601× | 0.015280000, 0.013840000, 0.014000000 | 10.40% |
| 2 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.014320000 | 1462.1484 | 33.1117× | 0.015600000, 0.014480000, 0.014320000 | 8.94% |
| 3 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.014320000 | 1462.1484 | 33.1117× | 0.014640000, 0.014320000, 0.014640000 | 2.23% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x256x6144-bf16-8fb190809a17bfcaff1a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x256x6144-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x256x6144-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
### 第二轮 FP16 · G348 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.472799987 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.472799987 | 44.2850 | 1.0000× | 0.476000011, 0.472799987 | 0.68% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.529680014 | 39.5295 | 0.8926× | 0.529680014, 0.535279989 | 1.06% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.584399998 | 35.8281 | 0.8090× | 0.586319983, 0.584399998 | 0.33% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x256x6144-fp16-G348-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0013`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x256x6144-fp16-G348-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0031`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x256x6144-fp16-G348-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。

<a id="mnk-6656-384-7168"></a>
## MNK 6656 × 384 × 7168

### 第一轮 BF16 · G349 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.624960005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 0.546880007 | 67.0009 | 1.1428× | 0.549839973, 0.546880007 | 0.54% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.624960005 | 58.6301 | 1.0000× | 0.643999994, 0.624960005, 0.630720019 | 3.05% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 0.628880024 | 58.2646 | 0.9938× | 0.639440000, 0.628880024 | 1.68% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x384x7168-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0112`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x384x7168-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0006`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x384x7168-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0112`。
### 第二轮 FP16 · G349 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.624960005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.624960005 | 58.6301 | 1.0000× | 0.640320003, 0.627200007, 0.624960005 | 2.46% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.638239980 | 57.4101 | 0.9792× | 0.640799999, 0.638239980 | 0.40% |
| 3 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 0.674319983 | 54.3384 | 0.9268× | 0.681439996, 0.674319983 | 1.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x384x7168-fp16-G349-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0005`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x384x7168-fp16-G349-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x384x7168-fp16-G349-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。

<a id="mnk-6656-512-4096"></a>
## MNK 6656 × 512 × 4096

### 第一轮 BF16 · G350 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.555760026 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.020400001 | 1368.4944 | 27.2431× | 0.021199999, 0.020400001, 0.021040000 | 3.92% |
| 2 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.023119999 | 1207.4952 | 24.0381× | 0.024240000, 0.023119999, 0.023920000 | 4.84% |
| 3 | [57278aa2e4dbb73c63c3](#r1-57278aa2e4dbb73c63c3) | nowasp · 64/64/64 | fine | 0.039600000 | 704.9820 | 14.0343× | 0.041999999, 0.039600000, 0.039600000 | 6.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x512x4096-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x512x4096-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x512x4096-bf16-57278aa2e4dbb73c63c3`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0088`。
### 第二轮 FP16 · G350 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.555519998 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.555519998 | 50.2543 | 1.0000× | 0.557280004, 0.555519998 | 0.32% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.580079973 | 48.1266 | 0.9577× | 0.583280027, 0.580079973 | 0.55% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.623279989 | 44.7909 | 0.8913× | 0.625360012, 0.623279989 | 0.33% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp16-G350-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0013`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp16-G350-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0031`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp16-G350-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0135`。
### 第一轮 BF16 · G426 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.557359993 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9fbc98f298852c6db1bc](#r1-9fbc98f298852c6db1bc) | nowasp · 64/64/64 | fine | 0.552079976 | 50.5675 | 1.0096× | 0.552800000, 0.552079976 | 0.13% |
| 2 | [39eb19a1c9076aa7945a](#r1-39eb19a1c9076aa7945a) | nowasp · 64/64/64 | fine | 0.556400001 | 50.1749 | 1.0017× | 0.556400001, 0.558319986 | 0.35% |
| 3 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 0.557359993 | 50.0884 | 1.0000× | 0.558160007, 0.557359993 | 0.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp32-9fbc98f298852c6db1bc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0074`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp32-39eb19a1c9076aa7945a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0074`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
### 第二轮 FP16 · G426 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.555999994 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.555999994 | 50.2109 | 1.0000× | 0.556479990, 0.555999994 | 0.09% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.579919994 | 48.1399 | 0.9588× | 0.580879986, 0.579919994 | 0.17% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.621680021 | 44.9062 | 0.8944× | 0.622640014, 0.621680021 | 0.15% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp16-G426-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp16-G426-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0033`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x4096-fp16-G426-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0137`。

<a id="mnk-6656-512-7168"></a>
## MNK 6656 × 512 × 7168

### 第一轮 BF16 · G351 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.979200006 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.976400018 | 50.0361 | 1.0029× | 0.978479981, 0.976400018 | 0.21% |
| 2 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.979200006 | 49.8930 | 1.0000× | 0.982320011, 0.979200006 | 0.32% |
| 3 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.981199980 | 49.7913 | 0.9980× | 0.984879971, 0.981199980 | 0.38% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x7168-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0112`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x7168-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0014`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x7168-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0112`。
### 第二轮 FP16 · G351 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.979520023 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.979520023 | 49.8767 | 1.0000× | 0.979520023, 0.983200014 | 0.38% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 1.000720024 | 48.8201 | 0.9788× | 1.002239943, 1.000720024 | 0.15% |
| 3 | [3d4c2441dd5696bafd6f](#r2-3d4c2441dd5696bafd6f) | wdra88 · 128/128/64 | coarse | 1.109120011 | 44.0487 | 0.8832× | 1.111199975, 1.109120011 | 0.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp16-G351-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0013`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp16-G351-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0031`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp16-G351-3d4c2441dd5696bafd6f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0111`。
### 第一轮 BF16 · G427 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.975839972 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 0.975839972 | 50.0648 | 1.0000× | 0.977440000, 0.975839972 | 0.16% |
| 2 | [39eb19a1c9076aa7945a](#r1-39eb19a1c9076aa7945a) | nowasp · 64/64/64 | fine | 0.978320003 | 49.9379 | 0.9975× | 0.978320003, 0.978559971 | 0.02% |
| 3 | [9fbc98f298852c6db1bc](#r1-9fbc98f298852c6db1bc) | nowasp · 64/64/64 | fine | 0.979759991 | 49.8645 | 0.9960× | 0.980400026, 0.979759991 | 0.07% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp32-39eb19a1c9076aa7945a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp32-9fbc98f298852c6db1bc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
### 第二轮 FP16 · G427 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.980480015 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.980480015 | 49.8279 | 1.0000× | 0.980480015, 0.982400000 | 0.20% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 1.001680017 | 48.7733 | 0.9788× | 1.002959967, 1.001680017 | 0.13% |
| 3 | [fc1f34eeb0619026b2a6](#r2-fc1f34eeb0619026b2a6) | wdra88 · 128/128/64 | coarse | 1.106080055 | 44.1697 | 0.8864× | 1.109359980, 1.106080055 | 0.30% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp16-G427-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp16-G427-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0033`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x512x7168-fp16-G427-fc1f34eeb0619026b2a6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0169`。

<a id="mnk-6656-1024-2048"></a>
## MNK 6656 × 1024 × 2048

### 第一轮 BF16 · G352 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.535120010 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.534319997 | 52.2483 | 1.0015× | 0.534319997, 0.535359979 | 0.19% |
| 2 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.534560025 | 52.2248 | 1.0010× | 0.535920024, 0.534560025 | 0.25% |
| 3 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.535120010 | 52.1701 | 1.0000× | 0.538479984, 0.535120010 | 0.63% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x1024x2048-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0113`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x1024x2048-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0112`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x1024x2048-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0014`。
### 第二轮 FP16 · G352 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.501519978 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.501519978 | 55.6654 | 1.0000× | 0.501519978, 0.506079972 | 0.91% |
| 2 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.535040021 | 52.1779 | 0.9374× | 0.535040021, 0.536319971 | 0.24% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.545679986 | 51.1605 | 0.9191× | 0.558560014, 0.545679986, 0.550000012 | 2.36% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x1024x2048-fp16-G352-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x1024x2048-fp16-G352-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0013`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x1024x2048-fp16-G352-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。

<a id="mnk-6656-1024-4096"></a>
## MNK 6656 × 1024 × 4096

### 第一轮 BF16 · G428 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.052240014 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [9fbc98f298852c6db1bc](#r1-9fbc98f298852c6db1bc) | nowasp · 64/64/64 | fine | 1.050320029 | 53.1596 | 1.0018× | 1.052000046, 1.050320029 | 0.16% |
| 2 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 1.052240014 | 53.0626 | 1.0000× | 1.053679943, 1.052240014 | 0.14% |
| 3 | [39eb19a1c9076aa7945a](#r1-39eb19a1c9076aa7945a) | nowasp · 64/64/64 | fine | 1.054319978 | 52.9579 | 0.9980× | 1.054319978, 1.054960012 | 0.06% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x1024x4096-fp32-9fbc98f298852c6db1bc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0082`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x1024x4096-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x1024x4096-fp32-39eb19a1c9076aa7945a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0082`。
### 第二轮 FP16 · G428 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.967360020 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.967360020 | 57.7185 | 1.0000× | 0.974799991, 0.967360020 | 0.77% |
| 2 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 1.049440026 | 53.2042 | 0.9218× | 1.050719976, 1.049440026 | 0.12% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.062080026 | 52.5710 | 0.9108× | 1.117439985, 1.068639994, 1.062080026 | 5.21% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x1024x4096-fp16-G428-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x1024x4096-fp16-G428-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0015`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x1024x4096-fp16-G428-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。

<a id="mnk-6656-1024-7168"></a>
## MNK 6656 × 1024 × 7168

### 第一轮 BF16 · G429 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.883440018 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [32d663c1fff0030fd159](#r1-32d663c1fff0030fd159) | wdra88 · 128/128/64 | fine | 1.859200001 | 52.5551 | 1.0130× | 1.859200001, 1.866960049 | 0.42% |
| 2 | [f7627acc64f8279814ee](#r1-f7627acc64f8279814ee) | wdra88 · 128/128/64 | fine | 1.875759959 | 52.0912 | 1.0041× | 1.899440050, 1.875759959 | 1.26% |
| 3 | [0f7a5be9b0919f6d6b2e](#r1-0f7a5be9b0919f6d6b2e) | nowasp · 64/64/64 | coarse | 1.883440018 | 51.8787 | 1.0000× | 1.912559986, 1.883440018 | 1.55% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x1024x7168-fp32-32d663c1fff0030fd159`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0076`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x1024x7168-fp32-f7627acc64f8279814ee`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0076`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x1024x7168-fp32-0f7a5be9b0919f6d6b2e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0016`。
### 第二轮 FP16 · G429 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.718880057 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.718880057 | 56.8454 | 1.0000× | 1.718880057, 1.732640028 | 0.80% |
| 2 | [3d4c2441dd5696bafd6f](#r2-3d4c2441dd5696bafd6f) | wdra88 · 128/128/64 | coarse | 1.846320033 | 52.9218 | 0.9310× | 1.854480028, 1.846320033 | 0.44% |
| 3 | [22228385a301e0ff14c5](#r2-22228385a301e0ff14c5) | wdra88 · 128/128/64 | coarse | 1.850000024 | 52.8165 | 0.9291× | 1.850000024, 1.851760030 | 0.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x1024x7168-fp16-G429-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x1024x7168-fp16-G429-3d4c2441dd5696bafd6f`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0113`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x1024x7168-fp16-G429-22228385a301e0ff14c5`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0105`。

<a id="mnk-6656-2048-512"></a>
## MNK 6656 × 2048 × 512

### 第一轮 BF16 · G353 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.259440005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 0.234160006 | 59.6116 | 1.1080× | 0.235919997, 0.234160006 | 0.75% |
| 2 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 0.259279996 | 53.8362 | 1.0006× | 0.267439991, 0.261359990, 0.259279996 | 3.15% |
| 3 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.259440005 | 53.8030 | 1.0000× | 0.266479999, 0.259440005, 0.261680007 | 2.71% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x512-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0113`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x2048x512-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0113`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x2048x512-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0006`。
### 第二轮 FP16 · G353 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.250479996 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.250479996 | 55.7276 | 1.0000× | 0.253839999, 0.250479996 | 1.34% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.257600009 | 54.1873 | 0.9724× | 0.264079988, 0.257600009, 0.258960009 | 2.52% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.257759988 | 54.1536 | 0.9718× | 0.265280008, 0.259359986, 0.257759988 | 2.92% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x512-fp16-G353-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=3；原始结果 ID `6656x2048x512-fp16-G353-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x2048x512-fp16-G353-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0005`。

<a id="mnk-6656-2048-4096"></a>
## MNK 6656 × 2048 × 4096

### 第一轮 BF16 · G354 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.806800008 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 1.483919978 | 75.2528 | 1.2176× | 1.483919978, 1.521839976, 1.526319981 | 2.86% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 1.733600020 | 64.4146 | 1.0422× | 1.741199970, 1.733600020 | 0.44% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 1.782480001 | 62.6482 | 1.0136× | 1.782480001, 1.804479957 | 1.23% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=3；原始结果 ID `6656x2048x4096-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0113`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2048x4096-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0113`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2048x4096-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0113`。
### 第二轮 FP16 · G354 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.690320015 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.690320015 | 66.0639 | 1.0000× | 1.690719962, 1.690320015 | 0.02% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.764879942 | 63.2729 | 0.9578× | 1.765759945, 1.764879942 | 0.05% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.816640019 | 61.4702 | 0.9305× | 1.827679992, 1.816640019 | 0.61% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp16-G354-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp16-G354-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp16-G354-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0005`。
### 第一轮 BF16 · G430 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 1.838080049 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 1.511759996 | 73.8670 | 1.2159× | 1.526319981, 1.511759996 | 0.96% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 1.756479979 | 63.5755 | 1.0465× | 1.763679981, 1.756479979 | 0.41% |
| 3 | [249d0e074b4086e8aa18](#r1-249d0e074b4086e8aa18) | nowasp · 128/128/64 | fine | 1.821280003 | 61.3136 | 1.0092× | 1.821280003, 1.870319963, 1.822960019 | 2.69% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0075`。
- Rank 3：warmup=0，timed=3；原始结果 ID `6656x2048x4096-fp32-249d0e074b4086e8aa18`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0074`。
### 第二轮 FP16 · G430 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.677360058 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.677360058 | 66.5743 | 1.0000× | 1.677360058, 1.690719962 | 0.80% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 1.764479995 | 63.2873 | 0.9506× | 1.771440029, 1.764479995 | 0.39% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 1.810080051 | 61.6929 | 0.9267× | 1.810080051, 1.815199971 | 0.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp16-G430-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp16-G430-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2048x4096-fp16-G430-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。

<a id="mnk-6656-2048-7168"></a>
## MNK 6656 × 2048 × 7168

### 第一轮 BF16 · G431 · 输出 FP32

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.136399984 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [93e7f64a414acfefdf5a](#r1-93e7f64a414acfefdf5a) | nowasp · 128/128/64 | fine | 2.619839907 | 74.5927 | 1.1972× | 2.623919964, 2.619839907 | 0.16% |
| 2 | [6bffccea3bac78208f44](#r1-6bffccea3bac78208f44) | nowasp · 128/128/64 | fine | 3.084000111 | 63.3661 | 1.0170× | 3.084000111, 3.087039948 | 0.10% |
| 3 | [249d0e074b4086e8aa18](#r1-249d0e074b4086e8aa18) | nowasp · 128/128/64 | fine | 3.131680012 | 62.4013 | 1.0015× | 3.144320011, 3.131680012 | 0.40% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x7168-fp32-93e7f64a414acfefdf5a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0076`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2048x7168-fp32-6bffccea3bac78208f44`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0076`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2048x7168-fp32-249d0e074b4086e8aa18`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0076`。
### 第二轮 FP16 · G431 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 2.988640070 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 2.988640070 | 65.3879 | 1.0000× | 2.988640070, 2.999759912 | 0.37% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.109440088 | 62.8477 | 0.9612× | 3.111680031, 3.109440088 | 0.07% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 3.118880033 | 62.6574 | 0.9582× | 3.118880033, 3.122560024 | 0.12% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2048x7168-fp16-G431-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0177`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2048x7168-fp16-G431-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0121`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2048x7168-fp16-G431-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0007`。

<a id="mnk-6656-2112-7168"></a>
## MNK 6656 × 2112 × 7168

### 第一轮 BF16 · G355 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.061280012 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 2.605439901 | 77.3489 | 1.1750× | 2.605439901, 2.609280109 | 0.15% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 3.061280012 | 65.8313 | 1.0000× | 3.069360018, 3.061280012 | 0.26% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 3.073280096 | 65.5742 | 0.9961× | 3.075839996, 3.073280096 | 0.08% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2112x7168-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0089`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2112x7168-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0006`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2112x7168-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0089`。
### 第二轮 FP16 · G355 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.069760084 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 3.069760084 | 65.6494 | 1.0000× | 3.073120117, 3.069760084 | 0.11% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.071439981 | 65.6135 | 0.9995× | 3.071439981, 3.072319984 | 0.03% |
| 3 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.187279940 | 63.2288 | 0.9631× | 3.187279940, 3.196880102 | 0.30% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x2112x7168-fp16-G355-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0005`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x2112x7168-fp16-G355-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x2112x7168-fp16-G355-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。

<a id="mnk-6656-4608-7168"></a>
## MNK 6656 × 4608 × 7168

### 第一轮 BF16 · G356 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 7.686399937 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 7.276879787 | 60.4239 | 1.0563× | 7.295199871, 7.276879787 | 0.25% |
| 2 | [f9a975d65680b8ccd8fe](#r1-f9a975d65680b8ccd8fe) | wasp44 · 128/128/64 | fine | 7.368800163 | 59.6701 | 1.0431× | 7.389120102, 7.368800163 | 0.28% |
| 3 | [2943f23873d8087254e4](#r1-2943f23873d8087254e4) | wdra88 · 128/128/64 | fine | 7.405600071 | 59.3736 | 1.0379× | 7.416240215, 7.405600071 | 0.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x4608x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x4608x7168-bf16-f9a975d65680b8ccd8fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x4608x7168-bf16-2943f23873d8087254e4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0133`。
### 第二轮 FP16 · G356 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 6.212240219 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 6.212240219 | 70.7792 | 1.0000× | 6.232240200, 6.212240219 | 0.32% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 6.798319817 | 64.6773 | 0.9138× | 6.798319817, 6.804319859 | 0.09% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 7.404799938 | 59.3800 | 0.8389× | 7.404799938, 7.414720058 | 0.13% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x4608x7168-fp16-G356-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x4608x7168-fp16-G356-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x4608x7168-fp16-G356-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0063`。

<a id="mnk-6656-7168-256"></a>
## MNK 6656 × 7168 × 256

### 第一轮 BF16 · G357 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.515280008 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 0.485520005 | 50.3123 | 1.0613× | 0.486319989, 0.485520005 | 0.16% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 0.515280008 | 47.4065 | 1.0000× | 0.517679989, 0.515280008 | 0.47% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 0.516080022 | 47.3330 | 0.9984× | 0.516080022, 0.517040014 | 0.19% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x7168x256-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0114`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x7168x256-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0006`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x7168x256-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0114`。
### 第二轮 FP16 · G357 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.502640009 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 0.502640009 | 48.5987 | 1.0000× | 0.506320000, 0.502640009 | 0.73% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 0.507839978 | 48.1010 | 0.9898× | 0.512399971, 0.507839978 | 0.90% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 0.516799986 | 47.2671 | 0.9726× | 0.519200027, 0.516799986 | 0.46% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x7168x256-fp16-G357-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x7168x256-fp16-G357-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x7168x256-fp16-G357-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0006`。

<a id="mnk-6656-7168-2304"></a>
## MNK 6656 × 7168 × 2304

### 第一轮 BF16 · G358 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 3.770720005 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 2.994800091 | 73.4101 | 1.2591× | 2.998320103, 2.994800091 | 0.12% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 3.464479923 | 63.4579 | 1.0884× | 3.464479923, 3.474400043 | 0.29% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 3.729759932 | 58.9444 | 1.0110× | 3.779599905, 3.729759932 | 1.34% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x7168x2304-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x7168x2304-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x7168x2304-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
### 第二轮 FP16 · G358 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.185679913 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.185679913 | 69.0115 | 1.0000× | 3.185679913, 3.192399979 | 0.21% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 3.587519884 | 61.2815 | 0.8880× | 3.587519884, 3.594959974 | 0.21% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 3.758719921 | 58.4903 | 0.8475× | 3.794800043, 3.758719921 | 0.96% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x7168x2304-fp16-G358-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x7168x2304-fp16-G358-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x7168x2304-fp16-G358-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。

<a id="mnk-6656-7168-8192"></a>
## MNK 6656 × 7168 × 8192

### 第一轮 BF16 · G359 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 13.866399765 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 12.935440063 | 60.4296 | 1.0720× | 12.935440063, 12.949999809 | 0.11% |
| 2 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 13.022720337 | 60.0246 | 1.0648× | 13.034879684, 13.022720337 | 0.09% |
| 3 | [a57bcf6a22eedf684051](#r1-a57bcf6a22eedf684051) | wdra88 · 128/128/64 | coarse | 13.024319649 | 60.0173 | 1.0647× | 13.024319649, 13.024640083 | 0.00% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x7168x8192-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0089`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x7168x8192-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0089`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x7168x8192-bf16-a57bcf6a22eedf684051`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0063`。
### 第二轮 FP16 · G359 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 10.767279625 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 10.767279625 | 72.5981 | 1.0000× | 10.817520142, 10.767279625 | 0.47% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 12.819519997 | 60.9761 | 0.8399× | 12.872960091, 12.819519997 | 0.42% |
| 3 | [fc1f34eeb0619026b2a6](#r2-fc1f34eeb0619026b2a6) | wdra88 · 128/128/64 | coarse | 13.018560410 | 60.0438 | 0.8271× | 13.020879745, 13.018560410 | 0.02% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x7168x8192-fp16-G359-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x7168x8192-fp16-G359-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x7168x8192-fp16-G359-fc1f34eeb0619026b2a6`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0167`。

<a id="mnk-6656-9216-2048"></a>
## MNK 6656 × 9216 × 2048

### 第一轮 BF16 · G360 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.359839916 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 3.229599953 | 77.7977 | 1.3500× | 3.229919910, 3.229599953 | 0.01% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 4.157760143 | 60.4305 | 1.0486× | 4.168320179, 4.157760143 | 0.25% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 4.359200001 | 57.6380 | 1.0001× | 4.359200001, 4.390399933 | 0.72% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x9216x2048-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x9216x2048-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x9216x2048-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
### 第二轮 FP16 · G360 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.561680079 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.561680079 | 70.5441 | 1.0000× | 3.584160089, 3.561680079 | 0.63% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 4.145840168 | 60.6043 | 0.8591× | 4.172319889, 4.145840168 | 0.64% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 4.313360214 | 58.2505 | 0.8257× | 4.313360214, 4.341919899 | 0.66% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x9216x2048-fp16-G360-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x9216x2048-fp16-G360-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x9216x2048-fp16-G360-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0119`。

<a id="mnk-6656-12288-1536"></a>
## MNK 6656 × 12288 × 1536

### 第一轮 BF16 · G361 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 4.169360161 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 3.295680046 | 76.2379 | 1.2651× | 3.295680046, 3.303519964 | 0.24% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 3.808720112 | 65.9685 | 1.0947× | 3.808720112, 3.810080051 | 0.04% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 4.165359974 | 60.3203 | 1.0010× | 4.212719917, 4.165359974 | 1.14% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x12288x1536-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0090`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x12288x1536-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0090`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x12288x1536-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0090`。
### 第二轮 FP16 · G361 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 3.600159883 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 3.600159883 | 69.7901 | 1.0000× | 3.600159883, 3.629679918 | 0.82% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 4.061279774 | 61.8661 | 0.8865× | 4.092720032, 4.061279774 | 0.77% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 4.122879982 | 60.9418 | 0.8732× | 4.122879982, 4.188000202 | 1.58% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x12288x1536-fp16-G361-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0175`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x12288x1536-fp16-G361-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x12288x1536-fp16-G361-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0006`。

<a id="mnk-6656-12288-2048"></a>
## MNK 6656 × 12288 × 2048

### 第一轮 BF16 · G362 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 5.510000229 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 4.308959961 | 77.7467 | 1.2787× | 4.308959961, 4.317520142 | 0.20% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 4.959199905 | 67.5527 | 1.1111× | 5.035359859, 4.959199905 | 1.54% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 5.485680103 | 61.0694 | 1.0044× | 5.485680103, 5.495759964 | 0.18% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x12288x2048-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x12288x2048-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x12288x2048-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0134`。
### 第二轮 FP16 · G362 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 4.699600220 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 4.699600220 | 71.2842 | 1.0000× | 4.699600220, 4.715119839 | 0.33% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 5.377200127 | 62.3015 | 0.8740× | 5.381120205, 5.377200127 | 0.07% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 5.596159935 | 59.8638 | 0.8398× | 5.596159935, 5.657440186 | 1.10% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x12288x2048-fp16-G362-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0176`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x12288x2048-fp16-G362-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0079`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x12288x2048-fp16-G362-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0120`。

<a id="mnk-6656-16384-512"></a>
## MNK 6656 × 16384 × 512

### 第一轮 BF16 · G363 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 2.042320013 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 1.801599979 | 61.9833 | 1.1336× | 1.801599979, 1.805199981 | 0.20% |
| 2 | [b3864ea759acb2e2470a](#r1-b3864ea759acb2e2470a) | nowasp · 128/128/64 | coarse | 2.042320013 | 54.6776 | 1.0000× | 2.042320013, 2.042479992 | 0.01% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 2.044480085 | 54.6198 | 0.9989× | 2.053359985, 2.044480085 | 0.43% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x16384x512-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0090`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x16384x512-bf16-b3864ea759acb2e2470a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0007`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x16384x512-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0090`。
### 第二轮 FP16 · G363 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 1.895760059 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 1.895760059 | 58.9047 | 1.0000× | 1.895760059, 1.899440050 | 0.19% |
| 2 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 2.026160002 | 55.1137 | 0.9356× | 2.026160002, 2.030960083 | 0.24% |
| 3 | [ddb93ae85b5fc1cfdac0](#r2-ddb93ae85b5fc1cfdac0) | nowasp · 128/128/64 | coarse | 2.041599989 | 54.6969 | 0.9286× | 2.045120001, 2.041599989 | 0.17% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `6656x16384x512-fp16-G363-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0176`。
- Rank 2：warmup=0，timed=2；原始结果 ID `6656x16384x512-fp16-G363-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0120`。
- Rank 3：warmup=0，timed=2；原始结果 ID `6656x16384x512-fp16-G363-ddb93ae85b5fc1cfdac0`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0006`。

<a id="mnk-28672-16-4096"></a>
## MNK 28672 × 16 × 4096

### 第一轮 BF16 · G364 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.332879990 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [91c581677069aeb17efa](#r1-91c581677069aeb17efa) | nowasp · 64/64/64 | fine | 0.331279993 | 11.3442 | 1.0048× | 0.331600010, 0.331279993 | 0.10% |
| 2 | [cb8c8401e74da8da4f02](#r1-cb8c8401e74da8da4f02) | nowasp · 64/64/64 | coarse | 0.332879990 | 11.2896 | 1.0000× | 0.335119992, 0.332879990 | 0.67% |
| 3 | [e8f8cacd934b18fb3037](#r1-e8f8cacd934b18fb3037) | nowasp · 64/64/64 | fine | 0.333039999 | 11.2842 | 0.9995× | 0.338319987, 0.333039999 | 1.59% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `28672x16x4096-bf16-91c581677069aeb17efa`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
- Rank 2：warmup=0，timed=2；原始结果 ID `28672x16x4096-bf16-cb8c8401e74da8da4f02`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0015`。
- Rank 3：warmup=0，timed=2；原始结果 ID `28672x16x4096-bf16-e8f8cacd934b18fb3037`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
### 第二轮 FP16 · G364 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.334239990 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.334239990 | 11.2437 | 1.0000× | 0.334239990, 0.335040003 | 0.24% |
| 2 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.372880012 | 10.0786 | 0.8964× | 0.373679996, 0.372880012 | 0.21% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.401199996 | 9.3671 | 0.8331× | 0.402319998, 0.401199996 | 0.28% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `28672x16x4096-fp16-G364-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0014`。
- Rank 2：warmup=0，timed=2；原始结果 ID `28672x16x4096-fp16-G364-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0031`。
- Rank 3：warmup=0，timed=2；原始结果 ID `28672x16x4096-fp16-G364-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0136`。

<a id="mnk-40960-4608-7168"></a>
## MNK 40960 × 4608 × 7168

### 第一轮 BF16 · G365 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 47.956798553 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 43.983840942 | 61.5187 | 1.0903× | 43.983840942, 44.101760864 | 0.27% |
| 2 | [2943f23873d8087254e4](#r1-2943f23873d8087254e4) | wdra88 · 128/128/64 | fine | 44.435440063 | 60.8935 | 1.0792× | 44.435440063, 44.801921844 | 0.82% |
| 3 | [6489da0ccd4fcc840e07](#r1-6489da0ccd4fcc840e07) | wdra88 · 128/128/64 | fine | 44.443599701 | 60.8823 | 1.0790× | 44.463680267, 44.443599701 | 0.05% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `40960x4608x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
- Rank 2：warmup=0，timed=2；原始结果 ID `40960x4608x7168-bf16-2943f23873d8087254e4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
- Rank 3：warmup=0，timed=2；原始结果 ID `40960x4608x7168-bf16-6489da0ccd4fcc840e07`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0135`。
### 第二轮 FP16 · G365 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 36.566078186 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 36.566078186 | 73.9983 | 1.0000× | 36.632400513, 36.566078186 | 0.18% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 41.162960052 | 65.7346 | 0.8883× | 43.288158417, 41.162960052, 41.174800873 | 5.16% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 44.421600342 | 60.9125 | 0.8232× | 44.596401215, 44.421600342 | 0.39% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `40960x4608x7168-fp16-G365-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0176`。
- Rank 2：warmup=0，timed=3；原始结果 ID `40960x4608x7168-fp16-G365-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0080`。
- Rank 3：warmup=0，timed=2；原始结果 ID `40960x4608x7168-fp16-G365-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0064`。

<a id="mnk-40960-7168-2304"></a>
## MNK 40960 × 7168 × 2304

### 第一轮 BF16 · G366 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 23.769039154 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 17.795040131 | 76.0276 | 1.3357× | 17.827199936, 17.795040131 | 0.18% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 20.578720093 | 65.7434 | 1.1550× | 20.592960358, 20.578720093 | 0.07% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 23.688720703 | 57.1122 | 1.0034× | 23.776800156, 23.688720703 | 0.37% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `40960x7168x2304-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0136`。
- Rank 2：warmup=0，timed=2；原始结果 ID `40960x7168x2304-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0136`。
- Rank 3：warmup=0，timed=2；原始结果 ID `40960x7168x2304-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0136`。
### 第二轮 FP16 · G366 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 18.895280838 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 18.895280838 | 71.6007 | 1.0000× | 18.928880692, 18.895280838 | 0.18% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 21.681200027 | 62.4004 | 0.8715× | 21.717519760, 21.681200027 | 0.17% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 23.502000809 | 57.5659 | 0.8040× | 23.502000809, 23.580080032 | 0.33% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `40960x7168x2304-fp16-G366-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0176`。
- Rank 2：warmup=0，timed=2；原始结果 ID `40960x7168x2304-fp16-G366-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0080`。
- Rank 3：warmup=0，timed=2；原始结果 ID `40960x7168x2304-fp16-G366-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0120`。

<a id="mnk-53248-16-4096"></a>
## MNK 53248 × 16 × 4096

### 第一轮 BF16 · G367 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 0.609679997 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [834aa4489131137d480c](#r1-834aa4489131137d480c) | nowasp · 64/64/64 | fine | 0.593039989 | 11.7687 | 1.0281× | 0.595040023, 0.593039989 | 0.34% |
| 2 | [d6f728373b26c60d286e](#r1-d6f728373b26c60d286e) | wasp44 · 64/64/64 | fine | 0.598959982 | 11.6524 | 1.0179× | 0.601040006, 0.598959982 | 0.35% |
| 3 | [705b25ef7216fced2378](#r1-705b25ef7216fced2378) | wasp44 · 64/64/64 | coarse | 0.601599991 | 11.6013 | 1.0134× | 0.603439987, 0.601599991 | 0.31% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `53248x16x4096-bf16-834aa4489131137d480c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0136`。
- Rank 2：warmup=0，timed=2；原始结果 ID `53248x16x4096-bf16-d6f728373b26c60d286e`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0136`。
- Rank 3：warmup=0，timed=2；原始结果 ID `53248x16x4096-bf16-705b25ef7216fced2378`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0031`。
### 第二轮 FP16 · G367 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 0.615040004 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [b1b7a85dcfa6235f5b61](#r2-b1b7a85dcfa6235f5b61) | wasp44 · 64/64/64 | coarse | 0.600480020 | 11.6229 | 1.0242× | 0.610000014, 0.600480020 | 1.59% |
| 2 | [d225c662f70591fd1883](#r2-d225c662f70591fd1883) | nowasp · 64/64/64 | coarse | 0.615040004 | 11.3478 | 1.0000× | 0.615040004, 0.625760019 | 1.74% |
| 3 | [9dfd08ef3b758668b547](#r2-9dfd08ef3b758668b547) | nowasp · 64/64/128 | coarse | 0.641839981 | 10.8739 | 0.9582× | 0.641839981, 0.643599987 | 0.27% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `53248x16x4096-fp16-G367-b1b7a85dcfa6235f5b61`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0032`。
- Rank 2：warmup=0，timed=2；原始结果 ID `53248x16x4096-fp16-G367-d225c662f70591fd1883`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0014`。
- Rank 3：warmup=0，timed=2；原始结果 ID `53248x16x4096-fp16-G367-9dfd08ef3b758668b547`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0136`。

<a id="mnk-53248-4608-7168"></a>
## MNK 53248 × 4608 × 7168

### 第一轮 BF16 · G368 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 62.254081726 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [f2ef7cbc9d2c6af21cfc](#r1-f2ef7cbc9d2c6af21cfc) | wdra88 · 128/128/64 | fine | 57.020401001 | 61.6898 | 1.0918× | 57.020401001, 57.027919769 | 0.01% |
| 2 | [f9a975d65680b8ccd8fe](#r1-f9a975d65680b8ccd8fe) | wasp44 · 128/128/64 | fine | 57.797920227 | 60.8599 | 1.0771× | 58.753120422, 57.797920227 | 1.65% |
| 3 | [2943f23873d8087254e4](#r1-2943f23873d8087254e4) | wdra88 · 128/128/64 | fine | 57.882961273 | 60.7705 | 1.0755× | 58.135120392, 57.882961273 | 0.44% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `53248x4608x7168-bf16-f2ef7cbc9d2c6af21cfc`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
- Rank 2：warmup=0，timed=2；原始结果 ID `53248x4608x7168-bf16-f9a975d65680b8ccd8fe`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
- Rank 3：warmup=0，timed=2；原始结果 ID `53248x4608x7168-bf16-2943f23873d8087254e4`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
### 第二轮 FP16 · G368 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 47.429599762 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 47.429599762 | 74.1642 | 1.0000× | 47.487281799, 47.429599762 | 0.12% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 53.098320007 | 66.2465 | 0.8932× | 53.098320007, 53.415279388 | 0.60% |
| 3 | [9340e264cd6a3747cab8](#r2-9340e264cd6a3747cab8) | wdra88 · 128/128/64 | coarse | 58.042961121 | 60.6030 | 0.8171× | 58.148159027, 58.042961121 | 0.18% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `53248x4608x7168-fp16-G368-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0176`。
- Rank 2：warmup=0，timed=2；原始结果 ID `53248x4608x7168-fp16-G368-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0080`。
- Rank 3：warmup=0，timed=2；原始结果 ID `53248x4608x7168-fp16-G368-9340e264cd6a3747cab8`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0064`。

<a id="mnk-53248-7168-2304"></a>
## MNK 53248 × 7168 × 2304

### 第一轮 BF16 · G369 · 输出 BF16

已记录 17 项，剩余 0 项；最优粗调 nowasp 基线 30.972959518 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [7383db2bfccbcb99422c](#r1-7383db2bfccbcb99422c) | nowasp · 128/128/64 | fine | 23.103279114 | 76.1273 | 1.3406× | 23.103279114, 23.127840042 | 0.11% |
| 2 | [8dccb3b0aa43ebe6451d](#r1-8dccb3b0aa43ebe6451d) | nowasp · 128/128/64 | fine | 26.674400330 | 65.9355 | 1.1611× | 26.674400330, 26.695840836 | 0.08% |
| 3 | [5fac3898406d7bf33e42](#r1-5fac3898406d7bf33e42) | nowasp · 128/128/64 | fine | 30.835359573 | 57.0381 | 1.0045× | 30.835359573, 30.863840103 | 0.09% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `53248x7168x2304-bf16-7383db2bfccbcb99422c`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
- Rank 2：warmup=0，timed=2；原始结果 ID `53248x7168x2304-bf16-8dccb3b0aa43ebe6451d`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
- Rank 3：warmup=0，timed=2；原始结果 ID `53248x7168x2304-bf16-5fac3898406d7bf33e42`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-20261028/pack-campaign/batches/0137`。
### 第二轮 FP16 · G369 · 输出 FP16

已记录 24 项，剩余 8 项；最优粗调 nowasp 基线 24.526159286 ms。

| Rank | 配置 | mode · tile | 阶段 | ms | TFLOPS | 相对基线 | 实际样本 ms | 波动 |
|---|---|---|---|---:|---:|---:|---|---:|
| 1 | [95353eef5bd8e1ef4c4a](#r2-95353eef5bd8e1ef4c4a) | nowasp · 128/128/64 | coarse | 24.526159286 | 71.7107 | 1.0000× | 24.526159286, 24.539119720 | 0.05% |
| 2 | [9aa90eadea15fdca1279](#r2-9aa90eadea15fdca1279) | nowasp · 128/128/64 | coarse | 28.257999420 | 62.2404 | 0.8679× | 28.257999420, 28.269199371 | 0.04% |
| 3 | [3b8038774d3f627923b9](#r2-3b8038774d3f627923b9) | nowasp · 128/128/64 | coarse | 30.656560898 | 57.3707 | 0.8000× | 30.656560898, 30.720079422 | 0.21% |

逐项采样和追溯：

- Rank 1：warmup=0，timed=2；原始结果 ID `53248x7168x2304-fp16-G369-95353eef5bd8e1ef4c4a`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0176`。
- Rank 2：warmup=0，timed=2；原始结果 ID `53248x7168x2304-fp16-G369-9aa90eadea15fdca1279`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0080`。
- Rank 3：warmup=0，timed=2；原始结果 ID `53248x7168x2304-fp16-G369-3b8038774d3f627923b9`；远端日志 `/root/jingyang/opt/unittest_shaobo/logs/fp16-autotune-fp16-20261029/pack-campaign/batches/0120`。

## 完整配置字典

同一轮内配置 ID 对应原始 config 字典；HSACO SHA256 用于核对实际二进制。`effective_options` 保留导出记录中的有效选项，未出现的字段不擅自填入假定默认值。evidence 包含 WASP/WDRA、BPP/async/MLS、LDS 和寄存器分区证据；runtime_signature/constants 用于核对 ABI。

<a id="r1-04f5799b1bd3f13ca9ca"></a>
### r1-04f5799b1bd3f13ca9ca

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "3e8a5f02b90331e25e0f705d00636614b0aec641ce438b4ec66d37dda16d5865",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-05c5c97400044a95f5e6"></a>
### r1-05c5c97400044a95f5e6

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "57e66f3cec9a59d36a7cb80fbc7fc27f4d756ad1d532d5f43e566c342674b3df",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  56",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-05df19f7638b5be06b68"></a>
### r1-05df19f7638b5be06b68

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "2c88b15f05353e2e5a3497c74a118e513eed350702dfe8feb2d9a68d593e173b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-06df5002ca9f92ab59cd"></a>
### r1-06df5002ca9f92ab59cd

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "db441ce6aa3155d2c8768af17f7ffbb784ba2f63746a9580358e8a1da2e87873",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-0eccce44229d587e140d"></a>
### r1-0eccce44229d587e140d

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "3e1824631e6b0e040e467016645d1b3617a43d6b1c4fa67e5a7e8b1b8dc733f1",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 114688,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 114688,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-0f7a5be9b0919f6d6b2e"></a>
### r1-0f7a5be9b0919f6d6b2e

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "6e92a9cd206dcbe867314c8efa5a3ab3e3d3008a6c1b61754cc56040a84977c1",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-17b66ad620da24d6f82b"></a>
### r1-17b66ad620da24d6f82b

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "f1ac8d43dc9c9f6d69725e10f45ae35f9becf0e55d40ec4137182aaa674ac33b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-247c182993fcb65dda68"></a>
### r1-247c182993fcb65dda68

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "ebd3836d921b56fcee3865415eba7268ec663e1a7402e4e11f319db702525dbf",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-249d0e074b4086e8aa18"></a>
### r1-249d0e074b4086e8aa18

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "bca76ea4b7a0466269a697abda68869b7bba00b7cc49202d69dee943d54cd001",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-2943f23873d8087254e4"></a>
### r1-2943f23873d8087254e4

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        64,
        64,
        192,
        192
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      64,
      64,
      192,
      192
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "a946c3e553c73a999c2b2ed53ba9b366b79904149289962465e81425c3e014eb",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-32d663c1fff0030fd159"></a>
### r1-32d663c1fff0030fd159

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "2f15df88bcbd587d2e67fa86a91cac84109c33c2510a5ac099aa6c946614525b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-3349c6b9c585549f7448"></a>
### r1-3349c6b9c585549f7448

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        88,
        208,
        208
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      88,
      208,
      208
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d6bf6c58913e0fb8550e953dc11fd6e5acd7e4fa550b76ef3febe226bf2ae1bd",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-3717d3df609be657fa32"></a>
### r1-3717d3df609be657fa32

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "864efa05fed2624ea4dd91aca48759c5f7f04211706a0e6a0c838f3428ed283c",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-37d075650464a6ce6f43"></a>
### r1-37d075650464a6ce6f43

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "ac6dfd69271c048d49cf622bb4203caca3b970ff1d653fcc29efdbe3b7484f86",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-39eb19a1c9076aa7945a"></a>
### r1-39eb19a1c9076aa7945a

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "6e92a9cd206dcbe867314c8efa5a3ab3e3d3008a6c1b61754cc56040a84977c1",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-3b67ff7fe683b39aa55a"></a>
### r1-3b67ff7fe683b39aa55a

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d3548b2315dd9fe78f01377a5489e7d8b35fb3fa7958f386581378cb8b56dcfd",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-3ed295572d5209a7ecdd"></a>
### r1-3ed295572d5209a7ecdd

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "c44436535640fd38b917c72be54367dab937796c87b9ca22866bda65531bc0db",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-3f9697ffe28602a898a4"></a>
### r1-3f9697ffe28602a898a4

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "36201b9a5ae9676fa3571b01e2e0f2803ad5f56e6e18ab08e73fed4e970f7050",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-455ec4bcfd3552d92371"></a>
### r1-455ec4bcfd3552d92371

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d6bf6c58913e0fb8550e953dc11fd6e5acd7e4fa550b76ef3febe226bf2ae1bd",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-46a1fbc6df36ffed1a49"></a>
### r1-46a1fbc6df36ffed1a49

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        88,
        208,
        208
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      88,
      208,
      208
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "955dc6b3db04995e52bdf3b824142e69dcc89407a7085b6a407aefa93d7d1b59",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-4906f7e92d0963838f87"></a>
### r1-4906f7e92d0963838f87

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        88,
        208,
        208
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      88,
      208,
      208
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "b2467c9a7ceb3d3e9c4f12bf7274851594b79ddf2321c54b2640a1d9ab757d53",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-498436a13632e968d1a9"></a>
### r1-498436a13632e968d1a9

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "4ed9d720af6c4eb3bd781c4763ac053315a709cee4d77c9ab52e45faecf6d862",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-4c7c4f499206cb9b1221"></a>
### r1-4c7c4f499206cb9b1221

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "0c96d485a786c8bc9cbdb2014e38de95ad031e9eea29aed4e0eadf325f259d52",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  56",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-4d692db862aea00189bd"></a>
### r1-4d692db862aea00189bd

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "4a19cd8bbb01f1853983d8cd432bcccf48e811dbefe85a7ea1ee29c498919f36",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-51908aa291df6bf1d852"></a>
### r1-51908aa291df6bf1d852

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 8,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "fc3699c53047f8caeb101ea705e9af2d3b89b9212100435a16260952a1423ff2",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-57278aa2e4dbb73c63c3"></a>
### r1-57278aa2e4dbb73c63c3

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 4,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 4,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "19c1e660e42b3a9c2a18ba24fb9e4dd4033bed1ee34c6fed0c73848e41d65809",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 49152,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 49152,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-5e627b894b84cb1bc438"></a>
### r1-5e627b894b84cb1bc438

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "bca76ea4b7a0466269a697abda68869b7bba00b7cc49202d69dee943d54cd001",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-5fac3898406d7bf33e42"></a>
### r1-5fac3898406d7bf33e42

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "5716ff43d8248a298b1d08ae6db5dde77bdab981fd14a3b1e87dc58989bba6d8",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-623dfda53d317222ceaf"></a>
### r1-623dfda53d317222ceaf

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "93fe0e41c796718d460c7571af4e0e77f509fc1996992c7a26601b3dc0096f65",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-6489da0ccd4fcc840e07"></a>
### r1-6489da0ccd4fcc840e07

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "aa5fa023ea57165902793670414ffc9c66bc9b282b3b30c21578805ad8782f4d",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-64ee2c9768bc4fbee283"></a>
### r1-64ee2c9768bc4fbee283

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "c4694fdae6d2a1c04f0d28b84953055093fe4b500df898a7133132a543df42d4",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 114688,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 114688,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-6bffccea3bac78208f44"></a>
### r1-6bffccea3bac78208f44

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "0decf15a80fa1916f398e6bc3dce876a641dc9ffc47619c25f860471b539bb68",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-705b25ef7216fced2378"></a>
### r1-705b25ef7216fced2378

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "864efa05fed2624ea4dd91aca48759c5f7f04211706a0e6a0c838f3428ed283c",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-7383db2bfccbcb99422c"></a>
### r1-7383db2bfccbcb99422c

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "b3fdf7786e811dc589dcc89b5943635babaeefe02096fcbdf6c56dbd5f3eb35d",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-7539c91fdab289ff899b"></a>
### r1-7539c91fdab289ff899b

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "b2467c9a7ceb3d3e9c4f12bf7274851594b79ddf2321c54b2640a1d9ab757d53",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-7559008785dcec902504"></a>
### r1-7559008785dcec902504

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "9fc509dd19727372874042088d60151929de2aa4c9dbaa6d2b9412ff6a506f68",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-834aa4489131137d480c"></a>
### r1-834aa4489131137d480c

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "f795cb0f5b4009886474893e50fca2f5124db969ef89e2304972f317790fdcd1",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-880a05c17a0b9653b61c"></a>
### r1-880a05c17a0b9653b61c

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "73f7ad7ddd5e68fd57a3d53eb73ee6a8566adedcce9e2c5da526a4aebd33f9f9",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 114688,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 114688,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-8dccb3b0aa43ebe6451d"></a>
### r1-8dccb3b0aa43ebe6451d

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "45ec5535e4f4f7710f4ef2dc50f11f18cea6402045e5ec3460675a3566595526",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-8f86441f7995d544ff0e"></a>
### r1-8f86441f7995d544ff0e

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d7cdc884485cd9ca3e52bd8348f89c38639c3519a7e92520adce86268c4ad362",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-8fb190809a17bfcaff1a"></a>
### r1-8fb190809a17bfcaff1a

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "6794fb7eedce0c2e9860df15b0d0f75e715ea39665fd7cde90b3f6d6dae4357a",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-91c581677069aeb17efa"></a>
### r1-91c581677069aeb17efa

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "75f17200ef7a399488f37830bc72ba0cf72df9faaa7f87b85d0ee82f33f696fa",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-93e7f64a414acfefdf5a"></a>
### r1-93e7f64a414acfefdf5a

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "8e993603ffff4ba852c6c3d3a21a31216b6ff14a0882c0a9d91f56d05c93f9e4",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-9dbcc1f02010f1b928b8"></a>
### r1-9dbcc1f02010f1b928b8

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "b7b439c599a0db19cdedcf8e1f5d7faab176e1c8c2d69d64e2e9b889bedf00c7",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-9fbc98f298852c6db1bc"></a>
### r1-9fbc98f298852c6db1bc

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "26431202972a270f11be5de1ee01eb47949cf9fb1c399869fe1b4744036762ff",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-a1f7c7d7b8b31636bb20"></a>
### r1-a1f7c7d7b8b31636bb20

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "93fe0e41c796718d460c7571af4e0e77f509fc1996992c7a26601b3dc0096f65",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-a2ec7ee991efd60bd336"></a>
### r1-a2ec7ee991efd60bd336

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "ac6dfd69271c048d49cf622bb4203caca3b970ff1d653fcc29efdbe3b7484f86",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-a57bcf6a22eedf684051"></a>
### r1-a57bcf6a22eedf684051

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "aa5fa023ea57165902793670414ffc9c66bc9b282b3b30c21578805ad8782f4d",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-a8db5435a105c79a8cd1"></a>
### r1-a8db5435a105c79a8cd1

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "51696e64116cd285a5053f09ad4d6eadc226807a42a1d44516e295af7056d8af",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-ab18f5bc5035fd03b9d3"></a>
### r1-ab18f5bc5035fd03b9d3

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "4cfe162b0274ad64c2c585cf0c8114bc2f34d55602a0fb4855c4f2127f303ee5",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-b0fb3701e2e7d92ec37f"></a>
### r1-b0fb3701e2e7d92ec37f

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "3e1824631e6b0e040e467016645d1b3617a43d6b1c4fa67e5a7e8b1b8dc733f1",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 114688,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 114688,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-b3864ea759acb2e2470a"></a>
### r1-b3864ea759acb2e2470a

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "5716ff43d8248a298b1d08ae6db5dde77bdab981fd14a3b1e87dc58989bba6d8",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-b8391187ee0536d6c14a"></a>
### r1-b8391187ee0536d6c14a

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 8,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "7a7107171a2e43583de73514ef489c557be09dc9b92796a4efd83f38b11fbf8d",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-b8657961b1124e881dd5"></a>
### r1-b8657961b1124e881dd5

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        64,
        64,
        192,
        192
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      64,
      64,
      192,
      192
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "fa1bc94ba0ad5678b7d97b2d8c8fed598bffb8b2ba56f96984237a7b958fd45b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-be5aa35e5bf6b2f23e8e"></a>
### r1-be5aa35e5bf6b2f23e8e

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "eb17a83dc1d55836261e5b1b9d3fafaf5a94785c6a64c38b2855b1505a16ccd3",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-c0cddd92b5bb88b9f96e"></a>
### r1-c0cddd92b5bb88b9f96e

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "4ab04e5be2ebb2274ee9dfcb99889ed1da0d92066d13a4a42fd39aa50ddd1cdd",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-cb8c8401e74da8da4f02"></a>
### r1-cb8c8401e74da8da4f02

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "4bf3db7faf18228ffdc2ca2a9d227960c07782ccf3af65def55c5126593d3968",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-d6f728373b26c60d286e"></a>
### r1-d6f728373b26c60d286e

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "75a588348a243286cae5188097efe02a01ff322572603a2403087a4cd40dfe92",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-dbddcf203270db2ec3f7"></a>
### r1-dbddcf203270db2ec3f7

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        64,
        64,
        192,
        192
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      64,
      64,
      192,
      192
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "9a3fc54e17c04abfdee5d777a6c7b384bc9191fe6cdd1ee6500071b529d919fd",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-e8f8cacd934b18fb3037"></a>
### r1-e8f8cacd934b18fb3037

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0,
      "num_warps": 4
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "4bf3db7faf18228ffdc2ca2a9d227960c07782ccf3af65def55c5126593d3968",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r1-e9119b3c019fe173f6ed"></a>
### r1-e9119b3c019fe173f6ed

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "7fae740fcf2155196c554294acfdbc8a3551fe3c63b517660ff41e5517d46937",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-eee8f4403014e9f7aacb"></a>
### r1-eee8f4403014e9f7aacb

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "31ab2d8a395ced6bd8f178e003424d06744027e1ef81ad26094d6367b172d184",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r1-f1a574149008cfa85abd"></a>
### r1-f1a574149008cfa85abd

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "0c96d485a786c8bc9cbdb2014e38de95ad031e9eea29aed4e0eadf325f259d52",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  56",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-f2ef7cbc9d2c6af21cfc"></a>
### r1-f2ef7cbc9d2c6af21cfc

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d6a967e3f60cb9eaf07075ce73c1e1cb040dde919bda316022933f2df3a61c0b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-f7627acc64f8279814ee"></a>
### r1-f7627acc64f8279814ee

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp32",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "074bb9410e0c964961ce7fc7ea71d4a633b93a138bd9b9bdd5690665bd3d5740",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "fp32",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*fp32",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r1-f914f5100c7c0a6581fe"></a>
### r1-f914f5100c7c0a6581fe

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "b2467c9a7ceb3d3e9c4f12bf7274851594b79ddf2321c54b2640a1d9ab757d53",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r1-f9a975d65680b8ccd8fe"></a>
### r1-f9a975d65680b8ccd8fe

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "bf16",
    "input_dtype": "bf16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "eeec41219d0e78a649d0b8c5ccd31417b788b371ce284c3c03c79ca2a5ade9b0",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [
      "v_mmac_f32_16x16x16_bf16"
    ],
    "input_signature": "bf16",
    "output_signature": "bf16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "c_ptr": "*bf16",
    "bias_ptr": "*bf16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-09b0e5995723b60e5a09"></a>
### r2-09b0e5995723b60e5a09

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "439479dc0b60f1f3e509d5036940961dd89714a1b40179336890113a86b0197a",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 98304,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 98304,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-2102b4428dc21ca7e8dc"></a>
### r2-2102b4428dc21ca7e8dc

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        88,
        208,
        208
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      88,
      208,
      208
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d1a45fba4bbcee74e53f05f2f5ecacf50db9000c09761db8575d6d5b794dd19f",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 98304,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 98304,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-22228385a301e0ff14c5"></a>
### r2-22228385a301e0ff14c5

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 8,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "82b0e488b22bb5db2d0936a26adba108b2bcffa45497093e7809b20b93c3de66",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r2-3b63491ed6531d89a40d"></a>
### r2-3b63491ed6531d89a40d

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "8e4563fb7b28a14d21e6b57edb14c51879ed461b9e0ee95113c9f16195880681",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 106496,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 106496,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r2-3b8038774d3f627923b9"></a>
### r2-3b8038774d3f627923b9

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "8e61655b605c48cf3a299c35194ff73fb633a3f873ebc5aa34e5c0cb74257f8b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-3cf95baae0ff329949b1"></a>
### r2-3cf95baae0ff329949b1

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "b4a78cd088cc335b7898301f2831aca942e9c32d823824f061d427c582491d3d",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-3d4c2441dd5696bafd6f"></a>
### r2-3d4c2441dd5696bafd6f

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 4,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "8796d5acff1db4faf2a0ec6ef751819429534b903343806f8c93a970bb079bb3",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 4,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r2-467010bd09efdd1d21de"></a>
### r2-467010bd09efdd1d21de

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        160,
        160,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      160,
      160,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "38c41f362c13d03f68146b5af6cfbc1ac9f268c92831ee59f231614e930fa953",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 124",
      "s_set_vgpr_size 160",
      "s_set_vgpr_size 156"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 124",
      "s_set_vgpr_size 160",
      "s_set_vgpr_size 156"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 124",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 160",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 156",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r2-484d307f1978ef3f454e"></a>
### r2-484d307f1978ef3f454e

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 1,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 8
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 1,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 8,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "1afce2c8f9c75b9379d5c6f6b7d2418a017caa6c06db26a6359558ec9a633301",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-49eeff2882f3db7855ce"></a>
### r2-49eeff2882f3db7855ce

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "d1a45fba4bbcee74e53f05f2f5ecacf50db9000c09761db8575d6d5b794dd19f",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 98304,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 98304,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-768b997665c94dc584f7"></a>
### r2-768b997665c94dc584f7

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 16,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 8
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 8,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "0e4d568d1e0b7914fa579d8a96d4a575fdf03befcf7199fd0f21f24d79686d16",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 18432,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 18432,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-8196eabaa94d5c892734"></a>
### r2-8196eabaa94d5c892734

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": true,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": true
  },
  "sha256": "14e4fcdcc4936296f3675c608211c299982e871dd80a5e36547e028ae80eb39f",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": true,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 16",
      "s_set_vgpr_size 244",
      "s_set_vgpr_size 244"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 73728,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 16",
      "s_set_vgpr_size 244",
      "s_set_vgpr_size 244"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  16",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 244",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 244",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr",
    "USE_MATRIX_STORE": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1,
    "USE_MATRIX_STORE": false
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 73728,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-9340e264cd6a3747cab8"></a>
### r2-9340e264cd6a3747cab8

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        16,
        244,
        160
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      16,
      244,
      160
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "1663196e004569984bbe29527fe9f1f6e8996a05230fd5fe846d12ac06e9249f",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 72",
      "s_set_vgpr_size 188",
      "s_set_vgpr_size 188"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  72",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 188",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

<a id="r2-95353eef5bd8e1ef4c4a"></a>
### r2-95353eef5bd8e1ef4c4a

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 8
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 8,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "25c27c5d0132b996c4a4cd9ed2926f9db2508ac8ba7b490b82ceb80612e1550c",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-9aa90eadea15fdca1279"></a>
### r2-9aa90eadea15fdca1279

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 3,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 3,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "70caf5ea230c739bdfcecb8c5a93a82dc02bffaf1e8029897becf21e7f2b6af8",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 65536,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 65536,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-9c257c155867db0a649b"></a>
### r2-9c257c155867db0a649b

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": 0,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": 0,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "dac665a2cff067d45ae109e166327d86591832e322c50ffd5a54e7baea26bf27",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-9dfd08ef3b758668b547"></a>
### r2-9dfd08ef3b758668b547

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 128,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "dac665a2cff067d45ae109e166327d86591832e322c50ffd5a54e7baea26bf27",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-a184436d067a2e030558"></a>
### r2-a184436d067a2e030558

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        2,
        4
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      2,
      4
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "2e64685b2f8a9eaecfda67c6eee775324c80ca6c9b5d08582eb79860603428c1",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 81920,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 81920,
  "block": [
    384,
    1,
    1
  ]
}
```

<a id="r2-b1b7a85dcfa6235f5b61"></a>
### r2-b1b7a85dcfa6235f5b61

```json
{
  "config": {
    "mode": "wasp44",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "687c93f794dde06a18df74d09f56be0b20da2d5c409899e9f07d7a7bdc0cfc0a",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-b51d8b6af4744a99ab8e"></a>
### r2-b51d8b6af4744a99ab8e

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 8,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "fbc4d5a25dc0452fe191a26d5a224476552f3510fbd27dda33099e9b37c9afda",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 98304,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 8,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 98304,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-bdcb5554a17204c8e26a"></a>
### r2-bdcb5554a17204c8e26a

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 16,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "8b76eda68b97426264eab685931febd24bbf13b4b3a0b120c36a53a21c919a2a",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 10240,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 10240,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-cd3eb736d84b1dffa424"></a>
### r2-cd3eb736d84b1dffa424

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "f4c5db3c55beaa6f682e4ff3ba3b3ca68d01e0cf626c656f72614032c1b05f27",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 98304,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 88",
      "s_set_vgpr_size 208",
      "s_set_vgpr_size 208"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  88",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 208",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 98304,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-d225c662f70591fd1883"></a>
### r2-d225c662f70591fd1883

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "f78062c42b9c120c37c546b136f3ec457d4ce29a31e3a4350eec8cda1adde739",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 16384,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 16384,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-d8809ef3e165e6edd726"></a>
### r2-d8809ef3e165e6edd726

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 3,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 3,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "56e9a07c007aef0fc70211dbdbcd6f9e5a904bfd7add5ba13dd341f964cac97a",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-ddb93ae85b5fc1cfdac0"></a>
### r2-ddb93ae85b5fc1cfdac0

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 4
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 4,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "ec8f1d0f5260551a999604a08413d437bdcc8f586664a2c5e2287c515cfc37be",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "probe lowering failed: ttg.async_copy_global_to_local illegal; excluded"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    256,
    1,
    1
  ]
}
```

<a id="r2-e2469f47dd001539f214"></a>
### r2-e2469f47dd001539f214

```json
{
  "config": {
    "mode": "wdra48",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        16,
        244,
        244
      ],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      16,
      244,
      244
    ],
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "b772c04edf654167d7a53fcd58a76cc4c8c69f6a12aceaefa626973a934359aa",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 40960,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 56",
      "s_set_vgpr_size 224",
      "s_set_vgpr_size 224"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  56",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 224",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 40960,
  "block": [
    768,
    1,
    1
  ]
}
```

<a id="r2-f89f00abebc2a438c6a0"></a>
### r2-f89f00abebc2a438c6a0

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 8,
      "use_block_pingpong": true
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 8,
    "use_block_pingpong": true,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "25c27c5d0132b996c4a4cd9ed2926f9db2508ac8ba7b490b82ceb80612e1550c",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-f8dc5db3e2c95b32a898"></a>
### r2-f8dc5db3e2c95b32a898

```json
{
  "config": {
    "mode": "nowasp",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": false,
      "wasp_wdra": false,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": true,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1,
      "num_warps": 8
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "wasp_wdra": false,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": true,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "num_warps": 8,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "wasp_partition_warps": null,
    "wasp_partition_regs": null,
    "use_mls": false
  },
  "sha256": "d1f7aa9c256179d5ed25e74ad9eeae30d8f747ceaf324165f535939d34bc283b",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": false,
    "wdra": false,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [],
    "private_segment_fixed_size": 0,
    "shared_bytes": 32768,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [],
    "wdra_instructions": [],
    "wdra_verified": false,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": false,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 32768,
  "block": [
    512,
    1,
    1
  ]
}
```

<a id="r2-fc1f34eeb0619026b2a6"></a>
### r2-fc1f34eeb0619026b2a6

```json
{
  "config": {
    "mode": "wdra88",
    "launch": {
      "GROUP_SIZE_M": 1,
      "WARP_SPECIALIZE": true,
      "wasp_wdra": true,
      "amdgpu_enable_max_ilp_scheduling_strategy": true,
      "hcu_use_gfx946_sched_model_v2": false,
      "wasp_partition_warps": [
        4,
        4,
        4,
        4
      ],
      "wasp_partition_regs": [
        32,
        32,
        224,
        192
      ],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "empty_arrive_after_mmac": false,
      "enable_v_mmac_cluster": 0,
      "num_stages": 2,
      "mmac_layout_force": -1
    },
    "out_dtype": "fp16",
    "input_dtype": "fp16",
    "use_mls": false,
    "abi": "row-row-unit-inner-strides-div16-v2"
  },
  "effective_options": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "wasp_wdra": true,
    "amdgpu_enable_max_ilp_scheduling_strategy": true,
    "hcu_use_gfx946_sched_model_v2": false,
    "wasp_partition_warps": [
      4,
      4,
      4,
      4
    ],
    "wasp_partition_regs": [
      32,
      32,
      224,
      192
    ],
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "empty_arrive_after_mmac": false,
    "enable_v_mmac_cluster": 0,
    "num_stages": 2,
    "mmac_layout_force": -1,
    "use_block_pingpong": false,
    "use_async_copy": false,
    "block_pingpong_async_global_prefetch": null,
    "block_pingpong_async_local_prefetch": 1,
    "use_mls": false
  },
  "sha256": "467c3bf61f37f7d611590e9810bb53a50b6616bdbf0a53a848c3dadeafc8f935",
  "private_segment_fixed_size": 0,
  "evidence": {
    "wasp": true,
    "wdra": true,
    "async_ir": false,
    "async_isa": false,
    "mls": false,
    "bpp_applied_ir": false,
    "bpp_lds_buffers_ir": false,
    "bpp": "not verified active; option alone is not proof",
    "actual_vgpr_partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "private_segment_fixed_size": 0,
    "shared_bytes": 131072,
    "bf16_mmac_instructions": [],
    "input_signature": "fp16",
    "output_signature": "fp16",
    "partition_instructions": [
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 64",
      "s_set_vgpr_size 192",
      "s_set_vgpr_size 192"
    ],
    "wdra_instructions": [
      "; wdra init",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr =  64",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder",
      "; wdra branch body with #vgpr = 192",
      "; wdra placeholder"
    ],
    "wdra_verified": true,
    "bpp_verified": false,
    "async_verified": false,
    "bpp_note": "switch accepted; applied IR/assembly evidence not established",
    "async_note": "not verified active; consult per-configuration compiler result; no global support claim"
  },
  "compiler_provenance": {
    "path": "/xukang/code/pmd-triton-sessions/mmac-descriptor-20261020/python/triton/backends/hcu/compiler.py",
    "sha256": "d0d34b8ff21ae334074fe37a4e6eaf17410ca476831c4e8a3e90f73f29717e39",
    "triton_version": "3.6.0"
  },
  "runtime_signature": {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "bias_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "constexpr",
    "stride_bk": "i32",
    "stride_bn": "constexpr",
    "stride_cm": "i32",
    "stride_cn": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "WARP_SPECIALIZE": "constexpr",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "HAS_BIAS": "constexpr"
  },
  "constants": {
    "GROUP_SIZE_M": 1,
    "WARP_SPECIALIZE": true,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "HAS_BIAS": false,
    "stride_ak": 1,
    "stride_bn": 1,
    "stride_cn": 1
  },
  "arg_names": [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "bias_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_bk",
    "stride_cm",
    "global_scratch_null",
    "profile_scratch_null"
  ],
  "shared_bytes": 131072,
  "block": [
    1024,
    1,
    1
  ]
}
```

## 归档来源与复核

第一轮源包：`/mnt/c/Users/xukang/Downloads/fp16-autotune-20261028-results.tar.gz`。

第二轮源包：`/mnt/c/Users/xukang/Downloads/fp16-autotune-fp16-20261029-stopped-20261030.tar.gz`。

逐 MNK 排名来自各包根目录 top3.json；完成数和阶段来自 pack-campaign/records.jsonl；机台信息来自归档内 service.py、start-pack.sh、console.log、EXPERIMENT.md、CAMPAIGN.md、compiled/*/result.json 和 Top3 clocks。本文只描述历史归档，不表示当前远端状态。

核对结果：153 个唯一 MNK、320 组“轮次/需求”、960 条 Top3、96 个完整配置条目；逐需求排名和记录数已与原始 JSON 核对。
