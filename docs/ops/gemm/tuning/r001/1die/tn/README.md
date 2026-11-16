# R001 BF16 TN — WASP 1P+1C，两槽位

MNK=64×64×2048，BM16/BN16/BK128，共16个TG，每TG一个搬运wave和一个计算wave。两槽位共64KiB LDS，每槽放4个BK128的A/B；完整K、FP32累加、BF16输出，无Split-K、额外归约、WDRA或汇编补丁。

公开接口 `gemm_a16w16(a, w, out=out, backend="auto")` 使用提交的Gluon配置。输入a[M,K]、w[N,K]均K连续；kernel内部按逻辑B转置读取。

本次PMD公开接口实测：完整dispatch **5.412 µs / 3.10 TOPS**，全grid shader窗口 **1.800 µs**。9次采样，去掉首次后取8次中位数；20项R001接口/配置/数值测试通过。随机/全1/随机、4096个输出、输出guard及全部16个TG完成检查通过；容差abs0.001+rel0.01，不是bitwise声明或实机验收。

- `record.json`：配置、环境、源代码/汇编/原生代码哈希、原始采样及校验结果。
- `kernel.amdgcn`：该版本直接编译生成的汇编。
- `full_grid.perf`：完整16TG、最后一次warm采集。
- `single_tg.perf`：单个16×16 tile，完整K；独立采集，不作为全grid性能数据。

完整dispatch包含PMD命令处理/分发开销；shader窗口从第一个wave开始到最后一个wave结束。PMD cp_latency=7898cycles，GC/shader周期455ps，约3.61µs差额主要是模拟分发开销。单次perf时间与9次中位数分别记录。各次重编译ELF调试信息可能不同，三次采集的汇编和`.text`哈希均已核对一致。

环境：node60 / kxu_pmd_8569_fp8_20261017，PMD CoreArch HEAD_1703、gfx946、1die48CU。Triton release/3.6.x base6b111ee3 + MR565，LLVM-triton含MR21，DTKClang6.3.3@3599a58d7；完整路径、版本和二进制哈希见record.json。

复现（在记录的PMD环境中）：

```bash
python docs/ops/gemm/tuning/r001/1die/tn/benchmark.py --record docs/ops/gemm/tuning/r001/1die/tn/record.json --output /tmp/r001-wasp11-full
python docs/ops/gemm/tuning/r001/1die/tn/benchmark.py --record docs/ops/gemm/tuning/r001/1die/tn/record.json --output /tmp/r001-wasp11-perf --perf
python docs/ops/gemm/tuning/r001/1die/tn/benchmark.py --record docs/ops/gemm/tuning/r001/1die/tn/record.json --output /tmp/r001-wasp11-single --single-tg-perf
```

回放脚本从record设置进程环境，并使用独立cache；output目录需不存在。更多槽位及手改汇编实验不属于本条记录。
