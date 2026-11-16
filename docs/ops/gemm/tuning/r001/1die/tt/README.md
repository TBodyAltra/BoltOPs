# R001 BF16 TT — WASP 1P+1C，三槽位

MNK=64×64×2048，BM16/BN32/BK128，共8个TG，每TG一个搬运wave和一个计算wave。三槽位共72KiB LDS，每槽放2个BK128的A/B；完整K、FP32累加、BF16输出，无Split-K、额外归约、WDRA或汇编补丁。

公开接口仍为 `gemm_a16w16(x, w, out=out, backend="auto")`：x[M,K]的stride=(2048,1)，w[N,K]的stride=(1,64)，即逻辑B[K,N]连续。kernel直接使用物理指针及固定TT stride，公开接口不复制权重。配置仅匹配gfx946/48CU、BF16输入输出、此完整MNK；显式配置的错误布局/精度/shape会被拒绝。

公开接口PMD实测：完整dispatch **5.832 µs / 2.88 TOPS**，全grid shader窗口 **2.219 µs**。9次采样，去掉首次后取8次中位数。26项TN/TT配置、分发、数值和错误TT contract测试通过。随机/全1/随机、4096个输出、输出guard及全部8个TG完成检查通过；SGPR/VGPR spill均为0。容差abs0.001+rel0.01，不是bitwise声明或实机验收。

- `record.json`：精度/stride、配置、源代码与汇编哈希、环境、原始采样及校验结果。
- `kernel.amdgcn`：该版本直接编译生成的汇编。
- `full_grid.perf`：全部8个TG、最后一次warm采集。
- `single_tg.perf`：单个16×32输出tile、完整K及原始N=64输出stride；独立采集，不作为全grid性能数据。

每槽包含2×A[16,128]（8KiB）和2×B[128,32]（16KiB），共24KiB。三槽的EMPTY/READY phase由Gluon显式管理；Compiler num_stages=2不代表这里只有两槽。Consumer按实际寄存器依赖等待，在释放槽位前等待本槽所有DS完成，再与producer填充重叠执行最后一个BK128的MMAC。

完整dispatch包含PMD命令处理/分发开销；shader窗口从第一个wave开始到最后一个wave结束。GC/shader周期455ps。单次perf时间和9次采样中位数分别记录；全grid/单TG的汇编和`.text`已核对一致，ELF调试信息可能不同。

环境：node60 / kxu_pmd_8569_fp8_20261017，PMD CoreArch HEAD_1703、gfx946、1die48CU。Triton release/3.6.x base6b111ee3 + MR565，LLVM-triton含MR21，DTKClang6.3.3@3599a58d7；完整路径、版本与二进制哈希见record.json。

复现（在记录的PMD环境中）：

```bash
python docs/ops/gemm/tuning/r001/1die/tt/benchmark.py --record docs/ops/gemm/tuning/r001/1die/tt/record.json --output /tmp/r001-tt-full
python docs/ops/gemm/tuning/r001/1die/tt/benchmark.py --record docs/ops/gemm/tuning/r001/1die/tt/record.json --output /tmp/r001-tt-perf --perf
python docs/ops/gemm/tuning/r001/1die/tt/benchmark.py --record docs/ops/gemm/tuning/r001/1die/tt/record.json --output /tmp/r001-tt-single --single-tg-perf
```

回放脚本设置记录的进程环境，使用独立cache，拒绝已存在的output目录。测量使用实际公开接口，并检查它选择的编译handle。未修改R001的TN调优结果。
