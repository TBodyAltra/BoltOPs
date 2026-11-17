# R009：525.59 与 511.27 TOPS 版本对照

| 项目 | 525.59 TOPS 版本 | test.patch 版本 |
|---|---|---|
| LLVM | c3cca66 + ds_read2 + 禁止 C 构造下沉诊断补丁 | dtk-llvm-7.2.1@264542e7e3 + test.patch |
| Triton | release/3.6.x@f8cbbee1 | 相同 |
| 完整 Kernel | 183.863680 μs / 404096 cycles | 189.014280 μs / 415416 cycles |
| 完整数值及边界检查 | PASS | PASS |
| Consumer VGPR / LDS | 235 / 114944 B | 相同 |
| MMAC / matrix store / v_perm | 384 / 8 / 0 | 相同 |
| ds_read2_b32 / ds_read_b64 / ds_read_matrix_trans_format | 48 / 6 / 96 | 相同 |
| packed MUL / FMA / BF16 CVT | 192 / 192 / 64 | 相同 |
| v_mov_b32（包含 e32） | 192 | 200 |
| v_mov_b64_e32 | 19 | 18 |
| .text 字节数 | 21744 | 21772 |

共同环境为 1 DIE /48 CU、GC455ps，完整 2560×9216×2048，48 TG×15 tiles，一次完整发射，seed21。性能数据不是单 TG PERF 的 duration。新版本相对上一版本耗时增加2.8013%，超过1%阈值，判定 REGRESSION。

两版最终汇编中，96条以零C初始化的MMAC均直接使用v[0:3]，无需逐处重新构造零C。test.patch修正TwoAddress子寄存器COPY分析和HCU MMOP early-clobber活跃区间，未禁止CGP下沉优化本身。

逐条提取指令并忽略物理寄存器编号后，全文件的指令序列差异仅落在两个consumer的尾部：最后一组MMAC结束后、最终scale运算和matrix store前的寄存器搬运区。新版增加8条MOV32、减少1条MOV64，净增加7条静态指令，代码增大28字节；其余部分存在寄存器分配差异。A/B producer、主循环MMAC/FMA、scale读取及barrier的操作码序列保持一致，没有新增scale等待或改变预取批数。

因此目前能确认的是寄存器分配、尾部搬运和后续代码位置发生变化，不能仅凭7条额外静态指令断言其直接造成全部11320周期的回退。SQCI非重复miss仅289→292，重复miss为73791→71421；这些聚合统计也不能证明特定指令供给瓶颈。525.59 TOPS版本未抓取PERF，尚不能对两版逐周期等待分布做对照。

后续若继续定位，应分别控制尾部搬运、指令布局和代码加载地址，而不是再次把回退归为“零C没有共享”。本次附上的single_tg.perf对应test.patch版本，其.text/.rodata及资源metadata与完整性能测试一致。
