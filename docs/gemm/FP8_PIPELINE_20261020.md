# FP8×FP8 流水候选与阻塞（2026-10-20）

本轮按“PMD修通少量关键流水，上卡后autotune选配置”的方向推进。
基线为 `perf/gemm-optimization` 的 `2d3ced8`；源码、默认配置与编译器均未修改，
只在仓库外副本增加warp-specialize入口、BK64 scale索引及常量分组实验。

## 结果与候选

28个编译预检场景有18个编译通过、10个WDRA/WASP阻塞；另做4个纯dot编译对照。
后续含长K、A E8M0及整行scale补测，共27个单CTA数值/性能点通过精度门限。
不是完整需求矩阵或完整GPU性能验收，编译通过也不代表所请求流水实际选中。

| 路径 | 同配置配对结果 | 当前判定 |
| --- | --- | --- |
| vLLM风格64×64/BK128/4warps async，K4096 | 普通77145→async66327 cycles，下降14.02% | 保留候选；实际生成async LDS加载，零spill；配对中的量化group_n/group_k均为constexpr |
| aiter风格128×128/BK64/8warps BPP，K1024 | 普通54620.7→BPP50586.7，下降7.39% | 确认BPP实际应用 |
| 同上，K4096 | 普通212888.7→BPP162118.3，下降23.85% | 同BK有收益，但不等于全局最优 |
| aiter128×128/BK128/8warps普通流水，K4096 | 130158.3 cycles | 比上述BK64 BPP更快，必须保留作跨BK参考 |
| vLLM128×128/BK64 BPP | 未出现BPP应用标记；常量分组后仍未选中 | 不能计作BPP支持；开关改变cache swizzle不等于实现该流水 |
| 大tile async | aiter128×128的4/8warps均变慢；vLLM8warps仅约2.3%差距 | 未找到足够稳健的晋升依据 |
| WDRA48/WDRA44/WASP44 | 实际post-dot scale循环编译阻塞 | 先修编译器数据流调度，不继续扫参数 |

输入为FP8 E4M3，W scale使用E8M0/N1/K128，默认A scale为FP32/K128，输出BF16。
候选另验证A E8M0/K128和FP32整行scale，均通过atol=.01/rtol=.01。
BK64改变分段缩放与累加顺序：两个长K点各1元素差3.8147e-6，另一个整行scale点
1元素差.00390625，均在门限内，不能声称全部bitwise一致。
W浮点scale、尾块和多CTA候选验收仍需补齐，不能把旧普通路径的smoke直接当作新流水验收。

## WDRA/WASP具体阻塞

当前生产循环没有warp_specialize标记。实验副本增加标记后，实测拓扑为：
WDRA48 `(4,4,4)` / wdra=True；WDRA44 `(4,4)` / True；WASP44 `(4,4)` / False。
WASP stages是LDS缓冲深度，当前编译器允许2/4，不可直接套普通流水的stages3。

两个内核的WDRA44/WASP44、WDRA48及WDRA48+async均停在：

```
RewritePartitionDependencies.cpp:219
UseInfo::getMaxUseDistance
Assertion `dist > 0 && "expected verifier to check schedule validity"' failed.
```

10个初始失败、1个constant-group重试和2个BK64 WDRA48重试共13个场景出现同一断言。
纯`acc=tl.dot(a,b,acc)`对照可以编译出WDRA初始化及abarrier，表明不是缺FP8 MMAC。
问题与dot后scale乘法/独立累加的数据流相关，具体SSA边尚未定位。
修复方向是核对消费分区归属、loop-carried accumulator和跨分区stage距离；
不能删除assert、跳过scale或替换为软件实现后计PASS。

纯dot WDRA48 128×128/BK128对照的metadata LDS=98304，超过arch_info 65536，
因此未执行，也不计精度PASS。修通调度后还需在容量域内选择BK64/矩形tile或改善布局。

## BPP边界

BlockPingpong.cpp:619–627明确排除BK128的“dot结果先缩放再独立累加”模式。
非BufferLds选择路径还要求M/N按128粒度，64×64不能盲目启用BPP。
当前有效实验使用128×128/BK64/8warps。
aiter副本把scale索引改为`(k*BK)//128`，让BK64继续正确读取K128 scale；
原生产kernel和包装层的GROUP_K==BK限制均未改。

## 证据与下一步

所有脚本、配置、日志、StatLog、IR与汇编在仓库外：
`/xukang/code/pmd-triton-sessions/fp8-pipeline-probes-20261020/`。
关键文件为REPORT.md、performance-summary.csv、final-results.json和safety-proof.json。
每worker的ir/保存TTIR/TTGIR/LLIR/AMDGPU汇编，完整编译失败含MLIR复现。

每个性能点先warmup编译、执行预热一次，再测3次system.gcSimCycles，JIT不计时。
hostSeconds仅作预算；完整资源、指令、LGKM/L2与抖动记录在外部结果中。
早期临时脚本曾因GPUTarget不能JSON序列化导致写盘失败，已在repaired-*重试；
不把该脚本错误记作编译器失败。

下一步优先修post-dot scale的WDRA调度，再核实资源域和数值；保留小tile async与
128×128/BK64 BPP作为候选，定位vLLM BPP未匹配原因。补尾块/跨CTA精度后再接入
可选config；默认配置保持原样，上卡再对这些流水家族做autotune。
五个暂停进程保持T、CPU ticks不变，三个原业务工区干净，S03/libtriton哈希不变。
