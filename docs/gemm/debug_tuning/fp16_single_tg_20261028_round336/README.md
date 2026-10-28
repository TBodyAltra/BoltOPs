# BK128：小block分片流水，64KiB预算优化

完整地图9103候选、503条选择。本批60个候选、3个配方独立确认、更新6条64KiB排名，497条选择不变。所有新配方实际LDS48KiB。

|逻辑BM×BN×BK|布局|初测利用率|唯一独立复测利用率|复测相对原64KiB最佳提升|
|---|---|---:|---:|---:|
|64×128×128|TT|32.885511%|31.091544%|42.735692%|
|64×128×128|TN|32.302839%|32.319832%|27.585481%|
|128×64×128|TN|27.421226%|29.226525%|34.711255%|

WASP开启、WDRA关闭，producer/consumer见逐项配置（[2,4]或[1,4]），num_stages2、UF1、WASP_GLOBAL_K_FRAGMENTS=2、逻辑BK128/实际global与dotK64、MMA_K_SPLIT1，纯MLS、无async/BPP/matrix_store，物理scratch/VGPR/SGPR spill0。best/best_no_spill不变；仅best_64k及best_64k_no_spill更新。百分比是有用FLOPs/4096模型峰值的单TG斜率比，不是BU实卡忙碌率。

初测60个候选全部稳态、MMOP工作比1、fullGOLD通过。初测242模型/968dispatch，含60个4TG非原点M/N及odd17正确性边界；边界排除性能斜率。唯一独立复测3配方各一套16/32/64系列及2控制，共11模型/44dispatch，9对实际HSACO的allocated/非debug ELF段一致、metadata除cache hash外相同、runtime/config/输入契约一致。初测与复测每项相对原同预算最佳均>=3%、稳态残差<=2%。所有253模型点和1012份原始dispatch stats的无损gzip均附带，包括57个未获提升的初测配方。

K仍是运行时输入，输入域K%128=0；逻辑16/32/64轮对应物理32/64/128轮，odd17对应物理34轮。M/N为运行时输入，无目标MNK特化或TG排布变更。Bolt297完整kernel patch及源码附带，基线840d816b671983527e5b1acc831c5e67618fe981；native209 SHA e2ae375d24cd4c61ee9085f19f26967597f4e8355c1781461c0b74d92bba815a，附完整Triton209源码补丁。默认globalfragment1旧控制兼容性证明保持不变。既有控制之一的SGPR spill4如实保留。128³暂缓。

另附64次未进入PMD的UF探索失败记录：332的48次stage2 UF2/4、334的8次普通WASP stage1、335的8次受支持WDRA单缓冲stage1。每次job/result/完整编译日志gzip均保留，失败原因按原日志记录。16个abarrier资源限制及普通WASP阶段约束均未改写，没有通过增大资源上限规避编译错误。diagnostic333的8个点仅作为循环开销假设的计数器证据；wave/qcycle等待计数重叠，不能等同CU关键路径或MMAC忙碌率，两个循环长度的斜率不能代替32轮稳态及独立复测。
