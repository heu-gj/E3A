# H2 — HOI4D Real Data Integrity & Visualization Audit

任务日期：2026-11-22（Asia/Singapore）。基准 `155820d65c2f8e47101d5aa204d0fcea79cb6776`；分支 `audit/h2-hoi4d-realdata-integrity`。数字一致性 **PASS**，详细结论见 [report.json](report.json) 的checks与digital_integrity_status，物理GT状态单独保留。整体 **PARTIAL**：数字读取一致性不等价于物理监督已经认证。

## 1. 前置检查与边界

创建分支前工作区干净、HEAD等于指定基准。19份P1-EgoExo本地保留材料与其历史manifest逐份大小/SHA256一致；仅两个local_only子目录被本地忽略，父目录与新报告仍对Git可见。egg-info是editable安装产物，另行本地忽略。未删除、移动或覆盖上述材料，也未提交本地exclude规则。

所有执行使用仓库 **.venv / Python3.10.19 / PyTorch2.6.0+cpu / NumPy1.26.4 / OpenCV4.11.0**。pip check通过，cache权限警告不影响依赖检查。经用户明确批准，只新增pytest及轻量依赖；没有修改IceVLM-Seg或其他Conda环境。参见 [environment.json](logs/environment.json)。

未修改原始数据、生产Dataset/decoder、Backbone、Heads、Loss、Trainer。未下载数据或模型、未训练、未执行MANO forward。源路径均相对于配置的数据根；repository与archive路径有各自命名空间。本报告没有宣称pixel_valid已被任意下游Loss正确消费。

## 2. 真实序列与H1重新执行

选择既有H1序列 `ZY20210800001/H1/C20/N30/S295/s04/T4`（Chair），本地提取布局沿用P1。具体发布revision仍UNKNOWN，不把目录名当版本证据。

已原样重跑H1：`python scripts/smoke_test_hoi4d_dataset.py --height 540 --width 960 --num-workers 0`，设置真实HOI4D_ROOT，OMP/MKL线程为4。完整stdout、stderr与解析JSON分别在 [smoke_stdout.txt](logs/smoke_stdout.txt)、[smoke_stderr.txt](logs/smoke_stderr.txt)、[smoke_test.json](logs/smoke_test.json)。退出码0、38个batch、300/300源帧；16个目标无效像素、467个有效MANO手帧、14事件、34个有效Time clip、0个有效Action clip。

38×8=304次帧观测，末尾292..299对齐clip与前一clip重复292..295；**304个objpose读取记录不是304个独立JSON**。覆盖及重复索引在证据中显式区分。

## 3. 独立参考与随机读取实验

参考不调用Dataset的mask/resize/pose/手姿/未来事件构建函数：两路视频各从0顺序read到EOF，逐帧保存原始像素哈希、PTS与本地缩放参考。RGB独立执行BGR通道反转与INTER_AREA；depth关闭RGB转换后读uint16，缩放用源数组 `[::2,::2]`，不用Dataset的resize helper。mask参照固定官方C20表独立硬编码RGB映射；逐2×2的all-valid逻辑独立计算保守像素有效性。

随后以seed220打乱38个Dataset clip，实际读取所有300个源帧；重复帧也比较。另用实际生产raw seek helper读所有clip，比较未缩放数组哈希与顺序参考，以覆盖seek、RGB通道顺序、深度原始值。两者仍共享OpenCV后端，因此结论不超出当前软件栈的数字一致性。

对照JSON与逐帧指标见 [rgb_depth_integrity.json](evidence/rgb_depth_integrity.json)，随机访问顺序与执行信息见 [audit_execution.json](evidence/audit_execution.json)。每帧包括元素数、不同元素数、最大/平均绝对误差；不是只以frame count/FPS宣称相同。

实测全量数字一致性 **PASS**：RGB/depth不同元素数均0、最大误差均0；所有raw seek哈希等于独立顺序解码；所有mask/validity、原始JSON、MANO参数与K对照一致。详细布尔结论与浮点时间误差见 [summary_metrics.json](evidence/summary_metrics.json)。

## 4. RGB / Depth

两路原始流均1920×1080、15FPS、顺序实际解码300帧。Dataset目标540×960。RGB比较float32归一化值并核对raw RGB哈希；depth比较完整uint16源值和float32输出的原始数值，无 `/1000`、无显示归一化。

**Dataset深度tensor为float32，不是uint16**。需要证明的是uint16原始整数值在float32中精确保留；此数值范围可被float32精确表达，不是把dtype不同误判为数值损失。独立源全序列最大值7999。源深度零值占比均值 **28.5445761%**，缩放后约 **28.5389976%**，逐帧统计及depth_valid逐像素比较保留在JSON。

[frame_alignment.csv](evidence/frame_alignment.csv) 分开保存文件索引、sample index、实测顺序视频PTS、index/FPS、Dataset float32 timestamp及raw frameId。float32时间的舍入不等于传感器时钟。等帧数、FPS或精确的数字像素复制均不证明RGB/Depth物理空间配准、采集同步或传感器无丢帧。

## 5. Motion Mask与frame220

全部300张P模式PNG显式转RGB后检查。固定官方HOI4D Script commit **788b43552176c77c01c3e98dd5f101fd6a94ee7a** 的C20表：palette1 Chair（semantic2/instance1）、palette2 Right Hand（47/2）、palette3 Left Hand（47/3）、黑色背景0。此处instance是类别局部provisional编号，不是E3A slot或独立认证的Persistent ID。

[mask_integrity.json](evidence/mask_integrity.json) 记录每帧源SHA256、所有颜色/像素数、semantic/instance统计、源未知像素、目标无效像素及Dataset逐帧一致性。第220帧为唯一异常色：RGB `[0,0,128]`，全局palette4但超出C20定义，共 **37像素**；语义保持UNKNOWN，不补猜测。

frame220：原始Chair像素337,883、手部127,733；保守下采样后16个目标像素无效，已知Chair目标像素84,486。独立2×2 all-valid结果与Dataset pixel_valid逐像素比较；未知像素不能因为nearest没有采到而漏掉。已知前景应保留，未知对应的entity mask为0且pixel_valid=false，**该0仅是无效占位，不是有效负背景**。

本阶段核验Dataset输出的排除标志及mask fidelity；没有运行Loss，不声称下游训练实际消费了pixel_valid。因此是否能直接启用任意Loss仍须额外验证，不通过降低审计标准获得PASS。

## 6. Object Pose

[objpose_integrity.json](evidence/objpose_integrity.json)：重读0..299所有JSON，记录文件索引、raw frameId、isEffective、dataList数量、object id/label和box存在性，比较Dataset保存的完整raw dict以及单独frameId tensor。未公开原始filePath内部人名目录或整套原始GT。

源记录均id1/label chair，31帧有非空camera1 box、269帧无非空box。混合raw编号：0→1、10→2、150→16、299→31；其余多数等于文件索引。不能以固定偏移解释。Dataset必须以文件索引寻址、原始frameId独立保存，不能拿raw frameId寻址视频，也不根据数字相同建立Entity Slot关联。

coordinates / units / transform direction 均 **UNKNOWN**；不画未经验证的pose投影、不称其为可靠3D物理GT。

## 7. MANO

[mano_integrity.json](evidence/mano_integrity.json)：源Right234、Left233，共467个唯一有效手帧。扫描全部文件，shape分别poseCoeff48、beta10、trans3、kps2D21×2，记录dtype、finite与源数组SHA256；缺帧逐索引公开。对Dataset对应四个字段逐元素比较，2D坐标只按x/y均0.5缩放，poseCoeff/beta/trans不做坐标或单位变换。

缺文件的valid必须false，零参数只是占位；不把零手误作真值。Right/Left顺序独立核对。缩放算术通过不证明源kps2D坐标语义、可见性或正确相机投影；没有MANO forward、没有恢复21关节3D，也不产生新的Hand loss GT。

## 8. Action Anticipation

[action_integrity.json](evidence/action_integrity.json)：真实14事件，declared/video duration均20秒；保留源event/start/end、顺序、重叠。微小重叠不擅自裁剪。用原始事件构造独立严格未来oracle：选 `min(startTime > cutoff)`，TTE=onset−cutoff。

比较38个实际Dataset clip（cutoff来自实际float32 timestamps末帧），复核34个有效Time clip、0个有效Action clip。另对每个真实start前ε、等于start、后ε及0/20秒运行边界对照；source double边界与float32 clip cutoff分开记录，避免把舍入混成规则错误。

没有传Action Vocabulary，分类无效是预期，不从单条序列构造正式训练词表。未来Target始终invalid，Action名字不能替代目标实例标注。未把未来帧输入模型（根本未运行模型）。

## 9. Camera Intrinsics

[intrinsics_integrity.json](evidence/intrinsics_integrity.json) 记录原始K、half-size期望K和38次Dataset输出指标。原K的fx=1060.2955、fy=1061.5068、cx=971.52105、cy=523.2619；960×540下x/y均按0.5缩放，底行保持 `[0,0,1]`。生产reader先转float32，审计明确比较预期float32，不把表示舍入当坐标漂移。

这是图像resize对应的矩阵算术验证，**不是相机物理标定、畸变模型或Depth坐标配准验收**。

## 10. 本地可视化

见 [visualizations/README.md](visualizations/README.md)。必选0/75/150/220/224/299，加真实Action start/end的floor/ceil相邻源帧，共32个位置（完整列表在audit_execution）。每个位置保存六联图及RGB、实例叠加、unknown指示、depth显示、原始MANO2D独立图，均位于 `visualizations/local_only/`，不上传GitHub。

图中明确标注PTS、当前Action区间/严格未来onset、Depth仅显示归一化、MANO2D语义未验证。Unknown的黄色可视化标记为可读性膨胀，证据像素数不膨胀。Overlay仅诊断，不能证明严格配准或物理接触。公开manifest逐文件记录相对路径/size/SHA256。没有图像生成模型或新GT。

## 11. 结论等级与剩余门槛

| 结论 | 状态范围与证据 |
|---|---|
| 全量数字读取/resize一致性 | 实测结果由report.json checks决定；全300帧源参考与随机读取，不依赖合成测试 |
| Mask未知区域排除 | Dataset层完整对照；Loss消费NOT_RUN |
| objpose原始字段保留 | 原始JSON对照；物理语义NEEDS_REVIEW |
| MANO原始参数/2D缩放 | 源数组对照；3D恢复NOT_RUN |
| Action/Time读取规则 | 源事件及所有clip/边界对照；目标实体无效 |
| K缩放 | 算术对照；物理标定NOT_RUN |
| 完整物理实体/3D/接触/目标链路 | NEEDS_REVIEW，Target/Interaction保持invalid |

本阶段不修改生产代码。若真实比对出现差异，证据保留并降低相应状态，不以合成单测通过掩盖；没有证据支持升级物理GT有效性。

## 12. 测试与复现

`logs/pytest.txt` 和 `logs/tests.json` 保存真实测试输出/退出状态。辅助测试覆盖指标、unknown2×2排除、混合frameId、uint16值保存、MANO缺帧与strict-future边界；此外运行现有Dataset/HOI4D/collate测试。合成测试不是全量真实数据审计的替代，真实证据另列。

本次 **37 tests passed**；真实Action严格未来边界独立核验42个cutoff，38个实际clip规则均一致。测试失败不会通过更改GT或生产代码绕过。

使用 `.venv` Python，设置HOI4D_ROOT，依次运行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/audit_hoi4d_realdata.py smoke
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/audit_hoi4d_realdata.py reference
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/audit_hoi4d_realdata.py audit
python scripts/audit_hoi4d_realdata.py report
python scripts/audit_hoi4d_realdata.py tests
python scripts/audit_hoi4d_realdata.py report
python scripts/audit_hoi4d_realdata.py seal
```

source root仅以参数/环境变量提供，不硬编码本机路径。证据、日志及源哈希在manifest和SHA256SUMS，原始/派生图像、完整标注和本地参考NPZ均不暂存。最终交付仅PR，禁止Merge；H2完成后等待独立审查，不进入H3。
