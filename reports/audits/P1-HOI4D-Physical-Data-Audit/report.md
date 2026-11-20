# P1-HOI4D-Physical-Data-Audit

审计日期：2026-11-21。最终状态：**PARTIAL**。已完成本地单序列审计，完整训练监督链路仍有阻塞。

Identity & Geometry follow-up 已重读原始数据，撤回首轮 frameId 固定偏移结论。以下正文已更正；新增证据和范围见第14节。模型、Heads、Loss、Trainer 和正式 decoder 未修改。

## 1. 范围、环境与证据

初始 `git status --short --branch`：工作区干净，分支 `feat/e3a-foundation`，HEAD `9f96d5813655e0fb9ceebad107b2ff2d1d2d5d07`。未修改原始数据、模型、正式 adapter 或训练配置；未解压归档、安装依赖或执行 GPU 训练。

审计程序：[audit_hoi4d_physical.py](../../../scripts/audit_hoi4d_physical.py)。证据目录：[归档证据](evidence/)。所有源文件的数据集根相对路径、大小见 [inventory.json](evidence/inventory.json)，选中序列全部文件均列出。关键源文件 SHA256 在 [结构化报告](report.json) 的 `source_fingerprints`。

使用现有 `python`，OpenCV 4.13.0、NumPy 2.2.6、PyTorch 2.5.1+cu121。此环境不是项目锁定训练环境（项目建议 NumPy <2）；仅做 CPU 数据读取。`ffmpeg/ffprobe` 不在 PATH，`manopth`、`smplx` 不可导入。无新增安装。

## 2. 本地发布版与目录（PARTIAL）

`HOI4D_ROOT` 未设置。归档中将实际数据集根记为 `.`，不公开本机根路径。

| 内容 | 相对根路径 | 发布形态 |
|---|---|---|
| RGB | `extracted/release/HOI4D_release` | `sequence/align_rgb/image.mp4` |
| Depth | `extracted/depth/HOI4D_depth_video` | `sequence/align_depth/depth_video.avi` |
| Mask / objpose / Action / 静态场景 | `extracted/annotations/HOI4D_annotations` | `sequence/{2Dseg/mask,objpose,action,3Dseg}` |
| Right MANO | `extracted/hand_pose/Hand_pose/handpose_right_hand` | `sequence/<integer>.pickle` |
| Left MANO | `extracted/hand_pose/Hand_pose/handpose_left_hand` | 同上 |
| 相机内参 | `extracted/camera_params/camera_params` | `ZY20210800001/intrin.npy` 等四相机 |

归档实际存在：`raw/archives/HOI4D_release.zip`（23,269,777,121 bytes）、`HOI4D_annotations.zip`（22,252,507,455 bytes），Depth 为 `raw/depth_archive/3516989438_g944468183/HOI4D_depth_video.tar.gz0` 至 `...gz6`。未读取归档内容或执行解压。版本只能确认是上述目录形态；具体 release revision / 下载源日期 / 完整性校验 manifest **UNKNOWN**。不能仅凭目录名称认定等同最新发布版本。

官方组织说明：[HOI4D Instructions](https://github.com/leolyliu/HOI4D-Instructions)。其中 `3Dseg` 是静态场景；动态 4D labels 由单独脚本生成，二者不能混用。

## 3. 序列选择（PASS：覆盖率选择，不代表所有模态验证通过）

选中：`ZY20210800001/H1/C20/N30/S295/s04/T4`；类别 C20 Chair，相机 ZY20210800001、人 H1、源对象 N30、房间 S295、布局 s04、任务 T4。

扫描范围仅限上述本地数据根下 RGB 序列，共 2,973 条。排序证据：[candidates.json](evidence/candidates.json)。先按存在的模态目录/文件数排序，再按 `min(mask,objpose,right,left)` 排序，再按左右手总数量，最终按 Sequence ID 字典序。8 项是 RGB、Depth、Mask、静态场景、objpose、Action、Right、Left 的“文件存在”分数；**静态场景存在不等价于动态实例存在**。分数分布 8:71、7:550、6:1559、5:793。

第一名左右手 234/233 帧，第二名 `ZY20210800001/H1/C20/N27/S292/s03/T3` 为 231/231，故选第一名。两者 mask/objpose 都为 300。文件存在率只用于选取候选；后续发现的异常会降低验证结论。

选中序列所有路径中的 `sequence` 均替换为上述 ID。逐文件大小、文件数和模态总 bytes 均保存在 inventory 和 JSON `modalities`，并非估算。

| 模态 | 文件数量 | 总 bytes |
|---|---:|---:|
| RGB | 1 | 9,695,354 |
| Depth | 1 | 50,654,776 |
| Motion | 300 | 3,052,846 |
| 静态3Dseg | 3 | 20,520,806 |
| objpose | 300 | 133,075 |
| Action | 1 | 2,481 |
| MANO Right | 234 | 209,430 |
| MANO Left | 233 | 208,535 |

## 4. 现有 E3A 实现审查（PASS：代码事实；真实可靠性另列）

已阅读 `docs/architecture/overview.md`、`docs/implementation/roadmap.md`、`docs/implementation/training.md`、`src/e3a/data/{hoi4d,actions,collate}.py` 和 `tests/test_hoi4d_data.py`。

| 实现 | 当前行为 | 本次结论 |
|---|---|---|
| `hoi4d_color_palette` | bit-interleaved RGB palette | 算法与官方 color_map 一致；类别允许色仍可能不足 |
| `decode_hoi4d_motion_mask` | uint8 RGB 输入；未知色 fail closed；C20 仅 palette 1,2,3 | 真实后段第4色导致异常，不能直接训练 |
| `build_hoi4d_entity_clip_targets` | 按 provisional instance 合并非手 parts，排除 semantic 47；不可见/缺失置 `mask_valid=False` | 规则明确，但没有真实持久身份凭据 |
| Action parser | `info.duration/events`、legacy marks、single event 三种布局，秒区间，严格时长检查 | 该序列 events 布局兼容；其余布局仍只是 synthetic 测试 |
| anticipation builder | 第一个 `start_time > cutoff`，差值为 TTE；未知词表仅禁用 action；target invalid | 与本条真实事件列表的边界验证一致 |
| `actions.py` | 确定性 dataset-prefixed 词表 | 不应将本次审计序列当作训练 split 生成正式词表 |
| collate | labels `[B,N]`、ids `[B,N,T]`、masks `[B,N,T,H,W]`、mask_valid、track_counts；缺失任务 false validity | 仅张量合同，不是原始数据读取与坐标转换 |

Clip contract：labels `[N]` 为前景0，ids `[N,T]` 是稳定 clip-local ID，可见时非负；不可见或未标注为 -1，mask=0；`mask_valid` 区分未标注与显式空。Right、Left 顺序的 hand target 为 `[T,2,21,3]`。当前缺 clip-level `entity_supervision_valid`，零实体窗口仍可能被当作全负类。

当前没有 HOI4D 原始 RGB-D Dataset、MANO forward reader、objpose reader、source-to-canonical 变换、源对象到动态 mask ID 映射、接触标签和事件目标映射。此前单元测试使用构造数组/字典。本次相关测试 **26 passed in 1.72s**，仅支持解析/张量逻辑，不证明物理 GT 正确。

## 5. RGB / Depth（解码 PASS；严格时空配准 PARTIAL）

两路均顺序 `VideoCapture.read()` 到 EOF，实际解码均为 300 帧，1920×1080、15 FPS、20秒。RGB FourCC h264，Depth ffv1。Depth 使用 `CAP_PROP_CONVERT_RGB=0`，输出 `[1080,1920] uint16`，不是默认 8-bit 三通道。没有同时存在 JPEG RGB / PNG Depth 的读取歧义：源是视频，mask 才是 PNG；只在报告目录输出抽样图与原始深度数组。

抽样序号 `0,75,150,224,299`，时间 `0,5,10,14.933333,19.933333` 秒，覆盖开始、25%、50%、75%、结束（后两者按可用帧范围取整）。每路全部视频时间位置见 JSON `videos.*.timestamps_ms`。编码流无中途读取失败，但原始传感器是否丢帧 **UNKNOWN**；没有采集时间戳，不能从300帧推断传感器从未丢帧。

五张六联图：0（本地保留、未公开：visualizations/local_only/frame_00000_audit.jpg）、75（本地保留、未公开：visualizations/local_only/frame_00075_audit.jpg）、150（本地保留、未公开：visualizations/local_only/frame_00150_audit.jpg）、224（本地保留、未公开：visualizations/local_only/frame_00224_audit.jpg）、299（本地保留、未公开：visualizations/local_only/frame_00299_audit.jpg）。每个时间位置另有 RGB、Motion、provisional instance、overlay PNG 及 `depth_raw.npy`。

官方 [decode.py](https://github.com/hoi4d/HOI4D_Script/blob/788b43552176c77c01c3e98dd5f101fd6a94ee7a/utils/decode.py) 指定15 FPS、从0编号；与此处视频一致。这支持编码帧序号对应，但 mask/MANO 没有独立时间字段，不能仅凭同名给出严格同步 PASS。

空间检查：椅子在 RGB、Depth 和 mask 的位置/轮廓整体对应。`frame_*_depth_edges.jpg` 以青色显示深度跳变、洋红色显示椅子 mask 边界。JSON `physical_diagnostics` 提供零偏移和 ±16px 网格平移的边缘距离；该度量受无效深度、遮挡和稀疏边缘影响，**不是标定结果**，不能宣布像素级零偏移。未估计或写入坐标变换。

Depth 原始值的非零分位数和零值比例逐样本见 JSON。首帧范围0–7983，零值24.7723%，非零中位数1181。Depth 单位 **UNKNOWN / NEEDS_REVIEW**。官方 4D 生成脚本调用 Open3D 默认深度转换，提示1000比例的候选约定，但原始流没有明确单位元数据；数值量级不作为确认毫米的依据。不得直接应用 `/1000` 作为已验证 GT。

## 6. Motion / Instance（PARTIAL / NEEDS_REVIEW）

逐帧真实 PNG 以 Pillow 转为 RGB 后调用现有 `decode_hoi4d_motion_mask()`。完整色彩、semantic、每个 provisional 实例的像素面积、bbox 和失败记录见 [mask_statistics.json](evidence/mask_statistics.json)。正确色彩顺序可由 `[128,0,0]` 椅子色与 overlay 复核。

官方 [label.csv](https://github.com/hoi4d/HOI4D_Script/blob/788b43552176c77c01c3e98dd5f101fd6a94ee7a/definitions/motion%20segmentation/label.csv) 的 C20 是 Chair / Right Hand / Left Hand；现有 semantic map 是2 /47 /47，实例1 /2 /3。真实后段出现 `[0,0,128]`（palette index4）：属于全局官方 palette，却超出 C20 官方表和代码映射。**unknown 的是类别语义/物理实体，不是调色板算法。** 精确失败帧和像素总数在 JSON `mask.raw_decoder_failed_frames/unknown_pixels`。

保留 fail-closed 结果。诊断程序仅为展示已知区域，将未知像素在临时副本中置背景以调用原 decoder，再将这些区域恢复为 semantic/instance `-1`；实例图用黄色标出 UNKNOWN。此图不是新增 GT。正式 Dataset 未使用该回退，未扩充代码中的类别映射。

实际异常仅 **frame220 / `2Dseg/mask/00220.png` 的37像素**，299帧直接解码通过；不能把一个局部异常夸大为全部mask损坏。该异常需隔离或确认语义后处理。异常全图（本地保留、未公开：visualizations/local_only/unknown_frame_00220.png） 与 局部放大（本地保留、未公开：visualizations/local_only/unknown_frame_00220_crop.png） 标黄显示未知区域。源PNG均为P模式，先显式转RGB；不是将索引数组直接交给decoder。

官方 [4D mapping](https://github.com/hoi4d/HOI4D_Script/blob/788b43552176c77c01c3e98dd5f101fd6a94ee7a/prepare_4Dseg/utils.py) 只定义类别局部编号。[scene2frame.py](https://github.com/hoi4d/HOI4D_Script/blob/788b43552176c77c01c3e98dd5f101fd6a94ee7a/prepare_4Dseg/scene2frame.py) 对动态前景 ID 加上静态场景的最大实例 ID。因此 decoder 的1/2/3 **不必等于最终4D输出的ID**。必须验证 offset 和源文件，而不是比较数字相等。

本条 `3Dseg/label.pcd` 的实际字段为 `x y z rgb label ID`，是静态点云；字段集合、静态 ID 范围见 JSON `static_pcd`。未找到选中序列的逐帧动态4D文件。源目录 N30、objpose.id=1、provisional instance1、static PCD ID 是四个不同命名空间，尚无验证的 join 表。

本序列 C20 仅一个已知物体 part，不能证明 Laptop/Bucket 等多部件应合并。相同椅子色跨抽样帧视觉连续，只能支持 provisional 身份，不足以证明全序列或所有类别的 Persistent Entity GT。缺像素不能区分遮挡、出画、消失或缺标，不生成负标签。

## 7. Object Pose（结构 PASS；物理 GT NEEDS_REVIEW）

真实 `objpose/0.json` root：`frameId,filePath,isEffective,dataList`；对象字段 `id,label,2dBox,center,dimensions,rotation,properties`。首项 `id=1,label=chair`；center `(-0.05367000165,0.16870085862,1.05134063188)`；rotation `(2.65051698685,-0.06126411259,-1.57726144791)`；dimensions `length=.37789615979,width=.35440362967,height=.51267838268`。`properties={}`。公开逐帧索引及数值诊断见 [objpose_statistics.json](evidence/objpose_statistics.json)。

全部 filename `0..299`；首轮关于固定偏移的结论错误，现已撤回。原始300份JSON中269份的frameId等于文件索引；另31份（0、10、…、290、299）含非空filePath，其frameId依次为1至31。字段可能混用了关键帧序号和插值文件索引，但官方文档未定义该语义，必须保留 **UNKNOWN**。`filePath` 是发布前内部PCD路径，不是本地可读取路径；没有显式采集时间戳。`isEffective=1` 的300条记录也不能证明可见性或时间对齐。详见 [原始字段复核](evidence/objpose_frameid_audit.json)。

300次Euler重建的最大 `||RᵀR-I||=1.92556890852693e-15`，det范围 `[0.9999999999999988,1.0000000000000016]`。静态label.pcd实际300,179点，静态ID为0、1、2；与前景局部ID数字重合不能视为物理身份相同。

官方 reader 使用 `Rotation.from_euler('XYZ', rot)`，所以此处按同一弧度约定重建 R，逐帧检查 `RᵀR≈I`、`det(R)≈1`；最大误差和det范围见 JSON `objpose`。**这是解析后数学自洽检查**：Euler 构造本身保证SO(3)，不能证明轴顺序、手性、实际物理方向或单位正确。原始文件不是四元数，也没有4×4变换矩阵，不做无依据的四元数顺序检查。

官方说明提供 CAD canonical frame 概念，但本地选中序列没有配套CAD。本次未确认 object-to-camera / object-to-world、左右手坐标约定、单位，以及 dimensions 是否还需缩放mesh。均为 **UNKNOWN**。JSON `physical_diagnostics` 的 `project_center_as_camera_hypothesis_uv` 只是将 center 当camera坐标的投影假设，不能据此认定坐标系。尚未做CAD重投影与Depth一致性验证，因此不能启用可靠物体3D loss。

## 8. MANO（参数结构 PASS；21关节恢复 BLOCKED）

左右手来源由 `handpose_right_hand` / `handpose_left_hand` 目录区分；pickle 自身没有 side 或 timestamp。实际字段：`poseCoeff (48,)`、`beta (10,)`、`trans (3,)`、`kps2D (21,2)`，全部实际shape/dtype/有限性在 `mano_{right,left}_statistics.json`。按只允许 NumPy array reconstruction 的受限 unpickler 读取。

Right 234帧，Left 233帧；并非300帧都有手姿，缺帧完整列表在 JSON `mano.*.missing_frames`。首末源文件 Right 26–272、Left 27–272，中间也有缺标。缺失不得解释为手不存在。`beta_max_deviation` 记录同一人该序列形状参数稳定性，不能默认整个H1都完全相同。

根据官方 MANO 示例，poseCoeff 前3维是 global orientation，后45维是非PCA姿态；beta为shape。示例使用 `use_pca=False,ncomps=45,flat_hand_mean=True`，输出关节除1000再加trans，得到camera-frame米。**此单位/坐标说明只支持官方手姿约定，尚未通过本地forward和投影验证，不可扩展为Depth/objpose单位证明。**

下一步需用户提供合法 MANO_RIGHT.pkl / MANO_LEFT.pkl 路径与模型版本，并在受控环境提供 manopth；模型下载需要 [MANO 官方注册及许可](https://mano.is.tue.mpg.de/)，本次没有下载或全盘搜索模型。左右手分别forward，按时间堆叠Right、Left得到 `[T,2,21,3]`，缺失hand/joint置valid=false。

[manopth 源码](https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py) 添加五个指尖并重排21关节，顺序目标为 wrist、thumb、index、middle、ring、pinky。应锁定实现版本，核验左右指尖索引差异与 E3A 顺序；不得对已重排的manopth输出再次盲目重排。用本地K将恢复的3D joints投影，对比源kps2D和mask，核验误差、单位与帧偏移后才能解锁hand loss。

本次仅在抽样图绘制真实 `kps2D`，并记录落在对应手mask内的点数；这是2D对应证据，不是21个3D joints恢复成功。未输出或写入正式 Dataset 的3D手 GT。

样本75/150/224的Right落点为17/21、19/21、16/21；Left为10/21、21/21、15/21。可见轮廓、投影点与遮挡边缘并非严格一致，不能把所有kps2D自动标记可见。两手该序列beta最大偏差均为0。

## 9. Action（时间表兼容 PASS；帧级事件语义 PARTIAL）

真实路径 `.../HOI4D_annotations/sequence/action/color.json`；root为 `filePath,info,events`。`info.duration=20.0`；14 events，字段 `id,event,startTime,endTime,properties`，properties全空。category不在JSON，需要保留路径C20上下文。公开JSON保留事件数量、字段名及边界检查；完整原标注快照仅本地保留。

时间单位在该序列可与20秒视频一致核对，区间未超时长；现有 parser 原样兼容。标签含 rest、Reachout、Grasp、Carrywithbothhands、putdown、Stop、carry，保留大小写。事件到RGB帧的采样约定采用 `t=i/15`，区间显示用 `[start,end)`；不把非整数帧边界强制四舍五入为 GT。

严格未来定义已在真实14事件的每个start前ε、等于start、后ε及五个抽样时间、20秒末尾核验，结果见 `action.boundary_checks`。边界相等时跳过当前事件，选下一start；没有未来事件时 action/time/target validity 全false；本次没有构造训练词表，空词表使action invalid但保留可用time target。

共46项检查全部通过。例如cutoff=3.213跳过该时刻Grasp，下一onset=4.091、TTE=.878；cutoff=18.374及20无未来事件。重叠区间中的cutoff遵循同一严格规则，不因仍处于前一动作而选择已开始事件。

真实JSON存在微小重叠和间隙：例如 Grasp结束12.969024390243902而下个事件开始12.969，putdown结束17.457463414634145而Stop开始17.457。parser保留原值，未强制修剪；未来onset按start排序，重叠不改变严格大于规则。事件ID是事件序号，不是objpose.id或mask实例ID。没有直接目标对象或接触字段，不凭 Carrywithbothhands 等名称推断目标椅子。

与视频时长一致不等于人工事件边界在每个视频帧精确正确。采集源时钟、annotation作者的帧舍入策略仍 UNKNOWN。

## 10. 跨模态关系与 loss 可用性

完整300帧对应表：[cross_modal_frames.csv](evidence/cross_modal_frames.csv)。视频路径用 `#decoded_frame=i` 标识内存解码序号，不伪造源图片文件。所有 dynamic Instance ID 保留UNKNOWN。

| Frame | RGB / Depth | Motion | Instance ID | Object Pose | MANO | Action at i/15 |
|---|---|---|---|---|---|---|
| 0 | 两视频frame0 | 00000.png | 椅子provisional1；动态UNKNOWN | 0.json / frameId1 | 双手缺失 | rest |
| 75 | frame75（候选关联） | 00075.png | 已知色按1/2/3；动态UNKNOWN | 75.json / raw frameId75 | Right/Left 75.pickle均存在 | Carrywithbothhands |
| 150 | frame150（候选关联） | 00150.png | 同上 | 150.json / raw frameId16 | Right/Left 150.pickle均存在 | Stop |
| 224 | frame224（候选关联） | 00224.png | 同上；本帧无未知色 | 224.json / raw frameId224 | Right/Left 224.pickle均存在 | carry |
| 299 | frame299（候选关联） | 00299.png | 动态UNKNOWN | 299.json / raw frameId31 | 双手缺失 | rest |

表中同号文件与视频的关联仅为待验证候选；CSV新增独立的文件索引、raw frameId、候选解码索引/实测PTS、UNKNOWN的已验证姿态视频索引及标注时间戳。它们不是严格同步GT。

| Loss | 状态 | 允许使用的证据与门槛 |
|---|---|---|
| Entity | PARTIAL | 已知palette区域可做诊断；未知色与持久实例映射未解决，完整clip不能直接训练 |
| Hand | BLOCKED | MANO参数存在；需合法模型、forward、关节顺序/重投影验证 |
| Geometry | NEEDS_REVIEW | 原始uint16深度可读；单位、标定、无效深度与camera trajectory方向需核验 |
| Interaction | BLOCKED | 无真实contact字段；2D相邻/手mask覆盖不等价3D接触 |
| Target Entity | BLOCKED | Action没有目标instance；不能把唯一已知椅子自动作为目标 |
| Action / Time | PARTIAL | 时间表/strict-future逻辑验证通过；仍需train-only词表、split、采样与无泄漏规则 |

Hand → Entity 有2D图像层面的部分对应；3D坐标和接触未验证。Entity → Action 只有共序列关系，没有目标实例监督。因此完整 Hand → Entity → Action 监督链路 **BLOCKED**。

## 11. 发现问题与建议修改文件

本次只新增审计脚本与报告。以下是下一阶段建议，尚未实施：

1. `src/e3a/data/hoi4d.py`：版本化类别palette schema；调查真实C20第4色的含义、范围和发布差异。继续fail closed，加入明确 unknown/coverage 处理政策，不能补猜测类。添加具备来源证据的实例命名空间转换；把objpose文件索引和frameId分开存储。
2. 新增 `src/e3a/data/hoi4d_dataset.py`：分根目录manifest、CPU视频按时间读取、深度保真读取、缺模态coverage、观测截止和未来事件查询；先输出可追溯样本，未验证监督保持false。
3. 新增 MANO/objpose adapter（可置 `src/e3a/data/hoi4d_geometry.py`）：锁定合法模型版本、坐标/单位、canonical joint order；在源投影对比通过前不对外启用3D GT。
4. `src/e3a/data/collate.py` 及相关criterion adapter：增加clip级entity监督有效标志，确保无标注/全遮挡窗口不被训练成全负类。
5. `src/e3a/data/actions.py`：基础词表逻辑无此序列发现的解析缺陷；需要实际train split生成可追溯类别词表与split manifest，不需更改未来严格大于定义。
6. `tests/test_hoi4d_data.py`：增加受环境变量控制的真实数据集成检查、未知色负例、混合frameId编号、真实微小重叠边界、Depth uint16保真、手缺帧；不能用synthetic测试替代这些门槛。
7. `docs/implementation/{roadmap,training}.md`：在后续核验完成时逐项更新，当前P1实体/坐标验收不能打勾。

## 12. 剩余阻塞与复现

需要的外部信息：本地数据发布revision/manifest；未知C20第4色的官方定义或更正标注；逐帧4D实例标注路径（若下载包另存，请提供），以及其与局部mask ID、objpose ID的官方映射；objpose的坐标系/变换方向/单位说明；Depth scale与标定说明；配套Chair N30 CAD路径；合法MANO模型路径与版本。Action若没有目标实例扩展标注，则 target/contact supervision 本身缺失，不能靠配置消除。

复现命令（在仓库根，写入审计目录）：

```bash
HOI4D_ROOT=<local-dataset-root> python scripts/audit_hoi4d_physical.py
HOI4D_ROOT=<local-dataset-root> python scripts/audit_hoi4d_physical.py --audit
HOI4D_ROOT=<local-dataset-root> python scripts/audit_hoi4d_physical.py --finalize
PYTHONPATH=src python -m pytest -q tests/test_hoi4d_data.py tests/test_action_vocabulary.py tests/test_entity_target_collate.py tests/test_e3a_collate.py
```

**P1-HOI4D-Physical-Data-Audit: PARTIAL**。审计发现了可读且部分对应的真实模态，同时确认了未知mask颜色、动态实例缺证、3D物理约定和MANO恢复依赖、Action目标实体缺失等训练门槛。此次停止在P1审计交付，不进入下一阶段。


## 13. 归档说明与证据等级

本目录是唯一维护的最终结果。源数据路径均相对于数据集根；代码路径相对于仓库；证据链接相对于本归档。旧报告中两份JSON逐字节一致，归档前已核对哈希。原始标注快照和图像不进入Git；保留在本目录的Git忽略区域，并记录在manifest的withheld inventory。

| 结论 | 状态 | 证据等级及位置 |
|---|---|---|
| 文件覆盖率与视频解码 | PASS | 真实文件/解码测量：evidence/inventory.json、observations.json 的 videos |
| mask色彩兼容性 | PARTIAL | 真实300帧检查：mask_statistics.json；frame220失败37像素 |
| 严格跨模态配准 | PARTIAL | 本地目视检查、时间戳与边缘诊断；图像未公开，尚非物理标定 |
| 动态物理身份 | NEEDS_REVIEW | 缺逐帧动态实例文件；不能以静态ID替代 |
| objpose结构/旋转数值 | PASS | objpose_statistics.json；仅数学自洽检查 |
| objpose物理坐标/单位 | NEEDS_REVIEW | 没有经验证变换或单位证据 |
| MANO参数结构/有限性 | PASS | mano_*_statistics.json |
| MANO 3D恢复 | BLOCKED / NOT_RUN | 资产与依赖不足，未执行forward |
| Action时间表与未来事件逻辑 | PASS | observations.json action.boundary_checks，46项 |
| Action帧级人工语义边界 | PARTIAL | 时长一致不等价逐帧人工边界验证 |
| 完整Hand→Entity→Action链路 | BLOCKED | 缺目标、接触与3D对应证据 |

测试结果单独记录在logs；合成测试不用于提升真实数据结论等级。源数据重审未在归档阶段重复执行。归档阶段仅整理先前结果并重新运行已有解析/colllation测试及归档完整性检查。详见README与manifest。

## 14. Identity & Geometry Bridge Follow-up

本节是归档之后的新一次真实数据复核，不与首轮合成测试或归档过程混淆。所有官方HOI4D结论固定到 **788b43552176c77c01c3e98dd5f101fd6a94ee7a**。[官方README](https://github.com/hoi4d/HOI4D_Script/blob/788b43552176c77c01c3e98dd5f101fd6a94ee7a/README.md)、任务/类别CSV及四份生成脚本的URL与逐文件SHA256保存在 [dynamic_4d_pipeline_audit.json](evidence/dynamic_4d_pipeline_audit.json)。未以main作为唯一依据。

### 14.1 frameId 与索引纠错

[JSON](evidence/objpose_frameid_audit.json) / [CSV](evidence/objpose_frameid_audit.csv) 记录重新读取的全部300份源文件SHA256、filename index、raw frameId、isEffective、dataList数量/字段及object id/label。filePath公开其内部路径的basename与完整字段SHA256，剔除其中的人名等内部目录信息；原始JSON保留在数据集原位，未修改。31份非空filePath的PCD basename数字都等于文件索引，但这仅是内部索引证据，不是视频同步证明。

| 文件索引 | raw frameId | filePath basename |
|---:|---:|---|
| 0 | 1 | 0.pcd |
| 1 | 1 | 空 |
| 9 | 9 | 空 |
| 10 | 2 | 10.pcd |
| 11 | 11 | 空 |
| 19 | 19 | 空 |
| 20 | 3 | 20.pcd |
| 21 | 21 | 空 |
| 149 | 149 | 空 |
| 150 | 16 | 150.pcd |
| 151 | 151 | 空 |
| 299 | 31 | 299.pcd |

**PASS仅限异常识别和原始字段测量**；frameId的实际业务含义 **UNKNOWN / NEEDS_REVIEW**。269份空filePath记录的frameId等于索引，31份非空记录依次编号，更像关键帧与插值输出混合，但只是推断。文件索引与视频帧的同号关联获得下节几何支持，仍为PARTIAL，不能声称每帧或采集时间戳已验证。

### 14.2 N30 → Chair mask → objpose → dynamic ID → slot

[identity_mapping.json](evidence/identity_mapping.json) 中，300份objpose均仅有 `id=1,label=chair`；C20/N30目录和非空filePath中的椅子30名称支持N30为本序列主要椅子实例。它不是最终E3A slot编号，也没有以数字相同进行join。

只有31帧有实际非空2dBox；与同号Chair mask的bbox IoU范围 **0.4999–0.6562**、中位数 **0.6054**。其余269帧没有2dBox，不能报告为“300帧box匹配成功”。将center作为camera坐标、使用实际intrin.npy投影的候选检验在300/300帧落入Chair mask；这支持粗粒度物体关联，却不能单独证明坐标约定、精确时序、物理单位和CAD身份。结果 **PARTIAL**。

静态PCD实例集合为 `{0,1,2}`。官方前景规则 `max(static IDs)+local foreground ID` 因此条件性给出 Chair=3、Right=4、Left=5。**规则计算PASS，生成实例身份NOT_RUN**，不能把这些预测编号冒充已经产生的标签。E3A slot需要观测clip中matcher assignment，本阶段未运行，保持NOT_RUN。

### 14.3 官方动态4D链路与相机方向

`3Dseg/output.log`实测1500行、300组三整数header+4×4矩阵，header为 `[i,i,i+1]`。旋转最大正交误差约 `2.29e-8`。Open3D **0.19.0** 读出300组extrinsic，逐元素与 `inverse(raw log matrix)` 的最大差为 `3.33e-16`；直接与raw比较最大差约1.423。

[Open3D LOG实现](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/io/file_format/FileLOG.cpp) 在读取时求逆，[点云factory实现](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudFactory.cpp) 又将extrinsic求逆后把camera点变到scene。因此在该工具链约定下，**raw log是camera→scene，解析出的extrinsic是scene→camera**。此结论有源码和本地API测量支持（PASS：约定与数值），不等于所有数据的标定已通过。

另对5帧做真实静态PCD到Depth的投影诊断，未生成任何4D labels：每5点采样、zbuffer、排除前景与零深度，使用实际intrin.npy。假设深度比例1000时，官方方向在0/75/150/224/299帧的绝对误差中位数为 **24.20 / 6.21 / 4.59 / 12.75 / 8.38 原始depth单位**；错误地直接使用raw矩阵为 **24.20 / 280.66 / 382.77 / 73.40 / 88.56**。首帧identity无方向区分力。数据支持官方方向，几何整体仍 **PARTIAL**：稀疏投影、遮挡、帧关联和物理单位未完全标定；比例1000尚非数据单位认证。

生成链路的具体缺口：

- 实际有 `label.pcd`（字段x/y/z/rgb/label/ID），没有 `label.txt`；不能直接重命名或跳过header当成等价输入。
- 有 `2Dseg/mask`，没有 `refine_2Dseg`；palette可供选择器读取不等于精修标注质量/来源一致。
- 固定commit的README和生成器自身使用不同输入名，证明不能仅以“main版本漂移”解释。发布revision差异(B)和其他标注包(D)仍UNKNOWN；需要格式转换、解码等预处理(C)，但官方所需转换与精修来源未获确认。
- wrapper对 `data_dir` 的迭代与join类型/目录查找不一致；读取mask路径和refine检查不一致；视频需要先解码，而生成器读取无padding的i.jpg/i.png，decode.py写入5位padding。
- `plyfile`缺失，未安装。默认入口300帧/24进程并捕获隐藏异常，未运行；1–5帧官方最小生成也因前提不满足而 **NOT_RUN / BLOCKED**。
- LOG reader提供的默认内参为525/640×480，官方生成器改用Kinect2固定参数，又不同于实际intrin.npy（例如cy=545.9533 vs523.2619）；不能把默认或硬编码内参当作已标定参数。

### 14.4 frame220 官方与E3A行为

[frame220_palette_audit.json](evidence/frame220_palette_audit.json)：在真实00220.png上执行固定commit的pixel2category.f，验证 `[0,0,128]` 是palette index4、37像素，超出C20三个前景标签后被官方 `continue` 跳过。官方scene2frame也因所有非黑像素从background剔除而不会把这些像素生成为background点。现有E3A decoder则ValueError，行为仍未改变。

技术建议：保留全部已知区域，将未知像素设 `ignore_index=-1` 并传递pixel validity。不能仅把未知像素清零后当成负背景训练，也不能只有track级mask_valid而忽略逐像素排除。建议的诊断实现和测试放在审计脚本；没有修改正式decoder或Loss。

### 14.5 C13/T4 多物体目标

[multi_object_target_audit.json](evidence/multi_object_target_audit.json)：本地有56条C13/T4候选，按所需模态完整性、min(mask,objpose,right)、总文件数和字典序排序，选 **ZY20210800001/H1/C13/N12/S179/s03/T4**。有300 mask、300 objpose、254 Right MANO、RGB/Depth视频及16个Action事件，无Left MANO。无需回退到C7。

该序列RGB与Depth也已各顺序解码300帧，实际15FPS、1920×1080；Depth读取为单通道uint16。实测PTS和首帧dtype/shape在JSON的videos中，cut图像的时间对应基于这份实测编码流，而不是假定沿用C20元数据。

官方映射 palette1=Knife（semantic31、local instance1），2=Right Hand（47、2），3=Apple（46、3）。Apple独立区域在 **300/300帧** 可见。全部objpose却仅有 **id=2,label=knife**，不是mask1，也没有Apple pose。数字2在此处甚至与mask中手的编号相同，进一步证明不能按数字join。

真实cut事件区间 **[14.998,16.716]秒**；本地抽样225、250帧可见手持所标注的刀朝Apple区域执行切割，其他刀具在背景中。源事件properties均空、无实体字段。任务和图像支持cut阶段Knife为工具、Apple为受作用实体；Grasp/Pickup等阶段又可能针对Knife，故 **N12不能自动成为所有未来事件的目标**。

Apple是独立2D实体（PASS：类别定义和像素统计），持久物理身份仅PARTIAL，没有独立4D生成/碎片身份说明；与每个Action的target relation未标注。Apple可作为人工复核候选，**不能输出有效Target Entity GT**。`target_entity_valid=False, track_id=-1`。所有衍生图片仅本地保留，不公开。

### 14.6 监督门槛与证据等级

| 项目 | 状态 | 证据等级 |
|---|---|---|
| 原始frameId异常及非固定偏移 | PASS | 300原JSON重读及SHA256 |
| frameId业务语义/采集时间戳 | NEEDS_REVIEW | 仅模式推断，保持UNKNOWN |
| Chair pose/mask对应 | PARTIAL | 31实际box、300投影候选；非精确同步认证 |
| LOG解析与逆矩阵约定 | PASS | 固定Open3D源码、本地300组API读值 |
| 相机物理几何 | PARTIAL | 5帧真实静态投影诊断，无单位认证 |
| 官方4D前景offset规则 | PASS | 固定代码与实际静态ID；生成结果NOT_RUN |
| 完整官方4D生成 | BLOCKED / NOT_RUN | 输入/预处理/依赖缺失，未偷偷替换 |
| frame220官方skip行为 | PASS | 固定选择器在真实mask执行 |
| Apple独立mask | PASS | 官方类别表+300帧像素统计 |
| Apple事件目标GT | BLOCKED | 无明确事件→实体关系与matcher映射 |

必须保持invalid：未知像素的监督；无来源映射的未来Target；无接触真值的Interaction；未恢复校验的MANO Hand；未经物理单位/坐标验证的objpose 3D GT；遮挡/出画/缺标伪负例及未验证的动态实例/slot映射。Action/Time解析可保留现有有效性规则，但不能据此启用Target监督。

复现补审：先在独立目录检出上述官方commit，再运行 `python scripts/audit_hoi4d_identity_bridge.py --root "$HOI4D_ROOT" --official-root <pinned-checkout>`；投影诊断追加 `--camera-only`。运行前设置真实本地HOI4D_ROOT。代码仅写归档，图片进入local_only，不写数据集。测试命令为 `python scripts/archive_hoi4d_audit.py tests --followup`；原28项保留，新增9项索引/ignore/offset/多物体映射/target门控回归。新增合成测试与发布证据回归不是额外物理认证。
