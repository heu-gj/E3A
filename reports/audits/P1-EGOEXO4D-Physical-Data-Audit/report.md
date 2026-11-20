# P1-EGOEXO4D-Physical-Data-Audit

最终状态 **PARTIAL**。真实 Relations 和视频可读，但本地缺 EgoPose、Camera Pose、Raw Keystep 和 MPS，完整 Hand → Entity → Future Action 监督链路仍 **BLOCKED**。不能以解析器单元测试代替缺失模态的真实验证。

## 1. 执行前提与范围

已核对 PR #1 为MERGED，Squash Commit及基础/远端任务分支初始提交均为 `bade1b8d68205335ce039490ded919965216fe70`。初始工作区干净；当前分支 `audit/p1-egoexo-physical-data`。未修改原始数据、生产adapter、Backbone、Registers、Heads、Loss、Trainer；未启动训练、下载数据或安装依赖。

唯一结果目录为本目录。脚本位于 `scripts/audit_egoexo_physical.py`、`scripts/seal_egoexo_audit.py`，测试位于 `tests/test_egoexo_physical_audit.py`。本文中的数据路径均相对于配置的数据根，代码路径相对于仓库；本地绝对路径不发布。

## 2. 真实版本、目录与文件覆盖

`EGOEXO_ROOT`未设置，在此前M3T数据根的合理候选目录中定位数据。只递归检查该已定位数据集根，并检查父目录候选名称；没有扫描整个磁盘。文件清单证据：[dataset_inventory.json](evidence/dataset_inventory.json)。

实际 `raw/metadata.json` 的 **version="v2", date="240325"**，此项PASS源于字段，不是目录名推测。下载收据和完整下载manifest未找到，下载日期UNKNOWN。三个Relations文件的实际 `ds="20240311_20:16:25"`；这不能独立证明每份annotation的V1/V2修订号。Annotation发布revision保持UNKNOWN。

本地共 **11,785个文件：11,775个MP4、10个JSON**。实际JSON只有 `raw/{physical_setting,metadata,participants,captures,visual_objects,takes}.json` 和 `raw/annotations/{splits,relations_train,relations_val,relations_test}.json`。metadata有5,035条Take，1,310条有本地Aria RGB，1,137条有本地Exo，1,312条有匹配的Relations对象记录。

| 文件 | 实际bytes | 检查 |
|---|---:|---|
| raw/takes.json | 43,160,564 | 完整读取metadata，未发布原文件 |
| raw/metadata.json | 4,685 | version/date字段及SHA256 |
| raw/annotations/relations_train.json | 1,226,475,944 | 按Take流式读取 |
| raw/annotations/relations_val.json | 381,112,365 | 按Take流式读取 |
| raw/annotations/relations_test.json | 483,973,346 | 按Take流式读取 |

三个Relations split分别含1,157 /267 /366个Take；train有 **137个UID不在本地takes.json** 中，原因UNKNOWN，可能涉及metadata过滤或版本差异，不能猜测。候选仅使用真实UID交集；未跨名称强行关联。没有全量解码11,775个视频。

本地不存在EgoPose手部/相机JSON、Raw Keystep、keystep_benchmark或MPS/calibration/timesync实际文件。takes中的可下载trajectory/VRS字段不能代替实际文件。缺失模态的根结构、record数量、关节有效率和误差均记NULL/NOT_RUN，不填零。

用户已确认目前没有其他本地标注目录可提供。因此本次保留这些缺失门槛，不自动下载、不补造GT，也不以另一条Take或合成测试代替缺失模态。

已进一步核对选中Take的capture元数据：camera记录只有device/path等描述，没有K/E矩阵。takes虽声明has_trimmed_trajectory=true、has_trimmed_vrs=true，对应VRS、eye_gaze、trajectory实际路径均不存在；详见 [mps_availability.json](evidence/mps_availability.json)。不是把“发布了资产”误认为“本地已下载”。

## 3. 选择Take与完整性排序

[take_inventory.csv](evidence/take_inventory.csv) 覆盖5,035条metadata；[candidate_selection_summary.json](evidence/candidate_selection_summary.json) 记录split统计和前20候选。

没有一个本地Take覆盖所有六种模态。可用Aria+Exo且至少两个非手对象的候选有695条；按Aria非手mask记录总数降序、非手对象数量降序、UID字典序选择第一条：

- Take UID：`51fc36b3-e769-4617-b087-3826b280cad3`。
- Take Name：`georgiatech_cooking_01_01_2`。
- 场景：烹饪Tomato & Eggs；Relations train记录。
- Aria：`aria04_214-1`；4个Exo文件存在，实际检查`cam01`（metadata best_exo且有Relations）。
- 16条非手对象、11,179份Aria非手encoded mask、25个对象/相机记录；没有hand track。
- 所有候选同样缺手姿、相机姿态和Raw Keystep，不存在可用于这些模态的第二条更完整Take。

选择理由是可用模态与标注覆盖，不是将缺模态Take伪装为完整Take。

## 4. 现有Adapter逐项审查

已阅读 `src/e3a/data/{egoexo,actions,collate}.py`、`tests/test_egoexo_data.py`、`docs/implementation/{roadmap,training}.md`。详细发现：[adapter_review.json](evidence/adapter_review.json)。生产代码未改。

| 函数 | 代码事实 / 本次状态 |
|---|---|
| parse_egoexo_hand_pose_annotations | 仅取frame_records[0]，按joint name显式构造Right/Left与21关节；忽略后续records。多record风险有构造反例，但本地没有EgoPose，实际发生率UNKNOWN，不能称为已发现真实发布版丢标 |
| _resolve_camera_extrinsics | 支持frame-first和camera-first，要求finite[3,4]；代码本身不证明SO(3)、方向、物理单位 |
| decode_egoexo_relation_mask | 官方SDK解码并检查shape；本次70份真实样本与SDK直接输出一致 |
| build_egoexo_relation_clip_targets | 正/显式缺席/未标注三态；有像素才创建可匹配轨迹；本次两条真实轨迹三帧合同检查通过 |
| parse_egoexo_keystep_segments | 使用step_unique_id，秒区间排序；不查真实视频时长、taxonomy/vocabulary、V2任务区间；真实兼容性未运行 |
| build_egoexo_keystep_vocabulary | 从caller提供的train标注构造稳定ID映射；训练split来源及全局vocabulary成员资格仍需Dataset层验证 |
| build_egoexo_anticipation_target | min(start>cutoff)，time差值；未知词表只禁action，target始终invalid；只有单元测试证据 |
| resolve_egoexo_video_path | Aria别名逻辑正确选择metadata stream，但本地只有downscaled路径，当前require_exists会失败；这是已验证的实际读取缺口 |

`actions.py`只提供dataset前缀词表，不是跨数据集语义映射。`collate.py`只检查合同和padding，不负责metadata路径、像素变换、相机标定或事件目标推断。没有标注的零轨迹clip不能被当成全负场景。

## 5. EgoPose：BLOCKED / NOT_RUN

[ego_pose_coverage.json](evidence/ego_pose_coverage.json) 记录缺失范围。官方 [EgoPose说明](https://docs.ego-exo4d-data.org/annotations/ego_pose/) 描述frame-keyed records、annotation2D/3D、placement及按名称定义的左右手关节；本地没有对应文件，不能验证Root、metadata.frame_number、placement取值、num_views_for_3d分布、缺失关节或每帧record基数。

Manual2D、Automatic2D、Manual3D、Automatic3D统计均为 **NULL（缺文件）**，不是0%有效率。未输出任何真实恢复的 `[T,2,21,3]`。代码目标顺序为Right→Left，Wrist、Thumb、Index、Middle、Ring、Pinky；按名称赋值不是依赖JSON迭代顺序。只有构造测试支持该实现。

`min_views_for_3d`为必填策略参数，manual>=3与automatic=None是当前实现约定；需要实际发布路径、placement与官方baseline核对后才能应用。没有本地证据允许把manual和automatic混为一类。

## 6. Camera Geometry与手部重投影：BLOCKED / NOT_RUN

[camera_pose_validation.json](evidence/camera_pose_validation.json)、[hand_reprojection.json](evidence/hand_reprojection.json)。没有真实K、E、3D或对应2D joints，无法运行正向与逆向变换对照。每摄像机、Manual/Automatic分组的median、P90均为NULL，绝不生成0误差作为替代。

当前 `p_camera=R*p_world+t` 是待验证假设。没有矩阵便不能声称正交性/行列式通过；即使通过也不能替代重投影。文档示例K为3×3、E为3×4，正文另有统称3×4的描述，本地格式必须以真实文件为准。物理单位 **UNKNOWN**，source_to_model_units=1只保持原值，不证明米制。

Aria画面明显存在鱼眼外观；本地仅有低分辨率视频，不能凭外观恢复畸变参数、裁剪或相机模型。没有MPS文件，不从其他数据集或HOI4D借用标定。

## 7. Video / Frame Alignment：PARTIAL

证据：[video_alignment.json](evidence/video_alignment.json)、[cross_modal_frames.csv](evidence/cross_modal_frames.csv)。真实takes元数据root_dir与relative_path指向原始分辨率文件，实际文件在同Take的 `frame_aligned_videos/downscaled/448/`。

| Camera | 实际相对路径（Take目录后缀） | Resolution | FPS | Container frame count |
|---|---|---:|---:|---:|
| Aria | frame_aligned_videos/downscaled/448/aria04_214-1.mp4 | 448×448 | 30 | 31,048 |
| Exo | frame_aligned_videos/downscaled/448/cam01.mp4 | 796×448 | 30 | 31,048 |

两视频container duration均为1,034.933333秒，与takes.duration_sec一致。未全量顺序解码这两条长视频，帧数为container报告值；只对18个实际读取位置进行有界seek。**448目录不是所有视频宽度为448**，Exo实际796×448。

另以真实cmu_bike01_2 metadata验证用户指定的 `aria01_214-1 → aria01/rgb`：逻辑路径选择正确、其本地低分辨率文件frame0可读，但原分辨率metadata路径同样不存在。见 [resolver_alias_audit.json](evidence/resolver_alias_audit.json)；该额外Take只验证路径，不替代主Take空间审计。

每摄像机五个全时段样本frame为 **0、7762、15524、23285、31047**，实测PTS为0、258.733333、517.466667、776.166667、1034.9秒；read后的索引及PTS见CSV，样本PTS与idx/30一致。但这并不证明原始传感器从未丢帧或有独立时钟同步。

为查看mask另读其自己的源frame **0、7770、15510、23280、31020**；不是把邻近mask画到错误时间的RGB上。所有图像保留本地。Aria五处及Exo15510目视显示recipe、tomato、bowl、utensils等区域整体对应；拥挤标签和分辨率降低影响边界判断，没有像素级注册PASS。

Relations annotation_fps=1，而实际annotated frame keys每30帧一个；实测视频30FPS支持1Hz标注采样。**不能把frame number除以annotation_fps来取视频时间**。[relation_frame_cadence.json](evidence/relation_frame_cadence.json) 保留各轨迹实际差分。

本Take task_start_sec=.3505、task_end_sec=1034.16666；实际视频时长仍匹配完整Take，不是该任务区间之差。Raw Keystep缺失，无法验证V2区间对事件解释的具体影响，未擅自减去task_start或制造事件。

## 8. Relations Mask：解码/覆盖检查PASS，训练配准PARTIAL

实际Root为 `ds,annotations`；Take按UID索引，object_masks按对象名称再按camera分组。选择Take有16个对象、25个camera records。字段为annotation、annotation_metadata、annotation_fps、annotated_frames；encodedMask和原始annotator metadata只保留本地，不公开。

官方 [Relations说明](https://docs.ego-exo4d-data.org/annotations/relations/) 用annotated_frames区分不可见与未标注。本次检查所有25个记录的覆盖集合：encoded frame均在annotated_frames内，异常数0；显式缺席合计 **7,577个对象-相机-帧**。Aria16条、cam01八条、cam02一条。

实际调用 `ego4d.research.util.masks.decode_mask`，SDK版本 **1.7.3**，源模块SHA256在 [relation_mask_decode.json](evidence/relation_mask_decode.json)。审计Python原环境无ego4d，但已存在的下载环境有官方纯Python包，缺NumPy/pycocotools；只将该包路径附加到已有审计环境，使用其现有依赖。没有复制自制decoder，没有安装依赖。

18个读取位置共出现 **70份真实encoded records**，SDK与E3A输出逐像素一致、shape匹配，样本空mask=0。不是全部Take encoded masks的穷举检查。source mask为Aria1408×1408、Exo2160×3840，均不同于实际低分辨率视频；图中最近邻缩放仅用于诊断，scale_x/scale_y分别记录，未宣称crop或畸变已校准。

| 源状态 | 当前处理 |
|---|---|
| frame在annotated_frames内且有mask | 正mask |
| frame在annotated_frames内但无mask | 显式缺席，mask_valid=true、id=-1 |
| frame不在annotated_frames | 未标注，mask_valid=false；不是负样本 |

对真实salt/sugar container两轨迹、frame **300/16740/30990** 调用现有clip builder，输出 `[2,3,1408,1408]`。salt ids=`[0,0,-1]`、sugar ids=`[1,1,1]`，两轨迹三帧均mask_valid=true；末帧salt是有标注的缺席，不是缺标。IDs是此clip排序生成的局部编号，不是独立认证的跨场景持久身份。该源分辨率合同PASS不表示可直接配448图像训练。

## 9. Keystep：BLOCKED / NOT_RUN

Raw Keystep和keystep_benchmark均未找到，明确分别记录，未将两者混用。`keystep_timeline.csv`只有缺失状态行，UNKNOWN时间/ID，绝非合成事件。真实parser兼容、视频时长越界、重叠/间隙、vocabulary membership、taxonomy父子关系及所有事件前/等于/后的边界检查均未执行。

官方 [Keystep定义](https://docs.ego-exo4d-data.org/annotations/keystep/) 区分scenario内step_id和跨scenario step_unique_id；现有代码使用后者，方向与格式意图一致。但无法根据文档把本地不存在的事件判为PASS。严格未来规则与无未来/未知词表处理仅由原有单元测试支持。本次未生成实际训练词表、未读取未来视频作为模型输入，也没有模型运行。

## 10. Hand → Entity → Future Action与角色

没有EgoPose、选中Relations也没有hand track，所以不能建立实际手部3D到对象关系；视频中能看到手不等于接触GT。没有Raw Keystep或官方event→track映射，不能将对象名称或step description中的名词转成Target标签。

在所选视频中可研究的候选角色包括：spatula/measuring spoons为工具候选，bowl/cups/containers为容器候选，sliced tomato/scallions/omelette为被操作材料候选，recipe为说明载体；人是交互主体。**这些只是名字与目视场景支持的候选，事件级角色均UNKNOWN**，同一实体随时段可变，不能硬编码为GT。

Relations用同一对象名称组织多视角，但本次未独立验证跨视角身份；保持UNKNOWN。E3A第一版只把Aria帧作为时间轴，Exo用于诊断，不把同步相机依次拼成T。

## 11. 监督可用性

[supervision_readiness.json](evidence/supervision_readiness.json)。

| Task | 状态 | 门槛 |
|---|---|---|
| Entity | PARTIAL | 真实解码/覆盖可用；需显式downscaled resolver、图像mask一致变换和可见clip策略 |
| Hand | BLOCKED | 需真实manual/automatic EgoPose与关节投影 |
| Geometry | BLOCKED | 需真实camera/MPS、坐标方向、单位和畸变校验 |
| Interaction | BLOCKED | 缺可靠contact，保持interaction_valid=false |
| Target | BLOCKED | 缺事件目标、物体身份及匹配slot链路，保持target_entity_valid=false |
| Action | BLOCKED | Raw Keystep缺失，不能只用take.task_name构造逐事件GT |
| Time | BLOCKED | 无未来onset，take task_start/end不等价事件时间 |

因此当前可得到的真实监督证据仅支持**条件性的对象mask监督**；没有一条完整的本地Hand→Entity→Future Action监督链路。不应启动声称完整实体落地预判的训练。

## 12. 下一步建议与所需输入

先提供实际EgoPose manual/automatic手部及camera pose根目录、Raw Keystep annotations/taxonomy/vocabulary、可用MPS/calibration/timesync路径和版本manifest；不自动下载。这些是新增真实验证输入，不是请求凭空补GT。

保持生产代码不变，后续在独立实现任务中考虑：`egoexo.py`明确多record选择/拒绝政策，支持显式metadata video variant并核验downscaled路径；Dataset层统一图像/mask resize、裁剪与camera intrinsics，过滤未标注空clip；Raw Keystep校验vocabulary、taxonomy、视频时长和split。跨视角、接触、事件目标必须有额外真实关系证据才能启用。构造反例不能作为本地多record发生的证明。

## 13. 测试、复现与数据安全

测试详见 `logs/tests.json` 与 `logs/commands.txt`：**31 passed in 1.78s**，包含要求的 `tests/test_egoexo_data.py -v`、Action Vocabulary、Entity Targets、E3A Collation和新增审计辅助/证据回归。所有测试与真实数据测量分开记载；missing-data数值用NULL，状态NOT_RUN，不伪造重投影误差或21关节。审计运行使用现有环境，不是已验收的完整训练环境。

复现依次运行 `scripts/audit_egoexo_physical.py inventory --root <dataset-root>`、`audit --root <dataset-root> --sdk-site <existing-official-sdk-site>`，再用 `scripts/seal_egoexo_audit.py report` 和 `seal` 生成结构化报告/校验和。代码不修改源数据；Raw selected relation与图像只在Git忽略区。公开元数据均数据根相对路径，无Token、annotator ID、原视频/Mask或权重。

三个大型Relations源文件以及选中Take全部9个视频已按字节计算SHA256，记录在dataset_inventory.json；不是对11,775个视频做全量解码。所有公开归档另由SHA256SUMS.txt校验。代码用`python`运行；`fingerprints --root <dataset-root>`阶段可复现源字节哈希。

每项PASS都有文件与scope：metadata字段、25组集合检查、70份真实SDK解码、两轨迹三帧合同。视频配准仅PARTIAL；EgoPose、几何和真实Keystep均BLOCKED/NOT_RUN。当前任务停止于P1审计交付，不进入P2、不合并PR。
