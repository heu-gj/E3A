# H1 — HOI4D Unified Supervision Builder + Geometry Loss 执行报告

**最终结论：PARTIAL PASS。** 指定 H1 定向单测和真实单序列、单 Batch Smoke Test 均通过；指定 Ruff 检查有6项未修复问题，全套测试有4项失败、2项依赖跳过，因此不宣称全部验收通过或完整模型训练成功。

执行时间：2026-11-25T19:42:36.841576+08:00。

## 1. 执行基线、环境与边界

- 源码分支：`feat/hoi4d-unified-supervision`，PR [#14](https://github.com/heu-gj/E3A/pull/14)。
- 实际被验证 Commit：`10efd0782a2e9a573576cc30b491b10b5dbb9412`，已在 fetch 后及进入独立 detached worktree 后核对。
- VGGT-Ω 固定子模块：`48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`；使用 `git submodule update --init third_party/vggt-omega` 初始化，子模块无本地改动。
- 使用原 E3A `.venv`。通过 `PYTHONPATH="$PWD/src"` 明确导入本 worktree E3A，不误用用户当前 Ego-Exo4D 分支。
- Python3.10.19、PyTorch2.6.0+cpu、NumPy1.26.4、pytest9.2.0、Ruff0.17.6；CUDA unavailable、CUDA version=None、device_count=0。nvidia-smi返回9，驱动无法连接，GPU型号/显存UNKNOWN，不据此断言无物理GPU。
- 环境缺少Ruff与VGGT包；依本轮安装授权，仅安装Ruff及固定本地VGGT editable（均`--no-deps`；VGGT另加`--no-build-isolation`）。未重新安装E3A、更改原editable指向或升级PyTorch/CUDA。默认PyPI镜像无Ruff候选，改用官方PyPI安装成功。
- `pip check`检查前后均为`No broken requirements found.`。E3A与VGGT包导入成功；不代表1B实例初始化或权重成功加载。
- 全程没有修改模型/监督实现、应用VGGT兼容补丁、加载大型权重、进行Optimizer Step或完整模型训练。原用户工作区保持干净，分支与HEAD未变。没有执行SHA256审计、扫描全数据集或重新下载/解压数据。
- 本报告按本轮明确要求位于`reports/execution/`，不改变既有审计协议或重复归档旧报告。

环境原始检查（路径脱敏）：

```json
{
  "python": "3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]",
  "python_executable": "${E3A_VENV}/bin/python",
  "torch": "2.6.0+cpu",
  "numpy": "1.26.4",
  "cuda_available": false,
  "cuda_version": null,
  "cuda_device_count": 0,
  "pytest": "9.2.0",
  "ruff": "0.17.6",
  "e3a_import": "${H1_WORKTREE}/src/e3a/__init__.py",
  "vggt_import": "${H1_WORKTREE}/third_party/vggt-omega/vggt_omega/__init__.py",
  "e3a_commit": "10efd0782a2e9a573576cc30b491b10b5dbb9412",
  "vggt_commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
  "gpu_probe": {
    "exit_code": 9,
    "stdout": "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.\n\n",
    "stderr": ""
  }
}
```

## 2. Ruff静态检查

执行：

```bash
ruff check \
  src/e3a/data/hoi4d_supervision.py \
  src/e3a/data/hoi4d_dataset.py src/e3a/data/__init__.py \
  src/e3a/models/heads/geometry_head.py src/e3a/engine/losses.py \
  scripts/smoke_test_hoi4d_dataset.py tests/test_hoi4d_supervision.py \
  tests/test_geometry_head.py tests/test_e3a_losses.py tests/test_hoi4d_dataset.py
```

退出码1：**6 errors**，其中I001两项、E501四项。全部原样记录，未运行`--fix`。

| 文件/行 | 规则 | 问题 |
|---|---|---|
| src/e3a/data/__init__.py:3 | I001 | 导入排序 |
| src/e3a/data/hoi4d_dataset.py:186 | E501 | 103>100字符 |
| src/e3a/data/hoi4d_dataset.py:705 | E501 | 102>100字符 |
| src/e3a/engine/losses.py:110 | E501 | 103>100字符 |
| src/e3a/engine/losses.py:119 | E501 | 102>100字符 |
| tests/test_hoi4d_dataset.py:1 | I001 | 导入排序 |

完整静态输出见附录A。

## 3. 定向单元测试

```bash
python -m pytest -q \
  tests/test_hoi4d_supervision.py tests/test_geometry_head.py \
  tests/test_e3a_losses.py tests/test_hoi4d_dataset.py \
  tests/test_hoi4d_data.py tests/test_e3a_collate.py tests/test_vggt_geometry.py
```

实际执行带`--junitxml`将本地原始结果保存于仓库外。退出码0：**44 passed /0 failed /0 skipped /0 errors**。

| 测试文件 | passed | failed | skipped | errors |
|---|---:|---:|---:|---:|
| test_hoi4d_supervision.py | 5 | 0 | 0 | 0 |
| test_geometry_head.py | 1 | 0 | 0 | 0 |
| test_e3a_losses.py | 9 | 0 | 0 | 0 |
| test_hoi4d_dataset.py | 9 | 0 | 0 | 0 |
| test_hoi4d_data.py | 17 | 0 | 0 | 0 |
| test_e3a_collate.py | 2 | 0 | 0 | 0 |
| test_vggt_geometry.py | 1 | 0 | 0 | 0 |

这44项是代码/合成张量测试，不是全数据集物理标定：

- Builder反投影及加权Center/Covariance测试通过；使用K逆矩阵生成射线，并以z分量归一化后乘原始Depth。
- `depth_valid`、有限正Depth、`pixel_valid`、`mask_valid`、非负Track ID、`track_counts`共同门控。未知像素与无效Depth的排除有定向测试；intrinsics_valid=false返回无效目标，声明有效但奇异K抛错，未静默产生GT。
- GeometryHead代码执行`R*c_world+t`与`R*Σ_world*Rᵀ`，并将无效结果置零。现有Head单测使用identity R加非零平移；因此不能把它写成非单位旋转、畸变或真实相机配准的充分验证。公式静态核对与测试覆盖范围分开报告。
- GeometryLoss使用同一次mask criterion返回的`track_to_slot`，按源Track行与预测Slot对齐；只用双方geometry_valid及可见ID的交集。
- per-clip预测/目标各自用有效Center的median absolute z归一化，Covariance除以各自scale²，不将源Depth当米；合成尺度变化、无效帧、Track ID变化检查通过。
- `test_geometry_loss_reuses_clip_match_and_is_scale_invariant`执行backward，预测Center有非零梯度；`test_geometry_weight_is_included_in_joint_objective`验证权重2时总目标包含2×Geometry Loss。Matcher在这些联合Loss测试中是替身，不能据此宣称真实上游Matcher/1B共享Backbone已完成端到端验证。
- Hand/Visibility、Interaction、Anticipation原有联合Loss与缺失标注门控测试通过。无可靠真实Hand/Contact/Target GT的任务没有被本次Smoke新增为有效监督。

## 4. 真实HOI4D单Batch Smoke Test

只使用默认序列`ZY20210800001/H1/C20/N30/S295/s04/T4`，不更换、不枚举其它序列。输入分辨率540×960；读取首个8帧Clip（解码索引0–7），Batch Size1，num_workers默认0。

```bash
export HOI4D_ROOT=<local-HOI4D-root>
python scripts/smoke_test_hoi4d_dataset.py \
  --root "$HOI4D_ROOT" \
  --sequence ZY20210800001/H1/C20/N30/S295/s04/T4 \
  --max-batches 1 --height 540 --width 960
```

退出码0、stderr空。原始stdout JSON完整如下：

```json
{
  "sequence_id": "ZY20210800001/H1/C20/N30/S295/s04/T4",
  "num_dataset_clips": 38,
  "batches_read": 1,
  "expected_video_frames": 300,
  "unique_source_frames_seen": 8,
  "all_source_frames_covered": null,
  "entity_supervised_clips": 1,
  "invalid_or_unannotated_mask_pixels": 0,
  "mano_valid_hand_frames": 0,
  "objpose_records_read": 8,
  "depth_valid_pixels": 3113806,
  "geometry_valid_track_frames": 8,
  "geometry_metric_valid_clips": 0,
  "action_segments_loaded": 14,
  "action_supervised_clips": 0,
  "time_supervised_clips": 1,
  "first_batch_shapes": {
    "images": [
      1,
      8,
      3,
      540,
      960
    ],
    "depth": [
      1,
      8,
      1,
      540,
      960
    ],
    "entity_masks": [
      1,
      1,
      8,
      540,
      960
    ],
    "pixel_valid": [
      1,
      8,
      540,
      960
    ],
    "intrinsics": [
      1,
      3,
      3
    ],
    "geometry_centers_camera": [
      1,
      1,
      8,
      3
    ],
    "geometry_covariance_camera": [
      1,
      1,
      8,
      3,
      3
    ],
    "geometry_valid": [
      1,
      1,
      8
    ]
  },
  "physical_units_and_objpose_coordinate_frame": "remain unknown",
  "model_forward_or_backward": "not run"
}
```

结果解释：

- images `[1,8,3,540,960]`、depth `[1,8,1,540,960]`、entity_masks `[1,1,8,540,960]`、pixel_valid `[1,8,540,960]`、intrinsics `[1,3,3]`。
- geometry_centers_camera `[1,1,8,3]`、geometry_covariance_camera `[1,1,8,3,3]`、geometry_valid `[1,1,8]`，**8个有效Track-Frame**（1轨×8帧），不是8个独立物体。
- Geometry metric-valid clip数**0**，保持非米制。Dataset的depth_units为unknown，Builder保留source-native units，metric_valid显式false，Scale-Invariant Loss不等于公制深度恢复。
- 有效Depth像素3,113,806 /4,147,200，剩余**1,033,394个**无效Depth像素由depth_valid排除。Mask未知/未标注像素在此首Batch为0，不表示其它帧不存在未知像素，也不表示本轮重新验证frame220；未知像素排除由单测覆盖。
- 该Batch候选1轨×8帧全部geometry_valid，失效Track-Frame0。单个对象Mask内有多少Depth支持点不是本CLI输出字段，没有虚构该数量；不把全图有效Depth像素等同于物体支持点数。
- 手部有效原始MANO帧0、未来Action分类有效0（未提供词表）、Time有效1、objpose读取8。这些是首Batch实测，不是全序列或全数据集覆盖统计。
- Smoke JSON没有打印centers/covariance实际值或逐Tensor有限性断言；本轮不能把shape和非零有效计数当作独立3D精度/数值全审计。Builder对有效目标本身会检查有限性。
- 数据集共38个Clip，但本轮只读取1个、8个源帧；`all_source_frames_covered:null`明确表示没有做300帧完整审查。

## 5. 全套测试及停止点

定向测试通过后执行`python -m pytest -q`（附加本地JUnit输出）。实际结果：**230 passed /4 failed /2 skipped /0 errors，2 warnings**，退出码1。

| 失败测试 | 核心异常 |
|---|---|
| test_vggt_aggregator_patch.py::test_extra_tokens_join_register_attention_without_mutating_original_prefix | Aggregator.forward不接受extra_tokens |
| test_vggt_dense_head_patch.py::test_dense_head_feature_return_preserves_frame_chunking | DenseHead.forward不接受return_features |
| test_vggt_forward_patch.py::test_extra_tokens_are_exposed_without_changing_camera_prefix | VGGTOmega.forward不接受extra_tokens |
| test_vggt_forward_patch.py::test_return_features_requires_enabled_dense_head | VGGTOmega.forward不接受return_features |

固定子模块保持官方原Commit干净状态，仓库既有兼容patch没有应用，故这些接口测试失败。问题已观察到，本轮不修改模型源码、不应用patch掩盖失败，停止进一步运行或修复。它们不是Geometry Builder定向单测失败，但会阻止宣称完整E3A Forward接口就绪。

2个Skip均在test_videomt_adapter.py，分别缺Detectron2与SciPy；因此实际固定上游Reference Matcher/Criterion集成仍未验证。2个warning来自现有合成重复ZIP测试的标准库提示，与真实数据读取无关；日志中的Python基础标准库路径不代表使用或修改了其它Conda环境。

附录B保留全部失败错误栈与pytest原始摘要（只替换本机敏感路径并移除行尾空白）。

## 6. 验证结论与未解决项

| 问题 | 结论与证据范围 |
|---|---|
| 监督构建器是否工作 | 是：44项定向测试通过，真实单Batch生成8个有效几何Track-Frame |
| Depth反投影是否正确 | 合成解析值和代码路径通过；真实物理单位/深度标定不在本轮验证范围 |
| Mask与3D Geometry对应 | 同一GT Track行通过有效像素与Depth加权生成Camera Center/Covariance；不是持久物理实体身份或真实像素配准认证 |
| Geometry Loss是否可反传 | 合成Loss测试确认Center梯度及objective权重；没有真实模型Backbone端到端backward |
| 原有Loss是否回归正常 | 指定联合Loss9项、全套Entity validity63项通过；真实上游Reference2项跳过，完整模型接口4项失败不能忽略 |
| 尚未解决 | Ruff6项；VGGT兼容patch未应用；Detectron2/SciPy缺失；CUDA当前不可用；无真实完整模型训练或权重加载验证 |

**PARTIAL PASS**限定H1 supervision代码定向测试和真实reader/builder smoke。完整静态/全套测试验收没有通过；应由ChatGPT独立决定格式修复与兼容补丁准备，不由本执行任务擅自改实现。没有进行1B初始化、权重加载、Optimizer Step、Checkpoint保存或完整模型训练；不进入H2。

## 7. 交付与文件发布边界

仅新增本report.md，包含脱敏日志及真实Smoke的统计JSON；没有复制原始Mask、Depth、视频、Pickle或权重，没有进行数据SHA256审计。所有源代码与测试保持被验证Commit原样。

本地报告分支按要求为`codex/h1-execution-report`，推送目标为`feat/hoi4d-unified-supervision`；不创建第二个PR、不修改main、不合并PR #14。最终报告提交SHA在交付消息给出，本文件记录的是被验证的源码Commit，避免自引用提交哈希。

## 附录A：Ruff完整输出

```text
I001 [*] Import block is un-sorted or un-formatted
  --> src/e3a/data/__init__.py:3:1
   |
 1 |   """E3A data batching and source-annotation utilities."""
 2 |
 3 | / from e3a.data.actions import action_label_key, build_dataset_prefixed_action_vocabulary
 4 | | from e3a.data.collate import collate_e3a_batch, collate_entity_targets
 5 | | from e3a.data.egoexo import (
 6 | |     build_egoexo_anticipation_target,
 7 | |     build_egoexo_keystep_vocabulary,
 8 | |     build_egoexo_relation_clip_targets,
 9 | |     load_egoexo_keystep_segments,
10 | |     parse_egoexo_keystep_segments,
11 | |     load_egoexo_hand_pose_annotations,
12 | |     parse_egoexo_hand_pose_annotations,
13 | |     resolve_egoexo_video_path,
14 | | )
15 | | from e3a.data.hoi4d import (
16 | |     build_hoi4d_action_vocabulary,
17 | |     build_hoi4d_anticipation_target,
18 | |     build_hoi4d_entity_clip_targets,
19 | |     decode_hoi4d_motion_mask,
20 | |     hoi4d_color_palette,
21 | |     load_hoi4d_action_segments,
22 | |     parse_hoi4d_action_segments,
23 | | )
24 | | from e3a.data.hoi4d_dataset import (
25 | |     HOI4DDataset,
26 | |     HOI4DSequencePaths,
27 | |     collate_hoi4d_batch,
28 | |     make_hoi4d_dataloader,
29 | |     resolve_hoi4d_sequence_paths,
30 | | )
31 | | from e3a.data.hoi4d_supervision import (
32 | |     HOI4DSupervisionBuilder,
33 | |     build_hoi4d_geometry_targets,
34 | | )
   | |_^
35 |
36 |   __all__ = [
   |
help: Organize imports
   |
8  |     build_egoexo_relation_clip_targets,
9  +     load_egoexo_hand_pose_annotations,
10 |     load_egoexo_keystep_segments,
11 +     parse_egoexo_hand_pose_annotations,
12 |     parse_egoexo_keystep_segments,
   -     load_egoexo_hand_pose_annotations,
   -     parse_egoexo_hand_pose_annotations,
13 |     resolve_egoexo_video_path,
   |

E501 Line too long (103 > 100)
   --> src/e3a/data/hoi4d_dataset.py:186:101
    |
184 |                 if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
185 |                     raise TypeError(
186 |                         f"HOI4D RGB must decode as uint8 BGR [H,W,3], got {value.shape}/{value.dtype}."
    |                                                                                                     ^^^
187 |                     )
188 |                 value = np.ascontiguousarray(value[..., ::-1])
    |

E501 Line too long (102 > 100)
   --> src/e3a/data/hoi4d_dataset.py:705:101
    |
703 |                 "objpose_units": "unknown",
704 |                 "depth_units": "unknown",
705 |                 "mano_status": "raw parameters and 2D keypoints only; no MANO forward reconstruction",
    |                                                                                                     ^^
706 |             },
707 |         }
    |

E501 Line too long (103 > 100)
   --> src/e3a/engine/losses.py:110:101
    |
108 |         if predicted_covariance.shape != (batch_size, frames, slots, 3, 3):
109 |             raise ValueError("Predicted entity covariance must have shape [B,T,K,3,3].")
110 |         if predicted_valid.shape != (batch_size, frames, slots) or predicted_valid.dtype != torch.bool:
    |                                                                                                     ^^^
111 |             raise TypeError("entity_geometry_valid must be bool [B,T,K].")
112 |         if target_centers.ndim != 4 or target_centers.shape[0] != batch_size:
    |

E501 Line too long (102 > 100)
   --> src/e3a/engine/losses.py:119:101
    |
117 |         if target_covariance.shape != (batch_size, max_tracks, frames, 3, 3):
118 |             raise ValueError("Target covariance must have shape [B,N,T,3,3].")
119 |         if target_valid.shape != (batch_size, max_tracks, frames) or target_valid.dtype != torch.bool:
    |                                                                                                     ^^
120 |             raise TypeError("geometry_targets.valid must be bool [B,N,T].")
121 |         if ids.shape != (batch_size, max_tracks, frames):
    |

I001 [*] Import block is un-sorted or un-formatted
  --> tests/test_hoi4d_dataset.py:1:1
   |
 1 | / import json
 2 | | import pickle
 3 | | from pathlib import Path
 4 | |
 5 | | import cv2
 6 | | import numpy as np
 7 | | import pytest
 8 | | from PIL import Image
 9 | |
10 | | from e3a.data.hoi4d import hoi4d_color_palette
11 | | from e3a.data.hoi4d_dataset import (
12 | |     HOI4DDataset,
13 | |     _NumpyArraysOnlyUnpickler,
14 | |     _decode_mask_with_validity,
15 | |     collate_hoi4d_batch,
16 | |     make_hoi4d_dataloader,
17 | |     resolve_hoi4d_sequence_paths,
18 | | )
   | |_^
help: Organize imports
   |
12 |     HOI4DDataset,
13 +     _decode_mask_with_validity,
14 |     _NumpyArraysOnlyUnpickler,
   -     _decode_mask_with_validity,
15 |     collate_hoi4d_batch,
16 |     make_hoi4d_dataloader,
17 |     resolve_hoi4d_sequence_paths,
18 | )
   -
19 |
   |

Found 6 errors.
[*] 2 fixable with `--fix`.
```

## 附录B：全套pytest完整输出及错误栈

```text
........................................................................ [ 30%]
........................................................................ [ 61%]
..................................................................F.F... [ 91%]
.FF...........s....s                                                     [100%]
=================================== FAILURES ===================================
__ test_extra_tokens_join_register_attention_without_mutating_original_prefix __

    def test_extra_tokens_join_register_attention_without_mutating_original_prefix():
        aggregator = make_lightweight_aggregator()
        images = torch.zeros(1, 2, 3, 16, 16)
        extra_tokens = torch.full((1, 2, 2, 4), 5.0, requires_grad=True)

>       outputs, expanded_prefix = aggregator(images, extra_tokens=extra_tokens)

tests/test_vggt_aggregator_patch.py:49:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1739: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = Aggregator(
  (patch_embed): FakePatchEmbed()
  (rope_embed): FakeRope()
  (frame_blocks): ModuleList(
    (0): AddOneBlock()
  )
  (inter_frame_blocks): ModuleList(
    (0): AddOneBlock()
  )
)
args = (tensor([[[[[0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0... 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.],
           [0., 0., 0.,  ..., 0., 0., 0.]]]]]),)
kwargs = {'extra_tokens': tensor([[[[5., 5., 5., 5.],
          [5., 5., 5., 5.]],

         [[5., 5., 5., 5.],
          [5., 5., 5., 5.]]]], requires_grad=True)}

    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
>           return forward_call(*args, **kwargs)
E           TypeError: Aggregator.forward() got an unexpected keyword argument 'extra_tokens'

${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1750: TypeError
___________ test_dense_head_feature_return_preserves_frame_chunking ____________

    def test_dense_head_feature_return_preserves_frame_chunking():
        head, calls = make_lightweight_dense_head()
        images = torch.rand(1, 5, 3, 16, 16)

>       depth, confidence, fused = head(
            [None],
            images,
            patch_token_start=2,
            frames_chunk_size=2,
            return_features=True,
        )

tests/test_vggt_dense_head_patch.py:43:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1739: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = DenseHead()
args = ([None], tensor([[[[[0.4629, 0.7283, 0.0060,  ..., 0.5807, 0.6260, 0.4240],
           [0.7590, 0.9671, 0.0131,  ..., ... 0.2690, 0.0141,  ..., 0.9528, 0.7082, 0.2951],
           [0.7191, 0.2756, 0.3804,  ..., 0.7404, 0.2112, 0.9830]]]]]))
kwargs = {'patch_token_start': 2, 'frames_chunk_size': 2, 'return_features': True}

    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
>           return forward_call(*args, **kwargs)
E           TypeError: DenseHead.forward() got an unexpected keyword argument 'return_features'

${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1750: TypeError
_________ test_extra_tokens_are_exposed_without_changing_camera_prefix _________

    def test_extra_tokens_are_exposed_without_changing_camera_prefix():
        model = make_model().eval()
        images = torch.rand(1, 2, 3, 32, 32)
        extra_tokens = torch.rand(1, 2, 3, 4)

>       output = model(images, extra_tokens=extra_tokens, return_features=True)

tests/test_vggt_forward_patch.py:127:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1739: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = VGGTOmega(
  (aggregator): FakeAggregator()
  (camera_head): FakeCameraHead()
  (dense_head): FakeDenseHead()
)
args = (tensor([[[[[0.2156, 0.5099, 0.7169,  ..., 0.6594, 0.7816, 0.6332],
           [0.7377, 0.7581, 0.3279,  ..., 0.1127, ...0.9575, 0.0697,  ..., 0.8683, 0.5430, 0.3998],
           [0.2002, 0.5854, 0.4884,  ..., 0.7651, 0.9946, 0.7837]]]]]),)
kwargs = {'extra_tokens': tensor([[[[0.9749, 0.8450, 0.3721, 0.7081],
          [0.1226, 0.5158, 0.1932, 0.7305],
          [0....],
          [0.1390, 0.2485, 0.3205, 0.0924],
          [0.2259, 0.6888, 0.8738, 0.3772]]]]), 'return_features': True}

    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
>           return forward_call(*args, **kwargs)
E           TypeError: VGGTOmega.forward() got an unexpected keyword argument 'extra_tokens'

${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1750: TypeError
_______________ test_return_features_requires_enabled_dense_head _______________

    def test_return_features_requires_enabled_dense_head():
        model = make_model()
        model.dense_head = None
        try:
>           model(torch.rand(1, 2, 3, 32, 32), return_features=True)

tests/test_vggt_forward_patch.py:143:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1739: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = VGGTOmega(
  (aggregator): FakeAggregator()
  (camera_head): FakeCameraHead()
  (dense_head): None
)
args = (tensor([[[[[7.8262e-01, 3.8291e-01, 1.6017e-01,  ..., 4.0651e-01,
            3.1496e-01, 6.5849e-01],
           [9....6928e-01],
           [8.1416e-01, 2.4821e-01, 3.1586e-01,  ..., 7.8443e-01,
            7.0155e-01, 9.9481e-01]]]]]),)
kwargs = {'return_features': True}

    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
>           return forward_call(*args, **kwargs)
E           TypeError: VGGTOmega.forward() got an unexpected keyword argument 'return_features'

${E3A_VENV}/lib/python3.10/site-packages/torch/nn/modules/module.py:1750: TypeError
=============================== warnings summary ===============================
tests/test_hoi4d_handpose_coverage.py::test_duplicate_archive_members_never_silently_override[False]
tests/test_hoi4d_handpose_coverage.py::test_duplicate_archive_members_never_silently_override[True]
  ${PYTHON_BASE_STDLIB}/lib/python3.10/zipfile.py:1541: UserWarning: Duplicate name: 'wrapper/Hand_pose/handpose_left_hand/ZY20210800001/H1/C20/N18/S286/s01/T1/30.pickle'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
SKIPPED [1] tests/test_videomt_adapter.py:234: could not import 'detectron2': No module named 'detectron2'
SKIPPED [1] tests/test_videomt_adapter.py:467: could not import 'scipy': No module named 'scipy'
FAILED tests/test_vggt_aggregator_patch.py::test_extra_tokens_join_register_attention_without_mutating_original_prefix
FAILED tests/test_vggt_dense_head_patch.py::test_dense_head_feature_return_preserves_frame_chunking
FAILED tests/test_vggt_forward_patch.py::test_extra_tokens_are_exposed_without_changing_camera_prefix
FAILED tests/test_vggt_forward_patch.py::test_return_features_requires_enabled_dense_head
4 failed, 230 passed, 2 skipped, 2 warnings in 6.26s
```

## 附录C：定向pytest完整输出

```text
............................................                             [100%]
44 passed in 3.50s
```
