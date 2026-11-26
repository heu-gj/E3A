# H1 Review Follow-up — 执行报告

**最终状态：PARTIAL PASS。** 定向测试、全套测试与真实HOI4D单Batch Smoke通过，之前四项VGGT兼容测试均恢复；指定Ruff检查仍有1项I001，因此不满足“所有检查通过”的PASS条件。未进行完整1B模型训练或加载权重。

执行时间：2026-11-26T13:30:15.507608+08:00。

## 1. 精确版本、环境与工作区

- 验证源码Commit：`a5326995a37fc461a4ac90ab0b224ba32480c21a`；从`origin/feat/hoi4d-unified-supervision`建立独立detached worktree后复核，未改生产源码。
- PR：[E3A #14](https://github.com/heu-gj/E3A/pull/14)；此次只提交本报告，不合并、不修改main。
- VGGT-Ω子模块原始Commit：`48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`。为执行现有可选Reference测试，另初始化固定VidEoMT `40a4fa4a048b00f97f36b3eee66ab62a4e1bea23`。
- 按用户本轮补充指令使用新Conda **e3a**，不使用原.venv/IceVLM-Seg。Python3.10.19，PyTorch2.6.0+cu118，NumPy1.26.4，pytest9.2.0，Ruff0.17.6；`pip check`通过。本轮没有安装依赖、升级PyTorch/CUDA/torchvision或更改editable安装指向。
- `PYTHONPATH="$PWD/src:$PWD/third_party/vggt-omega"`明确优先导入本次worktree E3A和已打补丁的VGGT，而非新环境editable默认的runtime旧代码；实际导入路径已断言。
- GPU可见上下文中CUDA=True、device_count=4，4×RTX4090D，各24564MiB，驱动535.230.02。只读查询，未运行GPU模型。测试/Smoke显式`CUDA_VISIBLE_DEVICES=''`在CPU执行，避免GPU占用；此进程内CUDA不可见是主动隔离，不是宿主故障。
- 原工作区仍为`fix/egoexo-frame-coverage` / `5d47a25fff014e4568c9cb59b372233fd727ff1f`，干净；没有切换原分支。HAND-02、H2材料和其它worktree未修改。

环境检查（绝对路径脱敏）：

```json
{
  "python": "3.10.19 (main, Jun 17 2026, 19:53:31) [GCC 14.3.0]",
  "prefix": "${E3A_CONDA_ENV}",
  "user_site": false,
  "torch": "2.6.0+cu118",
  "numpy": "1.26.4",
  "pytest": "9.2.0",
  "ruff": "0.17.6",
  "cuda_runtime": "11.8",
  "cuda_available": true,
  "cuda_devices": 4,
  "gpu_query": "NVIDIA GeForce RTX 4090 D, 24564 MiB, 535.230.02\nNVIDIA GeForce RTX 4090 D, 24564 MiB, 535.230.02\nNVIDIA GeForce RTX 4090 D, 24564 MiB, 535.230.02\nNVIDIA GeForce RTX 4090 D, 24564 MiB, 535.230.02",
  "e3a_source": "${FOLLOWUP_WORKTREE}/src/e3a/__init__.py",
  "vggt_source": "${FOLLOWUP_WORKTREE}/third_party/vggt-omega/vggt_omega/__init__.py"
}
```

## 2. VGGT兼容补丁应用与保护

执行`git submodule update --init third_party/vggt-omega`并确认固定SHA后，运行现有`bash scripts/apply_patches.sh`：

```text
Applied: 0001-extra-tokens-and-dense-features.patch
Applied: 0002-official-forward-feature-api.patch
```

子模块仅有以下3文件修改，`diff --check`通过：

```text
 M vggt_omega/models/aggregator.py
 M vggt_omega/models/heads/dense_head.py
 M vggt_omega/models/vggt_omega.py
 3 files changed, 88 insertions(+), 14 deletions(-)
```

两补丁均通过`git apply --reverse --check`；E3A patch文件未改。三个类的`__init__` AST与原子模块HEAD完全一致，静态证据表明没有改变参数定义/数量/Checkpoint Shape接口。本轮没有实例化1B模型或加载Checkpoint，不能将静态证据写成实际严格加载成功。子模块HEAD未变，补丁仅本地工作区修改，不暂存或提交子模块指针/内部改动。

```json
{
  "e3a_commit": "a5326995a37fc461a4ac90ab0b224ba32480c21a",
  "vggt_commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
  "files": [
    {
      "file": "vggt_omega/models/aggregator.py",
      "constructor_ast_identical": true
    },
    {
      "file": "vggt_omega/models/heads/dense_head.py",
      "constructor_ast_identical": true
    },
    {
      "file": "vggt_omega/models/vggt_omega.py",
      "constructor_ast_identical": true
    }
  ],
  "both_patches_reverse_check": true,
  "actual_1b_instance_or_checkpoint_shape_check": "NOT_RUN; constructor AST equivalence is static evidence only, not actual load",
  "patch_files_unchanged": true
}
```

## 3. Ruff：未通过（1 error，退出码1）

完整检查文件：

```bash
ruff check \
  src/e3a/data/__init__.py src/e3a/data/hoi4d_dataset.py \
  src/e3a/data/hoi4d_supervision.py src/e3a/geometry/entity_pooling.py \
  src/e3a/engine/losses.py src/e3a/models/heads/geometry_head.py \
  scripts/smoke_test_hoi4d_dataset.py tests/test_hoi4d_dataset.py \
  tests/test_hoi4d_supervision.py tests/test_entity_pooling.py \
  tests/test_geometry_head.py tests/test_e3a_losses.py
```

仅剩`tests/test_hoi4d_dataset.py:1`的I001（import block不符合格式，建议移除导入后额外空行）。未运行`--fix`、未修改测试文件。相较上次6项，本次修复了其余问题，但不能据此忽略最后一项。

Ruff完整stdout/stderr：

```text
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
13 | |     _decode_mask_with_validity,
14 | |     _NumpyArraysOnlyUnpickler,
15 | |     collate_hoi4d_batch,
16 | |     make_hoi4d_dataloader,
17 | |     resolve_hoi4d_sequence_paths,
18 | | )
   | |_^
help: Organize imports
   |
19 |
   -
20 | SEQUENCE = "ZY20210800001/H1/C20/N30/S295/s04/T4"
   |

Found 1 error.
[*] 1 fixable with the `--fix` option.
```

## 4. 定向测试：140 passed /1 skipped /0 failed /0 errors

按任务给定的17个文件执行`python -m pytest -q`，附加仓库外JUnit输出；退出码0。为控制CPU资源使用OMP/MKL线程数2，未修改测试实现。

| 文件 | passed | failed | skipped | errors |
|---|---:|---:|---:|---:|
| test_hoi4d_supervision.py | 6 | 0 | 0 | 0 |
| test_geometry_head.py | 1 | 0 | 0 | 0 |
| test_entity_pooling.py | 5 | 0 | 0 | 0 |
| test_e3a_losses.py | 9 | 0 | 0 | 0 |
| test_hoi4d_dataset.py | 9 | 0 | 0 | 0 |
| test_hoi4d_data.py | 17 | 0 | 0 | 0 |
| test_e3a_collate.py | 2 | 0 | 0 | 0 |
| test_vggt_geometry.py | 1 | 0 | 0 | 0 |
| test_vggt_aggregator_patch.py | 2 | 0 | 0 | 0 |
| test_vggt_dense_head_patch.py | 2 | 0 | 0 | 0 |
| test_vggt_forward_patch.py | 3 | 0 | 0 | 0 |
| test_vggt_feature_adapter.py | 2 | 0 | 0 | 0 |
| test_vggt_omega_adapter.py | 3 | 0 | 0 | 0 |
| test_e3a_encoder.py | 2 | 0 | 0 | 0 |
| test_e3a_model.py | 1 | 0 | 0 | 0 |
| test_entity_supervision_validity.py | 63 | 0 | 0 | 0 |
| test_videomt_adapter.py | 12 | 0 | 1 | 0 |

### 4.1 之前四项VGGT失败已全部恢复

| 测试 | 本轮结果 |
|---|---|
| test_extra_tokens_join_register_attention_without_mutating_original_prefix | PASS |
| test_dense_head_feature_return_preserves_frame_chunking | PASS |
| test_extra_tokens_are_exposed_without_changing_camera_prefix | PASS |
| test_return_features_requires_enabled_dense_head | PASS |

相关测试还验证extra tokens的Frame/Inter-frame Attention接入、原Camera/Register prefix保持、DenseHead排除新增Prefix、Feature Return的Frame Chunking，以及默认forward与固定官方输出契约一致。它们使用轻量替身/缩小模块测试代码路径，不意味着实际1B权重精度或训练已通过。

### 4.2 Geometry修复验证

- `test_geometry_builder_never_marks_zero_support_valid_when_min_points_is_zero`通过：全部depth_valid=false，min_points=0仍没有有效目标/支持质量。
- `test_pooling_never_marks_zero_support_valid_when_min_mass_is_zero`通过：全无效深度的空实体不会因min_mass=0被标为有效。
- GeometryHead使用非单位90°旋转测试：world center[2,1,1]经R/t得到camera center[0,4,4]；world covariance diag[4,1,0]变为camera diag[1,4,0]，明确验证RΣRᵀ。
- 深度反投影、Mask/Pixel/Depth/Intrinsics门控及Center/Covariance解析值测试通过；真实物理坐标标定仍不在这些单测证明范围。
- GeometryLoss复用同一次mask criterion返回的GT Track→Slot映射；Scale-Invariant归一化、无效几何忽略、Track变化拒绝、非零Center梯度和Geometry权重进入loss_objective均通过。
- E3AEncoder Token注入/分离、统一多任务Forward与原Hand/Interaction/Anticipation Loss测试通过，属于单元测试，不等于真实全模型训练。

### 4.3 唯一Skip

`test_videomt_adapter.py::test_real_videomt_criterion_contract_when_optional_dependencies_are_installed`因缺Detectron2跳过。SciPy1.13.1已在新环境存在，固定上游Matcher/Criterion的CPU boundary-stub测试执行通过；它仍不是安装真实Detectron2的端到端Reference测试。未为消除Skip安装大型依赖。

## 5. 全套测试：237 passed /1 skipped /0 failed /0 errors

定向测试成功后执行`python -m pytest -q`，附加仓库外JUnit。退出码0，2个warning来自故意构造的重复ZIP成员合成测试，非真实数据异常。没有失败栈，因为没有失败测试；不存在删除测试/额外跳过四项旧失败的行为。

全套结果与上一轮对比：原230 passed /4 failed /2 skipped，本轮237 passed /0 failed /1 skipped。新增零支持点回归、补丁恢复及SciPy可用性影响测试数；不是对旧结果改写。

完整测试输出见附录A/B。

## 6. 真实HOI4D Smoke及有限性核对

仅使用`ZY20210800001/H1/C20/N30/S295/s04/T4`，执行：

```bash
python scripts/smoke_test_hoi4d_dataset.py \
  --root "$HOI4D_ROOT" \
  --sequence ZY20210800001/H1/C20/N30/S295/s04/T4 \
  --max-batches 1 --height 540 --width 960
```

退出码0、stderr为空。完整原始JSON如下，与上一轮Smoke JSON逐字段完全一致：

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

同一首Clip（索引0–7）另做一次只读Dataset/Collate检查：Center/Covariance/Point Mass均有限，8个Track-Frame有效，无效几何0，depth_units保持unknown，metric_valid=[false]。该额外检查不扫描其它序列或其它Clip。

内存中将同一Batch的depth_valid全部设false（不写原始数据），以min_points=0调用Builder：geometry_valid数量0、point_mass总和0、结果有限。该项是实际Batch上的门控探针，不是把修改过的输入当真实GT。生产逻辑、原深度和Mask没有更改。

```json
{
  "scope": "One additional inspection of same first8-frame clip, no other sequence scanned; no source modification or model execution",
  "sequence_id": "ZY20210800001/H1/C20/N30/S295/s04/T4",
  "frame_indices": [
    [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ]
  ],
  "geometry_finite": {
    "centers_camera": true,
    "covariance_camera": true,
    "point_mass": true
  },
  "geometry_valid": [
    [
      [
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
      ]
    ]
  ],
  "metric_valid": [
    false
  ],
  "units": [
    "unknown"
  ],
  "coordinate_frame": "camera",
  "normalization": "scale_invariant_per_clip",
  "point_mass": [
    [
      [
        34094.0,
        34185.0,
        34208.0,
        34283.0,
        34119.0,
        34205.0,
        34089.0,
        34152.0
      ]
    ]
  ],
  "invalid_geometry_track_frames": 0,
  "invalid_depth_pixels": 1033394,
  "unknown_mask_pixels": 0,
  "zero_support_probe": {
    "input": "same batch with depth_valid all false, in-memory only",
    "min_points": 0,
    "geometry_valid_count": 0,
    "point_mass_sum": 0.0,
    "finite": true
  }
}
```

有效Geometry8表示1条实体轨迹×8帧，不是8个独立物体。有效深度3,113,806像素、无效1,033,394像素；首Clip没有未知Palette像素，不能据此宣称其它Clip无未知点。没有重做frame220或全量SHA/完整性审计。深度点只被视为source-native编码单位；metric_valid=false不是“错误”，而是防止伪造米制GT的必要声明。

## 7. 必须保留的架构与研究限制

1. **Camera-frame Geometry Loss不监督Camera Pose**：监督对象是相机坐标下的Entity Center/Covariance，不是相机外参；不把计算中使用预测Camera解码与直接Pose GT监督混为一谈。
2. **未实现World-frame Trajectory Supervision**：跨帧世界轨迹和长期物理状态仍未验证。
3. **没有启用Metric Geometry Loss**：真实Depth单位unknown，按Clip尺度归一化，不声称米制。
4. **没有实现MANO FK**：原始MANO参数不是已验证相机3D关节；本次没有恢复或启用Hand3D GT。
5. **没有生成Contact GT**：共现/距离不能替代物理接触标签。
6. **未解决跨类别、跨序列持久物理实体身份**：当前GT Track/Slot匹配是Clip范围，不推广成长期Persistent Physical Identity。
7. **未进行完整1B模型训练**：无权重加载、实际1B初始化、Optimizer Step或训练Checkpoint；单元测试中的backward不代表实际共享Backbone梯度已验证。

## 8. 最终结论与交付

**PARTIAL PASS**：VGGT兼容补丁成功、定向及全套测试无失败、真实Smoke通过且单位声明安全；但Ruff退出码1，不能按任务要求宣布所有静态/代码/数据管线验收为PASS。剩余I001需要ChatGPT独立判断并修复；本轮不改源码掩盖错误。

报告仅新增此report.md。原始日志在仓库外保存；报告含完整脱敏输出、真实Smoke JSON和必要检查证据，没有原始数据、模型权重、Mask、视频或派生可视化。未提交子模块本地补丁修改，也未改E3A patch文件。

本地报告分支按指定名称`codex/h1-review-followup-20261125`（名称保留用户要求；执行时间如上）。推送前再次fetch并要求远端仍为a5326995；如果前移或拒绝则停止，不Force Push。最终报告Commit由交付消息给出，本文件源码基线不是自引用Commit。

等待独立审查，不合并PR #14，不进入H2。

## 附录A：定向pytest完整输出

```text
........................................................................ [ 51%]
...............................................................s.....    [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_videomt_adapter.py:234: could not import 'detectron2': No module named 'detectron2'
140 passed, 1 skipped in 6.96s
```

## 附录B：全套pytest完整输出

```text
........................................................................ [ 30%]
........................................................................ [ 60%]
........................................................................ [ 90%]
................s.....                                                   [100%]
=============================== warnings summary ===============================
tests/test_hoi4d_handpose_coverage.py::test_duplicate_archive_members_never_silently_override[False]
tests/test_hoi4d_handpose_coverage.py::test_duplicate_archive_members_never_silently_override[True]
  ${E3A_CONDA_ENV}/lib/python3.10/zipfile.py:1541: UserWarning: Duplicate name: 'wrapper/Hand_pose/handpose_left_hand/ZY20210800001/H1/C20/N18/S286/s01/T1/30.pickle'
    return self._open_to_write(zinfo, force_zip64=force_zip64)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
SKIPPED [1] tests/test_videomt_adapter.py:234: could not import 'detectron2': No module named 'detectron2'
237 passed, 1 skipped, 2 warnings in 6.41s
```
