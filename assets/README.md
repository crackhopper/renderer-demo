# 测试资产基线

本目录包含 renderer-demo 项目的所有测试资产。资产直接提交到仓库，无需外部下载或 git LFS。

## 资产清单

| 资产 | 位置 | 大小 | 用途 |
|---|---|---:|---|
| DamagedHelmet | `models/damaged_helmet/` | 3.7 MB | PBR 主测试模型 |
| Sponza | `models/sponza/` | 51 MB | Shadow / 多 mesh / culling 压力场景 |
| Stanford Bunny | `models/stanford_bunny/` | 2.3 MB | 经典 baseline 模型 |
| Viking Room (mesh) | `models/viking_room/` | 469 KB | 兼容旧 demo |
| Viking Room (textures) | `textures/viking_room/` | 1015 KB | 兼容旧 demo |
| Studio Small 03 HDR | `env/` | 6.4 MB | IBL 环境贴图 |

**总计**: ~65 MB

## 体积预算

- 上限: **100 MB**
- 当前: ~65 MB
- 如需裁剪，优先移除 Stanford Bunny；DamagedHelmet、Sponza、HDR 和 Viking Room 不可移除

## 禁止引入

- 4K 及以上 HDR
- Bistro、Cornell Box 或其他额外大场景
- git submodule 或外部下载脚本

## 下游需求

- `REQ-010`: 测试资产与 `assets/` 目录约定（本 REQ）
- `REQ-011`: 以 `models/damaged_helmet/` 作为 glTF 测试输入
- `REQ-019`: 以 DamagedHelmet、Sponza 作为 demo 场景资产
- `REQ-028`: 以 `env/studio_small_03_2k.hdr` 作为环境贴图输入

## 路径定位

代码中使用 `cdToWhereAssetsExist(subpath)` helper 定位资产：

```cpp
#include "core/utils/filesystem_tools.hpp"

if (cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf")) {
    // cwd 已切换到包含 assets/ 的目录
}
```
