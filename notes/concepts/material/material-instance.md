# MaterialInstance：运行时状态

如果 `MaterialTemplate` 是菜谱，`MaterialInstance` 就是端上桌的那道菜。它是 scene、render queue 和 backend 真正持有的材质对象。

## 它保存了什么

| 字段 | 职责 |
|------|------|
| `m_template` | 指向所属的菜谱 |
| `m_bufferSlots` | 材质参数的 byte buffer（每个 material-owned UBO/SSBO 一个 slot） |
| `m_textures` | 全局默认纹理资源 |
| `m_passOverrides` | per-pass 的参数和纹理覆写 |
| `m_enabledPasses` | 当前启用的 pass 子集 |
| `m_passStateListeners` | pass 开关变化时通知 scene 的回调 |

## Buffer Slot 是怎样构造出来的

Instance 构造时，从 enabled pass 的 shader 反射里收集所有 material-owned buffer binding：

1. 遍历每个 enabled pass 的 `getMaterialBindings(pass)`
2. 对 `UniformBuffer` 和 `StorageBuffer` 类型的 binding 创建一个 `MaterialBufferSlot`
3. 每个 slot 有自己的 byte buffer（零初始化）、dirty 标记和 `IRenderResource` 包装器
4. 同名 buffer binding 跨 pass 必须布局一致，否则 assert

不支持的 descriptor 类型（如 standalone `Sampler`）会直接 FATAL。

## 写参数：两种方式

**推荐方式**——按 binding 名 + member 名精确定位：

```cpp
mat->setParameter(StringID("MaterialUBO"), StringID("roughness"), 0.5f);
```

**便利方式**——只按 member 名（单 buffer 时自动定位，多 buffer 时 assert）：

```cpp
mat->setFloat(StringID("roughness"), 0.5f);
```

底层都走同一条路径：定位 buffer slot → 查反射 member → 验证类型 → `memcpy` 到 offset。

纹理绑定：

```cpp
mat->setTexture(StringID("albedoMap"), textureSampler);
```

## Per-Pass 参数覆写

有时候同一个参数在不同 pass 下需要不同的值。比如 Forward pass 开启法线贴图但 Shadow pass 关闭。

```cpp
mat->setParameter(Pass_Forward, StringID("MaterialUBO"), StringID("enableNormal"), 1);
mat->setParameter(Pass_Shadow, StringID("MaterialUBO"), StringID("enableNormal"), 0);
```

per-pass 覆写会复制一份全局 buffer slot 到 `m_passOverrides[pass]`，之后对该 pass 使用独立的 buffer。

在 YAML 里对应 `passes.<pass>.parameters`：

```yaml
parameters:                          # 全局默认
  MaterialUBO.enableNormal: 0

passes:
  Forward:
    parameters:                      # Forward pass 覆写
      MaterialUBO.enableNormal: 1
```

## Pass 开关：为什么是结构变化

```cpp
mat->setPassEnabled(Pass_Shadow, false);
```

pass 开关和普通参数写入不同——它影响的不只是 draw 值，而是：

- 这个对象参加哪些 pass
- SceneNode 需要校验哪些 pass
- descriptor 资源的 pass 维度视图

因此 pass 开关变化会通过 listener 通知 scene，触发引用该材质的节点重验证。`setFloat` / `setTexture` / `syncGpuData()` 不走这条传播链。

## Descriptor 资源的收集

`getDescriptorResources(pass)` 按目标 pass 的反射 bindings 收集材质资源：

1. 枚举该 pass 的 material-owned bindings
2. buffer binding → 优先取 pass override slot，没有则取全局 slot
3. 纹理 binding → 优先取 pass override texture，没有则取全局纹理
4. 按 `(set << 16 | binding)` 排序输出

这保证了不同 pass 可以看到不同的参数默认值，而不会被"最后一次写入"互相覆盖。

## 和 YAML 的对应

```yaml
parameters:                          # → 全局 buffer slot 默认值
  MaterialUBO.baseColor: [0.8, 0.8, 0.8]
  MaterialUBO.shininess: 12.0

resources:                           # → 全局默认纹理
  albedoMap: white

passes:
  Forward:
    parameters:                      # → m_passOverrides[Forward].bufferSlots
      MaterialUBO.enableNormal: 1
    resources:                       # → m_passOverrides[Forward].textures
      normalMap: "textures/brick_normal.png"
```

## 继续阅读

- 代码入口：[material_instance.hpp](../../../src/core/asset/material_instance.hpp)
- 实现层设计：[../../subsystems/material-system.md](../../subsystems/material-system.md)
