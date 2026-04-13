## Why

`GlobalStringTable` 目前只处理"叶子字符串 ↔ uint32 ID"映射，但即将落地的 REQ-007 需要把 pipeline identity 从"一堆 hash 混起来"改成"把若干子 `StringID` 结构化 intern 成一个新的 `StringID`"。这种两级嵌套（例如 `PipelineKey = compose(ObjectRenderSig, MaterialRenderSig)`）无法用现有 API 表达，也无法反向解构回人类可读形式用于调试。

## What Changes

- 新增 `TypeTag` 枚举，标注叶子字符串与每一类结构化 ID（`PipelineKey`、`ObjectRender`、`MaterialRender`、`VertexLayout` 等）
- 新增 `GlobalStringTable::Intern(string_view)` 显式叶子入口（与现有 `StringID` 隐式构造等价，为结构化代码路径消歧义）
- 新增 `GlobalStringTable::compose(TypeTag, span<const StringID>)` 结构化 intern API：同一 `(tag, fields)` 始终返回同一 `StringID`，顺序敏感
- 新增 `GlobalStringTable::decompose(StringID)` 反向解构，返回 `optional<{tag, fields}>`
- 新增 `GlobalStringTable::toDebugString(StringID)` 递归展开为 `"PipelineKey(foo, VertexLayout(pos))"` 形式（带深度保护）
- `m_stringToId` 继续作为规范化键的去重基础，新增 `m_composedEntries: unordered_map<uint32_t, ComposedEntry>` 记录结构化元数据
- 线程安全：`m_composedEntries` 并入现有 `shared_mutex`
- `StringID(const char*)` / `StringID(const std::string&)` 隐式构造 **保留不动**，避免对现有调用点的大规模改动

**非破坏性**：所有现有 API 不变，纯 additive 扩展。

## Capabilities

### New Capabilities
<!-- 无 -->

### Modified Capabilities
- `string-interning`: 扩展 `GlobalStringTable` 支持结构化 compose/decompose/toDebugString 与 `TypeTag` 语义，并提供显式 `Intern(string_view)` 入口

## Impact

- **代码**：`src/core/utils/string_table.hpp`（新增 API 与字段）；新建 `src/core/utils/string_table.cpp` 承载非 inline 实现（当前全在头文件里）；新建 `src/test/integration/test_string_table.cpp` 覆盖 6 条用例
- **API**：纯 additive — 现有 `getOrCreateID` / `getName` / `StringID` 构造不受影响
- **下游**：REQ-007（pipeline identity 基于 interning 重构）整体依赖本需求；REQ-005（统一材质系统）可选地用 `Intern` 替换部分隐式构造，不强制
- **构建**：`string_table` 从 header-only 变为 header + source，需要在 `src/core/utils/CMakeLists.txt` 加入新源文件
- **性能**：pipeline 数量通常在数十到数百，`m_composedEntries` 额外内存开销可忽略
