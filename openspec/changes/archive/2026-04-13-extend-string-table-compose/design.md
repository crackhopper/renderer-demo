## Context

`src/core/utils/string_table.hpp` 当前是一个 header-only 的 singleton：`GlobalStringTable` 维护 `unordered_map<string, uint32_t>` 与 `vector<string>` 双向映射，加 `shared_mutex`；`StringID` 是薄 wrapper，隐式从 `const char*` / `const std::string&` 构造，用于 `MaterialInstance` / `MaterialTemplate` 的绑定 key。

REQ-007 将重构 pipeline identity，把目前基于 FNV hash 混合的方式替换为"把若干子 `StringID` 结构化 intern 成一个新的 `StringID`"。该重构需要的基础设施是：

1. 给每类结构化 ID 一个显式 `TypeTag`（`PipelineKey`、`ObjectRender`、`MaterialRender`、`VertexLayout` 等）
2. `compose(tag, fields) → StringID`，顺序敏感且可去重
3. `decompose(id) → optional<{tag, fields}>`，用于调试日志与单元测试断言
4. `toDebugString(id)` 递归渲染成 `"PipelineKey(foo, VertexLayout(pos))"` 形式

本变更纯 additive，不动任何现有调用点；`StringID` 隐式构造保留，避免大规模迁移。

## Goals / Non-Goals

**Goals:**

- 新增 `TypeTag` 枚举 + `Intern` / `compose` / `decompose` / `toDebugString` 四个 API
- `compose` 去重确定性：同一 `(tag, fields)` 永远返回同一 `StringID`，无需额外碰撞处理
- 线程安全：所有新 API 与现有 API 共享同一 `shared_mutex`
- 循环保护：`toDebugString` 固定最大递归深度，越界返回 `"<...>"`
- 测试覆盖：新增 `src/test/integration/test_string_table.cpp`，6 条用例覆盖叶子去重、结构化去重、顺序敏感、往返、渲染、并发

**Non-Goals:**

- 不改变 `StringID` 的构造签名或 `getOrCreateID` / `getName` 的行为
- 不引入内容可寻址 hash 或 murmur/FNV 混合逻辑 — 完全靠字符串规范化复用现有 `m_stringToId`
- 不支持删除 intern — `GlobalStringTable` 继续是增长型容器
- 不强制现有 `MaterialInstance` / `MaterialTemplate` 调用点迁移到 `Intern(...)`
- 不把 REQ-007 的 pipeline identity 重构纳入本变更范围

## Decisions

### Decision 1：规范化键格式 — `{TagName}(id1,id2,...)`

**选择**：`compose(tag, fields)` 的内部去重键是字符串 `"{TagName}({field0.id},{field1.id},...)"`，直接喂进 `m_stringToId`。

**替代方案**：
- (a) 自定义 `struct ComposedKey { TypeTag; vector<uint32_t>; }` + 自定义 hash — 需要另一张 map，要处理 hash 碰撞。
- (b) 用 `std::hash` 把 tag 和 fields 混合成 64-bit — 有碰撞风险，且无法保证跨运行稳定。

**理由**：复用 `m_stringToId` 意味着 (i) 不需要新 map；(ii) 所有字段都是 `uint32_t`，生成的规范化键字面量本身就唯一；(iii) 自动获得线程安全 — 规范化键走现有 `getOrCreateID` 路径。字符串构造成本对 pipeline 数量（几十到几百）完全可忽略。

### Decision 2：`m_composedEntries` 单独一张 map 存 `(tag, fields)` 元数据

规范化键只保证去重，但没有记录原始 `tag` 与 `fields`。因此新增：

```cpp
struct ComposedEntry {
  TypeTag               tag;
  std::vector<StringID> fields;
};
std::unordered_map<uint32_t, ComposedEntry> m_composedEntries;
```

`compose` 命中已有规范化键时，直接返回对应 `StringID`（`m_composedEntries[id]` 已存在）。首次构造时，先通过 `getOrCreateID(canonicalKey)` 获得 `id`，再在同一把写锁下 `m_composedEntries[id] = {tag, fields}`。

### Decision 3：统一锁语义 — 共用现有 `shared_mutex`

**选择**：`m_composedEntries` 和 `m_stringToId` / `m_idToString` 共用 `m_mutex`。读走 `shared_lock`，写走 `unique_lock`。

**理由**：`compose` 必然同时访问 `m_stringToId`（查/写规范化键）与 `m_composedEntries`（写元数据）。拆两把锁带来的锁序问题远大于收益。pipeline intern 不是高频路径。

**含义**：当前 `getOrCreateID` 的 double-checked locking 模式需要保留；`compose` 的写路径要在持有写锁期间同时写入 `m_stringToId`、`m_idToString`、`m_composedEntries`，保证原子性。

### Decision 4：`StringID` 隐式构造保留，`Intern` 是显式等价入口

**选择**：不改 `StringID(const char*)` / `StringID(const std::string&)`，额外加 `Intern(string_view)`。

**替代方案**：把隐式构造改成 `explicit` — 会级联影响 `MaterialInstance::setFloat("u_Time", 1.0f)` 等大量调用点，破坏 REQ-005 已完成的工作。

**理由**：REQ-006 的动机只是让 REQ-007 的嵌套 compose 代码路径可读性更好 — 在 `compose(PipelineKey, {Intern("foo"), compose(VertexLayout, {Intern("pos")})})` 这种参数包展开里，`Intern` 比"裸字符串字面量走 StringID 隐式构造"意图更清晰。其他代码路径继续用隐式构造。

### Decision 5：`toDebugString` 深度保护

固定 `kMaxDebugDepth = 16`。理论上调用方只要不人为把一个 composed id 循环喂进它自己的 fields 就不会触发；但 defensive coding 更稳。实现上在 public `toDebugString` 内部转发到 private `toDebugStringImpl(id, depth)`，`depth > kMaxDebugDepth` 时返回 `"<...>"`。

### Decision 6：从 header-only 迁移到 header + source

**选择**：把实现搬到新建的 `src/core/utils/string_table.cpp`。头文件只保留类声明、`StringID` 定义、`TypeTag` 枚举、inline 小函数（`operator==` 等）。

**理由**：
- `compose` / `decompose` / `toDebugString` 实现包含 `std::span`、`std::optional`、字符串拼接、递归，放头文件会让每个 TU 都 include 这些重量级头文件
- `m_composedEntries` 的数据结构定义需要 `#include <unordered_map>` 和 `ComposedEntry` — 放 header 反而更乱
- CMake 侧改动很小：`src/core/utils/CMakeLists.txt` 里把 `string_table.cpp` 加入对应 target 即可

## Risks / Trade-offs

- **[风险] 规范化键的字面量冲突**：如果未来新增一个 `TagName` 恰好等于某个用户字符串（例如 `"PipelineKey"` 同时被某处当普通字符串 intern），会共享同一个 `StringID`。
  **Mitigation**：`compose` 的规范化键永远带括号 `PipelineKey(...)`，而叶子 `Intern("PipelineKey")` 不带括号 — 两者规范化字符串不同，天然隔离。仅需保证 `TagName` 不和业务字符串的 `"Name(args)"` 形式完全重合；实际业务字符串几乎不会是这种形态。如真需要硬隔离，可在规范化键前加前缀 `"#PipelineKey(...)"`，成本 1 个字节但更保险 — 留作 follow-up。

- **[风险] `m_composedEntries` 内存泄漏**：`GlobalStringTable` 不支持删除，`compose` 会持续增长。
  **Mitigation**：pipeline 数量本就由 shader variant 限定，量级在数十到数百；对长生命周期进程可忽略。如果未来出现爆炸增长（例如每帧生成新 pipeline），应当追查调用方而不是给 GlobalStringTable 加 GC。

- **[风险] `toDebugString` 在热路径上被调用**：字符串拼接 + 递归比纯 hash 昂贵。
  **Mitigation**：该 API 明确标注为仅用于日志与测试断言，不在帧循环里调用。REQ-007 的运行时路径使用 `StringID::id` 进行比较，不碰 `toDebugString`。

- **[权衡] `Intern(string_view)` 与 `StringID(const std::string&)` 语义重复**：两条路径做相同的事，维护者可能困惑该用哪个。
  **Mitigation**：在 `string_table.hpp` 的类上方加简短注释说明：`Intern` 用于"显式结构化代码路径"，其他地方继续用 `StringID` 隐式构造。REQ-007 的代码评审应当强化这个约定。

- **[权衡] TypeTag 静态表维护**：新增 `TypeTag` 值时必须同步更新 `tagName()` 静态映射，否则 `toDebugString` 会返回 `"<unknown>"`。
  **Mitigation**：把 `tagName()` 写成 `switch` 并开启 `-Wswitch` — 编译器会在新增枚举值未处理时报 warning。
