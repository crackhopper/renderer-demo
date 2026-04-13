## 1. Header 声明扩展

- [x] 1.1 在 `src/core/utils/string_table.hpp` 新增 `enum class TypeTag : uint8_t { String, ShaderProgram, RenderState, VertexLayoutItem, VertexLayout, MeshRender, Skeleton, RenderPassEntry, MaterialRender, ObjectRender, PipelineKey }`
- [x] 1.2 在 `GlobalStringTable` public 区声明 `StringID Intern(std::string_view)`
- [x] 1.3 声明 `StringID compose(TypeTag, std::span<const StringID>)`
- [x] 1.4 声明 `struct Decomposed { TypeTag tag; std::vector<StringID> fields; }` 与 `std::optional<Decomposed> decompose(StringID) const`
- [x] 1.5 声明 `std::string toDebugString(StringID) const`
- [x] 1.6 private 区新增 `struct ComposedEntry { TypeTag tag; std::vector<StringID> fields; }` 与 `std::unordered_map<uint32_t, ComposedEntry> m_composedEntries`
- [x] 1.7 在类上方添加简短注释：`Intern` 专用于结构化代码路径；普通代码继续用 `StringID` 隐式构造

## 2. 源文件迁移与实现

- [x] 2.1 新建 `src/core/utils/string_table.cpp`，把 `getOrCreateID` / `getName` 的现有实现从 header 搬过来
- [x] 2.2 实现 `Intern(string_view)` 作为 `StringID{static_cast<uint32_t>(getOrCreateID(std::string{sv}))}` 的显式别名
- [x] 2.3 实现私有 `static std::string_view tagName(TypeTag)` 用 `switch` 表保证所有枚举值都有对应字符串；开启 `-Wswitch` 警告
- [x] 2.4 实现 `compose(tag, fields)`：构造规范化键 `"{tagName}({field0.id},{field1.id},...)"`，在写锁下调用 `getOrCreateID`，首次分配时同步写入 `m_composedEntries[id] = {tag, vector<StringID>{fields.begin(), fields.end()}}`
- [x] 2.5 实现 `decompose(id)`：读锁下查 `m_composedEntries`；命中返回；未命中但 `id` 是合法叶子时返回 `{TypeTag::String, {}}`；其他返回 `std::nullopt`
- [x] 2.6 实现 `toDebugString(id)`：转发到私有 `toDebugStringImpl(id, 0)`，递归深度超过 `kMaxDebugDepth = 16` 返回 `"<...>"`；叶子返回 `getName(id.id)`
- [x] 2.7 验证 `compose` 的写路径在同一把 `unique_lock` 内完成 `m_stringToId` / `m_idToString` / `m_composedEntries` 三张表的写入，保证原子性

## 3. 构建系统接入

- [x] 3.1 在 `src/core/utils/CMakeLists.txt`（或 `src/core/CMakeLists.txt`）把 `string_table.cpp` 加入对应 target 的源文件列表
- [x] 3.2 确认 `string_table.hpp` 的 include 没有因为从 header-only 迁走而丢失必要的 `<span>` / `<optional>` / `<string_view>` / `<unordered_map>` / `<vector>`
- [x] 3.3 运行 `cmake --build build` 验证全项目编译通过（特别是 `MaterialInstance` / `MaterialTemplate` 等既有 StringID 调用点）

## 4. 集成测试

- [x] 4.1 新建 `src/test/integration/test_string_table.cpp`，加入 `src/test/integration/CMakeLists.txt` 构建目标
- [x] 4.2 用例 1：`Intern("foo")` 与 `Intern("foo")` 返回相同 id（叶子去重）
- [x] 4.3 用例 2：`compose(PipelineKey, {a, b})` 两次调用返回相同 id（结构化去重）
- [x] 4.4 用例 3：`compose(PipelineKey, {a, b}) != compose(PipelineKey, {b, a})`（顺序敏感）
- [x] 4.5 用例 4：`decompose(compose(PipelineKey, {a, b}))->fields == {a, b}` 且 `tag == PipelineKey`（往返）
- [x] 4.6 用例 5：`toDebugString(compose(PipelineKey, {Intern("foo"), compose(VertexLayout, {Intern("pos")})}))` 等于 `"PipelineKey(foo, VertexLayout(pos))"`
- [x] 4.7 用例 6：并发 — 起 N 线程（≥8）每条循环 1000 次 `compose(PipelineKey, {a, b})` 同一输入，断言所有线程返回的 id 相同，断言 `m_composedEntries` 只有一条新增
- [x] 4.8 用例 7（额外）：`decompose(Intern("foo"))` 返回 `{TypeTag::String, {}}`；`decompose` 一个未分配的 id 返回 `std::nullopt`
- [x] 4.9 运行 `ctest --test-dir build --output-on-failure -R test_string_table`，全部通过

## 5. 验证与收尾

- [x] 5.1 运行 `openspec validate extend-string-table-compose --strict`，解决任何 spec 校验错误
- [x] 5.2 跑 `clang-format -i src/core/utils/string_table.{hpp,cpp} src/test/integration/test_string_table.cpp`
- [x] 5.3 运行 `cmake --build build && ctest --test-dir build --output-on-failure`，确认无回归
- [x] 5.4 把 `docs/requirements/006-string-table-compose.md` 的"实施状态"字段更新为已完成（但此时先不归档到 `finished/`，等 REQ-007 真正消费后再归档）
