#pragma once
#include <atomic>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace LX_core {

struct StringID;

/// TypeTag 标注一条 StringID 是叶子字符串还是某类结构化 compose 结果。
/// 叶子字符串（Intern / getOrCreateID）走 TypeTag::String；
/// 结构化 ID（compose）按具体类别标注。
enum class TypeTag : uint8_t {
  String = 0,
  ShaderProgram,
  RenderState,
  VertexLayoutItem,
  VertexLayout,
  MeshRender,
  Skeleton,
  MaterialPassDefinition,
  MaterialRender,
  ObjectRender,
  PipelineKey,
};

/// GlobalStringTable 是全局字符串 intern 表。
///
/// 两类 API：
///   1. 叶子字符串：getOrCreateID / Intern — 把普通字符串映射到 uint32 id
///   2. 结构化 ID：compose / decompose / toDebugString — 把若干子 StringID 按
///   TypeTag
///      结构化 intern 成一个新的 StringID，支持反向解构和人类可读渲染
///
/// Intern(string_view) 与 StringID(const std::string&) 隐式构造语义等价，
/// 专为结构化代码路径（嵌套 compose 参数包）提供无歧义的显式入口；
/// 普通代码继续用 StringID 隐式构造。
class GlobalStringTable {
public:
  static GlobalStringTable &get();

  uint32_t getOrCreateID(const std::string &name);
  const std::string &getName(uint32_t id) const;

  StringID Intern(std::string_view sv);

  StringID compose(TypeTag tag, std::span<const StringID> fields);

  struct Decomposed {
    TypeTag tag;
    std::vector<StringID> fields;
  };
  std::optional<Decomposed> decompose(StringID id) const;

  std::string toDebugString(StringID id) const;

private:
  GlobalStringTable();
  GlobalStringTable(const GlobalStringTable &) = delete;
  GlobalStringTable &operator=(const GlobalStringTable &) = delete;

  struct ComposedEntry {
    TypeTag tag;
    std::vector<StringID> fields;
  };

  std::string toDebugStringImpl(uint32_t id, int depth) const;

  uint32_t getOrCreateIDLocked(const std::string &name);

  mutable std::shared_mutex m_mutex;
  std::unordered_map<std::string, uint32_t> m_stringToId;
  std::vector<std::string> m_idToString;
  std::unordered_map<uint32_t, ComposedEntry> m_composedEntries;
  std::atomic<uint32_t> m_nextID;
};

struct StringID {
  uint32_t id = 0;

  StringID() = default;

  StringID(const char *name)
      : id(GlobalStringTable::get().getOrCreateID(name)) {}

  StringID(const std::string &name)
      : id(GlobalStringTable::get().getOrCreateID(name)) {}

  explicit StringID(uint32_t val) : id(val) {}

  bool operator==(const StringID &rhs) const { return id == rhs.id; }
  bool operator!=(const StringID &rhs) const { return id != rhs.id; }

  struct Hash {
    size_t operator()(const StringID &p) const {
      return static_cast<size_t>(p.id);
    }
  };
};

inline StringID MakeStringID(const std::string &name) { return StringID(name); }

} // namespace LX_core

namespace std {
template <>
struct hash<LX_core::StringID> {
  size_t operator()(const LX_core::StringID &p) const {
    return static_cast<size_t>(p.id);
  }
};
} // namespace std
