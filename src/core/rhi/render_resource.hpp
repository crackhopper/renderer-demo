#pragma once
#include "core/math/mat.hpp"
#include "core/math/vec.hpp"
#include "core/platform/types.hpp"
#include "core/utils/string_table.hpp"
#include <memory>
#include <vector>

namespace LX_core {
// 资源槽位类型，后端根据这个走不同的处理流程。
enum class ResourceType : u8 {
  None = 0,
  VertexBuffer,
  IndexBuffer,
  UniformBuffer,
  CombinedImageSampler,
  Special,
  Count
};

class IRenderResource {
public:
  virtual ~IRenderResource() = default;
  virtual ResourceType getType() const = 0;
  virtual const void *getRawData() const = 0;
  virtual u32 getByteSize() const = 0;

  /// Shader-side binding name this resource fills (e.g.,
  /// StringID("CameraUBO")). Empty StringID means "unnamed" — such resources
  /// are routed via the material path (textures) or not routed at all
  /// (vertex/index buffers).
  virtual StringID getBindingName() const { return StringID{}; }

  // 资源的唯一标识符，用于在渲染管线中查找资源
  // 直接使用地址作为句柄
  void *getResourceHandle() const { return (void *)this; }

  bool isDirty() const { return isDirty_; }
  void setDirty() { isDirty_ = true; }
  void clearDirty() { isDirty_ = false; }

private:
  bool isDirty_ = false;
};

using IRenderResourcePtr = std::shared_ptr<IRenderResource>;

// 确保与 GLSL 的 std140/std430 布局完全一致
struct alignas(16) PerDrawLayoutBase {
  Mat4f model = Mat4f::identity();
};

/// Transitional alias for the current engine-wide draw push-constant ABI.
using PerDrawLayout = PerDrawLayoutBase;

} // namespace LX_core
