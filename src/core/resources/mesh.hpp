#pragma once
#include "core/math/mat.hpp"
#include "core/resources/index_buffer.hpp"
#include "core/resources/vertex_buffer.hpp"
#include "core/scene/components/base.hpp"
// 建议引入一个简单的 AABB 结构用于剔除和碰撞
#include "core/math/bounds.hpp"
#include <assert.h>
#include <memory>
#include <vector>

namespace LX_core {

class Mesh {
  struct Token {};

public:
  Mesh(Token, VertexBufferPtr vb, IndexBufferPtr ib)
      : m_vertexBuffer(vb), m_indexBuffer(ib) {
    assert(m_vertexBuffer && "Mesh: VertexBuffer cannot be null!");
    assert(m_indexBuffer && "Mesh: IndexBuffer cannot be null!");
  }

  static std::shared_ptr<Mesh> create(VertexBufferPtr vb, IndexBufferPtr ib) {
    return std::make_shared<Mesh>(Token(), vb, ib);
  }

  // --- 只读访问器 ---
  const VertexBufferPtr &getVertexBuffer() const { return m_vertexBuffer; }
  const IndexBufferPtr &getIndexBuffer() const { return m_indexBuffer; }

  // --- 快捷信息接口 ---
  uint32_t getVertexCount() const { return m_vertexBuffer->getVertexCount(); }
  uint32_t getIndexCount() const {
    return (uint32_t)m_indexBuffer->indexCount();
  }

  /**
   * @brief 获取此 Mesh 的组合布局哈希
   * 用于在渲染管线中快速匹配或查找兼容的 PSO
   */
  size_t getLayoutHash() const {
    size_t hash = m_vertexBuffer->getLayoutHash();
    hash ^=
        m_indexBuffer->getLayoutHash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }

  const VertexLayout &getVertexLayout() const {
    return m_vertexBuffer->getLayout();
  }
  PrimitiveTopology getPrimitiveTopology() const {
    return m_indexBuffer->getTopology();
  }

  // --- 包围盒相关 (后续做视锥体剔除极其重要) ---
  void setBounds(const BoundingBox &box) { m_bounds = box; }
  const BoundingBox &getBounds() const { return m_bounds; }

  // --- 资源更新 ---
  void setVertexBuffer(VertexBufferPtr vb) {
    assert(vb);
    m_vertexBuffer = vb;
  }

  void setIndexBuffer(IndexBufferPtr ib) {
    assert(ib);
    m_indexBuffer = ib;
  }

  // 注意参数强制右值引用。
  // 因为在内部构造 VertexBuffer 时，会复用对应的指针。
  template <typename VType>
  static MeshPtr createWithBounds(std::vector<VType> &&vertices,
                                  std::vector<uint32_t> &&indices) {
    // 自动扫描顶点位置计算 Bounds
    BoundingBox bounds;
    for (const auto &v : vertices) {
      bounds.merge(v.pos); // 假设 VType 都有 .pos 属性
    }
    mesh->setBounds(bounds);

    auto vb = VertexFactory::create(std::move(vertices));
    auto ib = IndexBuffer::create(std::move(indices));
    auto mesh = create(vb, ib);

    return mesh;
  }

private:
  VertexBufferPtr m_vertexBuffer;
  IndexBufferPtr m_indexBuffer;

  BoundingBox m_bounds; // 存储该 Mesh 局部坐标系下的包围盒
};

using MeshPtr = std::shared_ptr<Mesh>;

} // namespace LX_core