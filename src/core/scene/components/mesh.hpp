#pragma once
#include "core/math/mat.hpp"
#include "core/resources/index_buffer.hpp"
#include "core/resources/vertex_buffer.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace LX_core {

template <typename VType> class Mesh : public IComponent {
  struct Token {};

public:
  Mesh(Token, VertexBufferPtr<VType> vertexBuffer, IndexBufferPtr indexBuffer)
      : vertexBuffer(vertexBuffer), indexBuffer(indexBuffer) {}

  static MeshPtr create(VertexBufferPtr<VType> vertexBuffer,
                        IndexBufferPtr indexBuffer) {
    return std::make_shared<Mesh>(Token(), vertexBuffer, indexBuffer);
  }

  VertexBufferPtr<VType> vertexBuffer;
  IndexBufferPtr indexBuffer;

  virtual std::vector<IRenderResourcePtr> getRenderResources() const override {
    return {std::dynamic_pointer_cast<IRenderResource>(vertexBuffer),
            std::dynamic_pointer_cast<IRenderResource>(indexBuffer)};
  }
};
template <typename VType> using MeshPtr = std::shared_ptr<Mesh<VType>>;

} // namespace LX_core