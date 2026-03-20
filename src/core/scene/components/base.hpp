#pragma once
#include "core/gpu/render_resource.hpp"
#include <memory>
#include <vector>

namespace LX_core {
  class IComponent {
  public:
    virtual ~IComponent() = default;
    virtual std::vector<IRenderResourcePtr> getRenderResources() const = 0;
  };
}