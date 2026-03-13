#pragma once
#include "../math/vec.hpp"
#include <algorithm>
#include <vector>

namespace LX_core {

class Buffer {
public:
  virtual ~Buffer() = default;
  virtual const void *data() const = 0;
  virtual size_t size() const = 0;
};

} // namespace LX_core