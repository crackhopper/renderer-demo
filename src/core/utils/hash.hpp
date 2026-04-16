#pragma once

#include <cstddef>
#include <functional>

namespace LX_core {

template <class T>
inline void hash_combine(std::size_t &seed, const T &value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace LX_core
