#include "core/utils/filesystem_tools.hpp"

#include <filesystem>
#include <iostream>

namespace {

int failures = 0;

#define EXPECT(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "[FAIL] " << __FUNCTION__ << ":" << __LINE__ << " " << msg  \
                << " (" #cond ")\n";                                           \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

void testDamagedHelmet() {
  auto saved = std::filesystem::current_path();
  bool ok = cdToWhereAssetsExist("models/damaged_helmet/DamagedHelmet.gltf");
  EXPECT(ok, "DamagedHelmet.gltf must be found");
  std::filesystem::current_path(saved);
}

void testHdrEnvironmentMap() {
  auto saved = std::filesystem::current_path();
  bool ok = cdToWhereAssetsExist("env/studio_small_03_2k.hdr");
  EXPECT(ok, "studio_small_03_2k.hdr must be found");
  std::filesystem::current_path(saved);
}

void testVikingRoom() {
  auto saved = std::filesystem::current_path();
  bool ok = cdToWhereAssetsExist("models/viking_room/viking_room.obj");
  EXPECT(ok, "viking_room.obj must be found");
  std::filesystem::current_path(saved);
}

void testNonexistent() {
  auto saved = std::filesystem::current_path();
  bool ok = cdToWhereAssetsExist("nonexistent/foo.bar");
  EXPECT(!ok, "nonexistent path must not be found");
  std::filesystem::current_path(saved);
}

} // namespace

int main() {
  testDamagedHelmet();
  testHdrEnvironmentMap();
  testVikingRoom();
  testNonexistent();

  if (failures == 0) {
    std::cout << "[PASS] All asset layout tests passed.\n";
  } else {
    std::cerr << "[SUMMARY] " << failures << " test(s) failed.\n";
  }
  return failures == 0 ? 0 : 1;
}
