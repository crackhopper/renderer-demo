#include "core/asset/mesh.hpp"
#include "core/asset/skeleton.hpp"
#include "core/frame_graph/pass.hpp"
#include "core/frame_graph/render_queue.hpp"
#include "core/rhi/index_buffer.hpp"
#include "core/rhi/vertex_buffer.hpp"
#include "core/scene/object.hpp"
#include "core/scene/scene.hpp"
#include "infra/material_loader/blinn_phong_material_loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace LX_core;
using namespace LX_infra;

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

MeshPtr makeMeshWithSkinningInputs() {
  auto vb = VertexBuffer<VertexPosNormalUvBone>::create(
      std::vector<VertexPosNormalUvBone>{
          VertexPosNormalUvBone{{0, 0, 0},
                                {0, 1, 0},
                                {0, 0},
                                {1, 0, 0, 1},
                                {0, 0, 0, 0},
                                {1, 0, 0, 0}},
          VertexPosNormalUvBone{{1, 0, 0},
                                {0, 1, 0},
                                {1, 0},
                                {1, 0, 0, 1},
                                {0, 0, 0, 0},
                                {1, 0, 0, 0}},
          VertexPosNormalUvBone{{0, 1, 0},
                                {0, 1, 0},
                                {0, 1},
                                {1, 0, 0, 1},
                                {0, 0, 0, 0},
                                {1, 0, 0, 0}},
      });
  auto ib = IndexBuffer::create({0, 1, 2});
  return Mesh::create(vb, ib);
}

MeshPtr makeMeshWithoutSkinningInputs() {
  std::vector<VertexPBR> vertices(3);
  vertices[0].pos = {0, 0, 0};
  vertices[0].normal = {0, 1, 0};
  vertices[0].uv = {0, 0};
  vertices[0].tangent = {1, 0, 0, 1};
  vertices[1].pos = {1, 0, 0};
  vertices[1].normal = {0, 1, 0};
  vertices[1].uv = {1, 0};
  vertices[1].tangent = {1, 0, 0, 1};
  vertices[2].pos = {0, 1, 0};
  vertices[2].normal = {0, 1, 0};
  vertices[2].uv = {0, 1};
  vertices[2].tangent = {1, 0, 0, 1};
  auto vb = VertexBuffer<VertexPBR>::create(std::move(vertices));
  auto ib = IndexBuffer::create({0, 1, 2});
  return Mesh::create(vb, ib);
}

SkeletonPtr makeSkeleton() {
  std::vector<Bone> bones = {
      Bone{"root", -1, Vec3f{0, 0, 0}, Quatf{}, Vec3f{1, 1, 1}},
  };
  return Skeleton::create(bones);
}

MaterialInstance::Ptr makeMaterial(bool skinning) {
  return loadBlinnPhongMaterial(
      ResourcePassFlag::Forward,
      {ShaderVariant{"USE_LIGHTING", true},
       ShaderVariant{"USE_SKINNING", skinning}});
}

bool hasBinding(const std::vector<IRenderResourcePtr> &resources,
                const char *bindingName) {
  const StringID id(bindingName);
  for (const auto &resource : resources) {
    if (resource && resource->getBindingName() == id)
      return true;
  }
  return false;
}

int runSelf(const std::filesystem::path &self, const char *mode) {
  const std::string cmd =
      "\"" + self.string() + "\" " + mode + " >/dev/null 2>&1";
  return std::system(cmd.c_str());
}

void testIndependentSceneNodeValidation() {
  auto node = SceneNode::create("node_base", makeMeshWithSkinningInputs(),
                                makeMaterial(false), nullptr);
  EXPECT(node->supportsPass(Pass_Forward),
         "independent SceneNode should validate without Scene");
  auto validated = node->getValidatedPassData(Pass_Forward);
  EXPECT(validated.has_value(), "validated pass data should exist");
  if (validated) {
    EXPECT(!hasBinding(validated->get().descriptorResources, "Bones"),
           "non-skinned node should not carry Bones");
  }
}

void testPassEnableStateRebuildsCache() {
  auto material = makeMaterial(false);
  auto node = SceneNode::create("node_toggle", makeMeshWithSkinningInputs(),
                                material, nullptr);
  EXPECT(node->supportsPass(Pass_Forward), "forward pass starts enabled");

  material->setPassEnabled(Pass_Forward, false);
  EXPECT(!node->supportsPass(Pass_Forward),
         "disabling a pass invalidates SceneNode cache");
  EXPECT(!node->getValidatedPassData(Pass_Forward).has_value(),
         "validated entry removed when pass disabled");

  material->setPassEnabled(Pass_Forward, true);
  EXPECT(node->supportsPass(Pass_Forward),
         "reenabling a pass rebuilds SceneNode cache");
  EXPECT(node->getValidatedPassData(Pass_Forward).has_value(),
         "validated entry restored when pass reenabled");
}

void testSkinningVariantChangesPipelineKeyAndAddsBones() {
  auto mesh = makeMeshWithSkinningInputs();
  auto baseNode =
      SceneNode::create("node_unskinned", mesh, makeMaterial(false), nullptr);
  auto skinnedNode =
      SceneNode::create("node_skinned", mesh, makeMaterial(true), makeSkeleton());

  auto baseData = baseNode->getValidatedPassData(Pass_Forward);
  auto skinnedData = skinnedNode->getValidatedPassData(Pass_Forward);
  EXPECT(baseData.has_value(), "unskinned validated data exists");
  EXPECT(skinnedData.has_value(), "skinned validated data exists");
  if (baseData && skinnedData) {
    EXPECT(baseData->get().pipelineKey != skinnedData->get().pipelineKey,
           "variant difference should change pipeline key");
    EXPECT(hasBinding(skinnedData->get().descriptorResources, "Bones"),
           "skinned validated entry should include Bones resource");
  }
}

void testRenderQueueConsumesValidatedSceneNode() {
  auto node = SceneNode::create("node_queue", makeMeshWithSkinningInputs(),
                                makeMaterial(false), nullptr);
  auto scene = Scene::create("SceneQueue", node);
  RenderQueue queue;
  queue.buildFromScene(*scene, Pass_Forward, RenderTarget{});

  EXPECT(queue.getItems().size() == 1, "queue should consume one SceneNode");
  auto validated = node->getValidatedPassData(Pass_Forward);
  EXPECT(validated.has_value(), "validated entry should still exist");
  if (!queue.getItems().empty() && validated) {
    EXPECT(queue.getItems()[0].pipelineKey == validated->get().pipelineKey,
           "queue should reuse SceneNode validated pipeline key");
    EXPECT(queue.getItems()[0].descriptorResources.size() >=
               validated->get().descriptorResources.size(),
           "scene-level resources should be appended after validated resources");
  }
}

void testSceneAssignsStableDebugId() {
  auto node = SceneNode::create("node_debug", makeMeshWithSkinningInputs(),
                                makeMaterial(false), nullptr);
  EXPECT(node->getDebugId() == StringID{},
         "detached SceneNode should not have a scene debug id yet");
  auto scene = Scene::create("SceneDebug", node);
  (void)scene;
  EXPECT(node->getDebugId() == StringID("SceneDebug/node_debug"),
         "scene attachment should assign stable scene/node debug id");
}

int duplicateMode() {
  auto material = makeMaterial(false);
  auto nodeA =
      SceneNode::create("dup_node", makeMeshWithSkinningInputs(), material);
  auto nodeB =
      SceneNode::create("dup_node", makeMeshWithSkinningInputs(), material);
  auto scene = Scene::create("DuplicateScene", nodeA);
  scene->addRenderable(nodeB);
  return 0;
}

int invalidSkinningMode() {
  auto node = SceneNode::create("bad_skinning", makeMeshWithoutSkinningInputs(),
                                makeMaterial(true), makeSkeleton());
  (void)node;
  return 0;
}

void testFatalSubprocesses(const std::filesystem::path &self) {
  EXPECT(runSelf(self, "--duplicate") != 0,
         "duplicate node names must terminate in subprocess");
  EXPECT(runSelf(self, "--invalid-skinning") != 0,
         "missing skinning vertex inputs must terminate in subprocess");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1) {
    const std::string mode = argv[1];
    if (mode == "--duplicate")
      return duplicateMode();
    if (mode == "--invalid-skinning")
      return invalidSkinningMode();
  }

  testIndependentSceneNodeValidation();
  testPassEnableStateRebuildsCache();
  testSkinningVariantChangesPipelineKeyAndAddsBones();
  testRenderQueueConsumesValidatedSceneNode();
  testSceneAssignsStableDebugId();
  testFatalSubprocesses(std::filesystem::absolute(argv[0]));

  if (failures > 0) {
    std::cerr << "FAILED: " << failures << " assertion(s)\n";
    return 1;
  }
  std::cout << "OK: all scene node validation tests passed\n";
  return 0;
}
