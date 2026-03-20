#pragma once
#include "core/gpu/render_resource.hpp"
#include "core/math/quat.hpp"
#include "core/math/vec.hpp"
#include "base.hpp"
#include <cassert>
#include <string>
#include <vector>
namespace LX_core {

const u32 MAX_BONE_COUNT = 128;

struct Bone {
  std::string name;
  int parentIndex;
  Vec3f position;
  Quatf rotation;
  Vec3f scale = Vec3f(1, 1, 1);
};

struct alignas(16) SkeletonUBO : public IRenderResource {
  SkeletonUBO(const std::vector<Bone> &bones, ResourcePassFlag passFlag)
      : m_passFlag(passFlag) {}

  void updateBy(const std::vector<Bone> &bones) {
    assert(bones.size() <= MAX_BONE_COUNT);
    if (bones.empty()) {
      return;
    }
    for (size_t i = 0; i < bones.size(); i++) {
      writeToUbo(bones[i], m_bones[i]);
    }
    setDirty();
  }

  bool setBone(int index, const Bone &bone) {
    if (index < 0 || index >= MAX_BONE_COUNT) {
      return false;
    }
    auto succ = writeToUbo(bone, m_bones[index]);
    if (succ) {
      setDirty();
      return true;
    } else {
      return false;
    }
  }

  bool writeToUbo(const Bone &bone, Mat4f &out) {
    auto r = bone.rotation.toMat4();
    auto p2 = Mat4f::translate(bone.position);
    auto p1 = Mat4f::translate(-bone.position);
    auto local = p2 * r * p1;
    if (bone.parentIndex == -1)
      out = local;
    else {
      if (bone.parentIndex < 0 || bone.parentIndex >= MAX_BONE_COUNT) {
        return false;
      } else {
        out = m_bones[bone.parentIndex] * local;
      }
    }
    return true;
  }

  virtual ResourcePassFlag getPassFlag() const override { return m_passFlag; }
  virtual ResourceType getType() const override {
    return ResourceType::UniformBuffer;
  }
  static constexpr u32 ResourceSize = MAX_BONE_COUNT * sizeof(Mat4f);
  virtual const void *getRawData() const override { return m_bones; }
  virtual u32 getByteSize() const override { return ResourceSize; }
  // 默认实现，返回None槽位
  virtual PipelineSlotId getPipelineSlotId() const override {
    return PipelineSlotId::SkeletonUBO;
  }

private:
  Mat4f m_bones[MAX_BONE_COUNT];
  ResourcePassFlag m_passFlag = ResourcePassFlag::Forward;
};

using SkeletonUboPtr = std::shared_ptr<SkeletonUBO>;

struct Skeleton : public IComponent {

  Skeleton(const std::vector<Bone> &bones,
           ResourcePassFlag passFlag = ResourcePassFlag::Forward)
      : bones(bones), ubo(std::make_shared<SkeletonUBO>(bones, passFlag)) {}

  bool addBone(const Bone &bone) {
    if (bones.size() >= MAX_BONE_COUNT) {
      return false;
    }
    bones.push_back(bone);
    ubo->setBone(bones.size() - 1, bone);
    return true;
  }

  void updateUBO() { ubo->updateBy(bones); }

  std::vector<IRenderResourcePtr> getRenderResources() const override {
    return {std::dynamic_pointer_cast<IRenderResource>(ubo)};
  }

private:
  std::vector<Bone> bones;
  SkeletonUboPtr ubo;
};

using SkeletonPtr = std::shared_ptr<Skeleton>;

} // namespace LX_core
