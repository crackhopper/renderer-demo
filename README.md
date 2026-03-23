# 小型渲染器 (LX Renderer)

一个基于 Vulkan 的模块化渲染引擎，使用现代 C++20 开发。

## 架构设计

```
src/
├── core/                    # 核心模块（平台无关）
│   ├── gpu/                # GPU 接口定义
│   │   ├── renderer.hpp    # 渲染器抽象接口
│   │   └── render_resource.hpp  # 渲染资源基类
│   ├── math/               # 数学库（Vec3f, Mat4f, Quatf）
│   ├── platform/           # 平台抽象（Window, types）
│   ├── resources/          # 资源类型
│   │   ├── vertex_buffer.hpp   # 顶点缓冲（模板化，支持多种格式）
│   │   ├── index_buffer.hpp     # 索引缓冲
│   │   ├── texture.hpp          # 纹理
│   │   └── shader.hpp           # Shader 标记类
│   └── scene/              # 场景图
│       ├── scene.hpp       # 场景容器
│       ├── camera.hpp      # 相机（含 UBO）
│       ├── light.hpp       # 光照
│       ├── object.hpp      # IRenderable 接口 + PushConstant
│       └── components/     # 组件（Mesh, Material, Skeleton）
│
├── graphics_backend/       # 图形后端实现
│   └── vulkan/             # Vulkan 后端
│       ├── vk_renderer.hpp # VulkanRenderer 封装
│       └── details/        # 内部实现
│           ├── vk_device.hpp
│           ├── vk_resource_manager.hpp
│           ├── commands/   # 命令缓冲管理
│           ├── descriptors/ # 描述符管理
│           ├── pipelines/  # 渲染管线（BlinnPhong 等）
│           ├── render_objects/ # RenderPass, Framebuffer, Swapchain
│           └── resources/  # GPU 资源（Buffer, Texture, Shader）
│
├── infra/                  # 基础设施
│   ├── window/             # 窗口管理（SDL/GLFW）
│   ├── gui/                # ImGui 集成
│   └── mesh_loader/        # 网格加载器
│
└── test/                   # 测试程序
    └── test_render_triangle.cpp  # 三角形渲染测试
```

## 核心概念

### 渲染资源体系

```
IRenderResource (抽象基类)
├── VertexBuffer<VType>     # 模板顶点缓冲
├── IndexBuffer             # 索引缓冲
├── UniformBuffer (UBO)     # 统一缓冲
├── CombinedTextureSampler  # 纹理 + 采样器
├── Shader (Vertex/Fragment) # Shader 标记
└── PushConstant (ObjectPC) # Push 常数
```

### 顶点格式

支持多种顶点格式（通过模板）:
- `VertexPos` - 仅位置
- `VertexPosColor` - 位置 + 颜色
- `VertexPosUV` - 位置 + UV
- `VertexNormalTangent` - 法线 + 切线
- `VertexBoneWeight` - 骨骼权重
- `VertexPosNormalUvBone` - 完整格式（位置 + 法线 + UV + 切线 + 骨骼）

### 场景图

```
Scene
├── mesh: IRenderablePtr     # 可渲染对象
├── camera: CameraPtr         # 相机（含 view/proj 矩阵）
└── directionalLight         # 方向光

IRenderable (接口)
└── RenderableSubMesh<VType> # 具体实现
    ├── mesh: MeshPtr<VType> # 网格数据
    ├── material: MaterialPtr # 材质
    ├── skeleton: SkeletonPtr # 骨骼（可选）
    └── objectPC: ObjectPCPtr # Push Constant
```

### 材质系统

```
MaterialBase (抽象)
└── MaterialBlinnPhong      # Blinn-Phong 材质
    ├── MaterialBlinnPhongUBO (baseColor, shininess, etc.)
    ├── albedoMap: CombinedTextureSampler
    └── normalMap: CombinedTextureSampler
```

## 构建

### 前置依赖

- CMake 3.16+
- Visual Studio 2022 或 GCC/Clang (C++20)
- Vulkan SDK 1.4+
- SDL3 或 GLFW

### 构建命令

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Debug
```

### 运行测试

```bash
# 三角形渲染测试
./build/test_render_triangle.exe
```

## 项目状态

### 已实现

- ✅ 核心类型系统（Vec3f, Mat4f, Quatf）
- ✅ 渲染资源抽象（VertexBuffer, IndexBuffer, Texture, Shader）
- ✅ 场景图基础结构（Scene, Camera, IRenderable）
- ✅ Vulkan 后端框架（设备、命令缓冲、描述符、流水线）
- ✅ 材质系统（Blinn-Phong）

### 待实现

- ⏳ 渲染循环完整实现（uploadData, draw 方法）
- ⏳ 骨骼动画系统
- ⏳ PBR 材质
- ⏳ 延迟渲染管线
- ⏳ 阴影映射
