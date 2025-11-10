cmake_minimum_required(VERSION 3.16)
project(VulkanGLFWDemo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------
# Vulkan
# -------------------------------
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found")
endif()

# -------------------------------
# GLFW
# -------------------------------
# 使用源码方式构建
add_subdirectory(external/glfw-3.4)

# -------------------------------
# 源文件
# -------------------------------
set(SOURCES
    src/main.cpp
)

# -------------------------------
# 可执行文件
# -------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# -------------------------------
# 包含目录
# -------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})

# -------------------------------
# 链接库
# -------------------------------
# GLFW target 名称是 glfw（源码 CMake 提供）
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw)

# -------------------------------
# Windows 特殊：定义宏
# -------------------------------
if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "_CRT_SECURE_NO_WARNINGS")
endif()

