cmake_minimum_required(VERSION 3.16)
project(VulkanGLFWDemo LANGUAGES C CXX)

# 设定 C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器编码选项
if(MSVC)
    # MSVC: 明确指定使用 UTF-8
    add_compile_options(/utf-8)
else()
    # GCC/Clang: 设置源码字符集
    add_compile_options(-finput-charset=UTF-8)
    add_compile_options(-fexec-charset=UTF-8)
endif()
# 定义一个预编译宏，方便我们自己使用
add_definitions(-D_UTF8_SOURCE)



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
# 主项目
# -------------------------------
set(SOURCES
    src/main.cpp
    src/utils.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw)
# -------------------------------
# Windows 特殊：定义宏
# 关闭MSVC的安全警告，例如，用 strcpy_s 替代 strcpy，用 scanf_s 替代 scanf
# 提高可移植性
# -------------------------------
if (WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "_CRT_SECURE_NO_WARNINGS")
endif()




# 引入shader编译的target
add_subdirectory(shaders)

# 引入 experiments 目录的构建
add_subdirectory(experiments)



