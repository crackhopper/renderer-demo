cmake_minimum_required(VERSION 3.16)
project(VulkanGLFWDemo LANGUAGES C CXX)

# 设置编译器编码选项
if(MSVC)
    # MSVC: 明确指定使用 UTF-8
    add_compile_options(/utf-8)
    add_definitions(-D_UTF8_SOURCE)
else()
    # GCC/Clang: 设置源码字符集
    add_compile_options(-finput-charset=UTF-8)
    add_compile_options(-fexec-charset=UTF-8)
endif()

# 设置执行字符集
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fwide-exec-charset=UTF-8)
endif()

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
    src/utils.cpp
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



# 添加使用 glslc.exe 编译shader 为 .spv 的目标
# 定义 shader 目录和输出目录
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# 确保输出目录存在
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# 打印调试信息
message(STATUS "SHADER_DIR: ${SHADER_DIR}")
message(STATUS "SHADER_OUTPUT_DIR: ${SHADER_OUTPUT_DIR}")
message(STATUS "VULKAN_SDK: $ENV{VULKAN_SDK}")

# 直接使用绝对路径，避免任何路径转换问题
if(WIN32)
  set(GLSLC_EXECUTABLE "$ENV{VULKAN_SDK}/Bin/glslc.exe")
else()
  set(GLSLC_EXECUTABLE "$ENV{VULKAN_SDK}/x86_64/bin/glslc")
endif()
message(STATUS "GLSLC_EXECUTABLE: ${GLSLC_EXECUTABLE}")

# 为每个shader创建自定义命令 - 使用绝对路径和引号
add_custom_command(
    OUTPUT "${SHADER_OUTPUT_DIR}/vert.spv"
    COMMAND "${GLSLC_EXECUTABLE}"
        -o "${SHADER_OUTPUT_DIR}/vert.spv"
        "${SHADER_DIR}/shader.vert"
    DEPENDS "${SHADER_DIR}/shader.vert"
    COMMENT "Compiling vertex shader"
    VERBATIM
)

add_custom_command(
    OUTPUT "${SHADER_OUTPUT_DIR}/frag.spv"
    COMMAND "${GLSLC_EXECUTABLE}"
        -o "${SHADER_OUTPUT_DIR}/frag.spv"
        "${SHADER_DIR}/shader.frag"
    DEPENDS "${SHADER_DIR}/shader.frag"
    COMMENT "Compiling fragment shader"
    VERBATIM
)

# 创建目标依赖这些输出文件
add_custom_target(CompileShaders
    DEPENDS 
        "${SHADER_OUTPUT_DIR}/vert.spv"
        "${SHADER_OUTPUT_DIR}/frag.spv"
    SOURCES
        "${SHADER_DIR}/shader.vert"
        "${SHADER_DIR}/shader.frag"
)



# -------------------------------
# experiments 目录的构建 （方便我们针对一些细节单独运行一些程序来测试）
# -------------------------------
set(EXPERIMENTS_TARGETS 
    CreateInstanceTest 
    InitGLFWTest
)

# 用于批量构建的自定义目标
add_custom_target(BuildExperiments)

# 遍历列表，批量定义 Target
foreach(TEST_NAME IN LISTS EXPERIMENTS_TARGETS)
    # 1. 定义可执行文件
    add_executable(${TEST_NAME} experiments/${TEST_NAME}.cpp)
    
    # 2. 设置包含目录和链接库
    target_include_directories(${TEST_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} PRIVATE Vulkan::Vulkan glfw)
    
    # 3. 设置输出目录到 experiments 子文件夹
    set_target_properties(${TEST_NAME} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/experiments"
    )
    
    # 4. 将其添加为 BuildExperiments 的依赖
    add_dependencies(BuildExperiments ${TEST_NAME})
endforeach()