# 添加使用 glslc.exe 编译shader 为 .spv 的目标
find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS $ENV{VULKAN_SDK}/Bin $ENV{VULKAN_SDK}/x86_64/bin
    REQUIRED
)

# 定义 shader 目录和输出目录
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# 自动查找所有着色器文件
file(GLOB SHADER_FILES
    "${SHADER_DIR}/*.vert"
    "${SHADER_DIR}/*.frag"
    # ...可以添加更多类型，如.comp, .geom等
)

set(SPV_FILES "") # 用于存储所有生成的.spv文件路径

foreach(SHADER_FILE ${SHADER_FILES})
    # 获取着色器文件的完整文件名（例如: shader.vert）
    get_filename_component(BASE_NAME ${SHADER_FILE} NAME) 
    message(STATUS "Shader file: ${SHADER_FILE}")
    
    # 构造输出 .spv 文件的完整路径
    # 示例: .../shaders/shader.vert.spv
    set(OUTPUT_SPV "${SHADER_OUTPUT_DIR}/${BASE_NAME}.spv") 
    message(STATUS "Output SPV file: ${OUTPUT_SPV}")

    # 将生成的 .spv 文件添加到列表中
    list(APPEND SPV_FILES "${OUTPUT_SPV}")

    # 定义自定义编译命令
    add_custom_command(
        OUTPUT "${OUTPUT_SPV}"
        COMMAND "${GLSLC_EXECUTABLE}"
            -o "${OUTPUT_SPV}"
            "${SHADER_FILE}"
        DEPENDS "${SHADER_FILE}"
        COMMENT "Compiling shader: ${BASE_NAME}"
        VERBATIM
    )
endforeach()

# 创建目标依赖所有生成的 .spv 文件
add_custom_target(CompileShaders
    DEPENDS ${SPV_FILES}
)

