# -------------------------------
# experiments 目录的构建 （方便我们针对一些细节单独运行一些程序来测试）
# -------------------------------
set(EXPERIMENTS_TARGETS 
    CreateInstanceTest 
    SDLIntegrationTest
)

# 用于批量构建的自定义目标
add_custom_target(BuildExperiments)

# 遍历列表，批量定义 Target
foreach(TEST_NAME IN LISTS EXPERIMENTS_TARGETS)
    # 1. 定义可执行文件
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    
    # 2. 设置包含目录和链接库
    target_include_directories(${TEST_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} PRIVATE Vulkan::Vulkan SDL3::SDL3)
    
    # 3. 设置输出目录到 experiments 子文件夹
    set_target_properties(${TEST_NAME} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/experiments"
    )
    
    # 4. 将其添加为 BuildExperiments 的依赖
    add_dependencies(BuildExperiments ${TEST_NAME})
endforeach()