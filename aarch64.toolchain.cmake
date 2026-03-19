set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 目标三元组
set(ARM_TRIPLE aarch64-linux-gnu)

# 关键：指定基础 Sysroot (Arch Linux 路径)
set(CMAKE_SYSROOT /usr/aarch64-linux-gnu)

# 优先使用 clang 交叉编译（GCC 9 缺少 vld1q_*_x4 相关 intrinsic 声明）
find_program(CLANG_PATH NAMES clang)
find_program(CLANGXX_PATH NAMES clang++)

if (CLANG_PATH AND CLANGXX_PATH)
    message(STATUS "Using clang cross toolchain for ${ARM_TRIPLE}")
    set(CMAKE_C_COMPILER   ${CLANG_PATH})
    set(CMAKE_CXX_COMPILER ${CLANGXX_PATH})
    set(CMAKE_C_COMPILER_TARGET   ${ARM_TRIPLE})
    set(CMAKE_CXX_COMPILER_TARGET ${ARM_TRIPLE})

    # 编译与链接都显式指定目标和 sysroot
    string(APPEND CMAKE_C_FLAGS_INIT " --target=${ARM_TRIPLE} --sysroot=${CMAKE_SYSROOT} -pthread")
    string(APPEND CMAKE_CXX_FLAGS_INIT " --target=${ARM_TRIPLE} --sysroot=${CMAKE_SYSROOT} -pthread")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " --target=${ARM_TRIPLE} --sysroot=${CMAKE_SYSROOT} -pthread")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " --target=${ARM_TRIPLE} --sysroot=${CMAKE_SYSROOT} -pthread")
else()
    message(STATUS "Clang not found, fallback to GCC cross toolchain")
    set(CMAKE_C_COMPILER /usr/bin/${ARM_TRIPLE}-gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/${ARM_TRIPLE}-g++)
endif()

# 交叉编译场景下，try_compile 只做静态库检查，避免执行目标架构程序
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# 设置搜索模式：严禁去宿主机找库和头文件
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# 如果你有额外的 SDK 目录，取消下面两行的注释并修改路径
set(SDK_DIR "/home/ray/arm64_sysroot")
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT} ${SDK_DIR})

# 强制修复：有些第三方库会无视 sysroot 跑去 /usr/include，
# 我们通过 CFLAGS 强制让编译器优先看 sysroot
# add_compile_options(-nostdinc)
include_directories(SYSTEM /usr/aarch64-linux-gnu/include)
include_directories(SYSTEM /usr/aarch64-linux-gnu/include/c++/9.3.0)
include_directories(SYSTEM /usr/aarch64-linux-gnu/include/c++/9.3.0/aarch64-linux-gnu)