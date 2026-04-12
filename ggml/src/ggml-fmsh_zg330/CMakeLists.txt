set(FMSH_ZG330_ARCH "auto" CACHE STRING "FMSH ZG330 target arch: auto, x64, arm64")
set_property(CACHE FMSH_ZG330_ARCH PROPERTY STRINGS auto x64 arm64)
set(FMSH_ZG330_MODELZOO_ROOT "/ModelzooDeps" CACHE PATH "FMSH ZG330 Modelzoo deps root")

if (FMSH_ZG330_ARCH STREQUAL "auto")
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(FMSH_ZG330_ARCH "arm64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
        set(FMSH_ZG330_ARCH "x64")
    else()
        message(FATAL_ERROR "FMSH_ZG330: unsupported processor ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()

if (UNIX)
    if (FMSH_ZG330_ARCH STREQUAL "arm64")
        list(APPEND CMAKE_PREFIX_PATH
            ${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake
            ${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake/opencv4
            ${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake/yaml-cpp
            ${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake/Icraft-HostBackend
            ${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake/icraft-zg330backend
        )
        link_directories(${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib)
        message(STATUS "FMSH_ZG330: configuring arm64 cross build with AXI runtime path")
    elseif (FMSH_ZG330_ARCH STREQUAL "x64")
        list(APPEND CMAKE_PREFIX_PATH
            ${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib/cmake
            ${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib/cmake/Icraft-HostBackend
            ${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib/cmake/icraft-zg330backend
        )
        link_directories(${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib)
        message(STATUS "FMSH_ZG330: configuring x64 build with socket runtime path")
    else()
        message(FATAL_ERROR "FMSH_ZG330: unsupported FMSH_ZG330_ARCH=${FMSH_ZG330_ARCH}")
    endif()
endif()

find_package(Icraft-HostBackend REQUIRED)
find_package(icraft-zg330backend REQUIRED)

ggml_add_backend_library(ggml-fmsh_zg330
    fmsh-zg330-netmake.cpp
    ggml-fmsh-zg330.cpp
)

target_link_libraries(ggml-fmsh_zg330 PRIVATE
    Icraft::HostBackend
    Icraft::ZG330Backend
)

if (FMSH_ZG330_ARCH STREQUAL "arm64")
    target_compile_definitions(ggml-fmsh_zg330 PRIVATE FMSH_ZG330_USE_AXI=1)
elseif (FMSH_ZG330_ARCH STREQUAL "x64")
    target_compile_definitions(ggml-fmsh_zg330 PRIVATE FMSH_ZG330_USE_SOCKET=1)
endif()
