set(TARGET_NAME ggml-fmsh_zg330)

set(FMSH_ZG330_MODELZOO_ROOT "/ModelzooDeps" CACHE PATH "Modelzoo deps root for FMSH ZG330 backend")
set(FMSH_ZG330_ARCH "x64" CACHE STRING "FMSH ZG330 build arch: x64 or arm64")
set_property(CACHE FMSH_ZG330_ARCH PROPERTY STRINGS x64 arm64)

if (FMSH_ZG330_ARCH STREQUAL "arm64")
    list(APPEND CMAKE_PREFIX_PATH "${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake/opencv4")
    list(APPEND CMAKE_PREFIX_PATH "${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/cmake/yaml-cpp")
    link_directories("${FMSH_ZG330_MODELZOO_ROOT}/aarch64/Static/lib/")
else()
    list(APPEND CMAKE_PREFIX_PATH "${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib/cmake/opencv4")
    list(APPEND CMAKE_PREFIX_PATH "${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib/cmake/yaml-cpp")
    link_directories("${FMSH_ZG330_MODELZOO_ROOT}/x64/Dynamic/lib/")
endif()

find_package(Icraft-BuyiBackend REQUIRED)
find_package(Icraft-HostBackend REQUIRED)
find_package(icraft-zg330backend REQUIRED)

ggml_add_backend_library(${TARGET_NAME}
                         ggml-fmsh-zg330.cpp
                         ggml-fmsh-zg330-netmake.cpp
                         ggml-fmsh-zg330-netmake.h
                         ../../include/ggml-fmsh-zg330.h)

target_link_libraries(${TARGET_NAME} PRIVATE Icraft::HostBackend Icraft::BuyiBackend Icraft::ZG330Backend)

if (GGML_FMSH_ZG330_DEBUG_COMPARE)
    target_compile_definitions(${TARGET_NAME} PRIVATE GGML_FMSH_ZG330_DEBUG_COMPARE)
endif()

if (NOT GGML_BACKEND_DL)
    add_executable(ggml-fmsh-zg330-flash-attn-test
                   flash-attn-netmake-cpu-test.cpp
                   ggml-fmsh-zg330-netmake.cpp
                   ggml-fmsh-zg330-netmake.h)
    target_include_directories(ggml-fmsh-zg330-flash-attn-test PRIVATE . ../../include)
    target_link_libraries(ggml-fmsh-zg330-flash-attn-test PRIVATE ggml-base ggml-cpu Icraft::HostBackend Icraft::BuyiBackend Icraft::ZG330Backend)
endif()
