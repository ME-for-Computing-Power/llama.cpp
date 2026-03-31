set(TARGET_NAME ggml-fmsh)

ggml_add_backend_library(${TARGET_NAME}
                         ggml-fmsh.cpp
                         ../../include/ggml-fmsh.h)

if (NOT GGML_FMSH_ICRAFT)
    message(FATAL_ERROR "ggml-fmsh: GGML_FMSH requires GGML_FMSH_ICRAFT=ON (fallback-only build is disabled)")
endif ()

set(_icraft_hints)
if (GGML_FMSH_ICRAFT_ROOT)
    list(APPEND _icraft_hints ${GGML_FMSH_ICRAFT_ROOT})
endif ()

# Cross toolchain friendly hints
if (CMAKE_SYSROOT)
    list(APPEND _icraft_hints ${CMAKE_SYSROOT} ${CMAKE_SYSROOT}/usr)
endif ()
if (CMAKE_FIND_ROOT_PATH)
    list(APPEND _icraft_hints ${CMAKE_FIND_ROOT_PATH})
endif ()

# Common local ARM sysroot location used in this repository environment.
if (EXISTS "/home/ray/arm64_sysroot/usr")
    list(APPEND _icraft_hints /home/ray/arm64_sysroot/usr)
endif ()
if (EXISTS "${CMAKE_SOURCE_DIR}/ref_proj/include")
    list(APPEND _icraft_hints ${CMAKE_SOURCE_DIR}/ref_proj)
endif ()

find_path(FMSH_ICRAFT_INCLUDE_DIR
    NAMES icraft-xir/ops/ops.h
    HINTS ${_icraft_hints}
    PATH_SUFFIXES include usr/include)

find_library(FMSH_LIB_ICRAFT_BUYI NAMES icraft_buyibackend HINTS ${_icraft_hints} PATH_SUFFIXES lib lib64 usr/lib usr/lib64 usr/lib/aarch64-linux-gnu)
find_library(FMSH_LIB_ICRAFT_HOST NAMES icraft_hostbackend HINTS ${_icraft_hints} PATH_SUFFIXES lib lib64 usr/lib usr/lib64 usr/lib/aarch64-linux-gnu)
find_library(FMSH_LIB_ICRAFT_XRT  NAMES icraft_xrt         HINTS ${_icraft_hints} PATH_SUFFIXES lib lib64 usr/lib usr/lib64 usr/lib/aarch64-linux-gnu)
find_library(FMSH_LIB_ICRAFT_XIR  NAMES icraft_xir         HINTS ${_icraft_hints} PATH_SUFFIXES lib lib64 usr/lib usr/lib64 usr/lib/aarch64-linux-gnu)

if (NOT FMSH_ICRAFT_INCLUDE_DIR)
    message(FATAL_ERROR "ggml-fmsh: iCraft include path not found (missing icraft-xir/ops/ops.h). Set GGML_FMSH_ICRAFT_ROOT.")
endif ()

if (NOT FMSH_LIB_ICRAFT_BUYI OR NOT FMSH_LIB_ICRAFT_HOST OR NOT FMSH_LIB_ICRAFT_XRT OR NOT FMSH_LIB_ICRAFT_XIR)
    message(FATAL_ERROR "ggml-fmsh: required iCraft runtime libraries are missing (need: icraft_buyibackend, icraft_hostbackend, icraft_xrt, icraft_xir). Set GGML_FMSH_ICRAFT_ROOT.")
endif ()

target_include_directories(${TARGET_NAME} PRIVATE ${FMSH_ICRAFT_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE
    ${FMSH_LIB_ICRAFT_BUYI}
    ${FMSH_LIB_ICRAFT_HOST}
    ${FMSH_LIB_ICRAFT_XRT}
    ${FMSH_LIB_ICRAFT_XIR})
target_compile_definitions(${TARGET_NAME} PRIVATE GGML_FMSH_USE_ICRAFT)

message(STATUS "ggml-fmsh: iCraft runtime integration enabled")
