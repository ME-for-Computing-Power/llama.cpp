#pragma once

#include <icraft-xir/core/network.h>

#include <cstdint>
#include <filesystem>
#include <string>

icraft::xir::Network fmsh_zg330_compile_network(
        const std::filesystem::path & cache_root,
        const std::string & net_name,
        int64_t k,
        int64_t n);
