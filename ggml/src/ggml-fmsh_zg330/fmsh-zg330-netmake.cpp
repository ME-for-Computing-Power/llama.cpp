#include "fmsh-zg330-netmake.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace fs = std::filesystem;

using namespace icraft::xir;

namespace {

static bool fmsh_log_enabled(void) {
    const char * env_cstr = std::getenv("GGML_FMSH_ZG330_LOG");
    if (env_cstr == nullptr) {
        return false;
    }
    const std::string env = env_cstr;
    return !env.empty() && env != "0" && env != "false" && env != "FALSE";
}

static void fmsh_log(const std::string & msg) {
    if (fmsh_log_enabled()) {
        std::cerr << "\rfmsh_zg330: " << msg << std::endl;
    }
}

static void fmsh_ensure_dir(const fs::path & path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec) {
        throw std::runtime_error("fmsh_zg330: failed to create directory " + path.string() + ": " + ec.message());
    }
}

static std::string fmsh_quote_sh(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('\'');
    for (char c : value) {
        if (c == '\'') {
            out += "'\"'\"'";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

static std::string fmsh_quote_toml(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char c : value) {
        switch (c) {
            case '\\':
            case '"':
                out.push_back('\\');
                out.push_back(c);
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    out.push_back('"');
    return out;
}

static void fmsh_run_checked(const std::string & cmd) {
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("fmsh_zg330: command failed (" + std::to_string(rc) + "): " + cmd);
    }
}

static bool fmsh_run_ok(const std::string & cmd) {
    return std::system(cmd.c_str()) == 0;
}

static void fmsh_build_matmul_onnx(
        const fs::path & work_dir,
        const std::string & net_name,
        int64_t k,
        int64_t n) {
    if (!fmsh_run_ok("python3 -c " + fmsh_quote_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("fmsh_zg330: missing Python package `onnx`");
    }

    const fs::path script_path = work_dir / "make_matmul_onnx.py";
    const fs::path onnx_path   = work_dir / (net_name + ".onnx");

    std::ofstream py(script_path);
    if (!py) {
        throw std::runtime_error("fmsh_zg330: failed to write " + script_path.string());
    }

    py << R"PY(
import sys
import onnx
from onnx import helper, TensorProto

m = int(sys.argv[1])
k = int(sys.argv[2])
n = int(sys.argv[3])
onnx_path = sys.argv[4]

A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [k, n])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])

node = helper.make_node("MatMul", inputs=["A", "W"], outputs=["Y"], name="MatMul_0")
graph = helper.make_graph([node], "fmsh_zg330_matmul", [A, W], [Y])
model = helper.make_model(graph, producer_name="ggml-fmsh_zg330", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, onnx_path)
)PY";
    py.close();

    std::ostringstream cmd;
    cmd << "cd " << fmsh_quote_sh(work_dir.string())
        << " && python3 " << fmsh_quote_sh(script_path.filename().string())
        << " " << 1
        << " " << k
        << " " << n
        << " " << fmsh_quote_sh(onnx_path.filename().string());
    fmsh_run_checked(cmd.str());
}

static void fmsh_write_compile_toml(const fs::path & work_dir, const std::string & net_name, int64_t k, int64_t n) {
    const fs::path toml_path = work_dir / (net_name + ".toml");
    const std::string jr = "./.cache/" + net_name + "_ZG/";

    std::ofstream ofs(toml_path);
    if (!ofs) {
        throw std::runtime_error("fmsh_zg330: failed to write " + toml_path.string());
    }

    ofs
        << "[parse]\n"
        << "net_name = " << fmsh_quote_toml(net_name) << "\n"
        << "framework = \"Onnx\"\n"
        << "network = " << fmsh_quote_toml(net_name + ".onnx") << "\n"
        << "jr_path = " << fmsh_quote_toml(jr) << "\n"
        << "target = \"zhuge\"\n"
        << "inputs = [[1," << k << "],[" << k << "," << n << "]]\n"
        << "inputs_layout = \"FD;FD\"\n"
        << "inputs_dtype = \"bf16;bf16\"\n"
        << "pre_method = \"nop;nop\"\n"
        << "pre_mean = \"nop;nop\"\n"
        << "pre_scale = \"nop;nop\"\n"
        << "channel_swap = \"nop;nop\"\n\n"
        << "[optimize]\n"
        << "json = " << fmsh_quote_toml(jr + net_name + "_parsed.json") << "\n"
        << "raw = " << fmsh_quote_toml(jr + net_name + "_parsed.raw") << "\n"
        << "jr_path = " << fmsh_quote_toml(jr) << "\n\n"
        << "[quantize]\n"
        << "json = " << fmsh_quote_toml(jr + net_name + "_optimized.json") << "\n"
        << "raw = " << fmsh_quote_toml(jr + net_name + "_optimized.raw") << "\n"
        << "jr_path = " << fmsh_quote_toml(jr) << "\n"
        << "target = \"zhuge\"\n"
        << "qdtype = \"bf16\"\n"
        << "forward_mode = \"image\"\n\n"
        << "[adapt]\n"
        << "json = " << fmsh_quote_toml(jr + net_name + "_quantized.json") << "\n"
        << "raw = " << fmsh_quote_toml(jr + net_name + "_quantized.raw") << "\n"
        << "jr_path = " << fmsh_quote_toml(jr) << "\n\n"
        << "[generate]\n"
        << "json = " << fmsh_quote_toml(jr + net_name + "_adapted.json") << "\n"
        << "raw = " << fmsh_quote_toml(jr + net_name + "_adapted.raw") << "\n"
        << "jr_path = " << fmsh_quote_toml(jr) << "\n";
}

static std::pair<fs::path, fs::path> fmsh_find_generated_artifacts(const fs::path & work_dir, const std::string & net_name) {
    const fs::path compact_json = work_dir / (net_name + "_ZG.json");
    const fs::path compact_raw  = work_dir / (net_name + "_ZG.raw");
    if (fs::exists(compact_json) && fs::exists(compact_raw)) {
        return { compact_json, compact_raw };
    }

    const fs::path cache_dir = work_dir / ".cache";
    const fs::path direct_json = cache_dir / (net_name + "_ZG.json");
    const fs::path direct_raw  = cache_dir / (net_name + "_ZG.raw");

    if (fs::exists(direct_json) && fs::exists(direct_raw)) {
        return { direct_json, direct_raw };
    }

    fs::path found_json;
    fs::path found_raw;
    if (fs::exists(cache_dir)) {
        for (const auto & entry : fs::recursive_directory_iterator(cache_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string filename = entry.path().filename().string();
            if (filename == net_name + "_ZG.json") {
                found_json = entry.path();
            } else if (filename == net_name + "_ZG.raw") {
                found_raw = entry.path();
            }
        }
    }

    if (found_json.empty() || found_raw.empty()) {
        throw std::runtime_error("fmsh_zg330: cannot find compiled artifacts under " + cache_dir.string());
    }

    return { found_json, found_raw };
}

static void fmsh_try_remove_file(const fs::path & path) {
    std::error_code ec;
    fs::remove(path, ec);
}

static void fmsh_cleanup_compile_workspace(const fs::path & work_dir, const std::string & net_name) {
    std::error_code ec;
    fs::remove_all(work_dir / ".cache", ec);
    fmsh_try_remove_file(work_dir / "make_matmul_onnx.py");
    fmsh_try_remove_file(work_dir / (net_name + ".onnx"));
    fmsh_try_remove_file(work_dir / (net_name + ".toml"));
}

} // namespace

Network fmsh_zg330_compile_network(
        const fs::path & cache_root,
        const std::string & net_name,
        int64_t k,
        int64_t n) {
    const fs::path work_dir = cache_root / net_name;
    fmsh_ensure_dir(work_dir);
    const fs::path compact_json = work_dir / (net_name + "_ZG.json");
    const fs::path compact_raw  = work_dir / (net_name + "_ZG.raw");

    try {
        auto [json_path, raw_path] = fmsh_find_generated_artifacts(work_dir, net_name);
        if (json_path != compact_json || raw_path != compact_raw) {
            std::error_code ecj;
            std::error_code ecr;
            fs::copy_file(json_path, compact_json, fs::copy_options::overwrite_existing, ecj);
            fs::copy_file(raw_path, compact_raw, fs::copy_options::overwrite_existing, ecr);
            if (!ecj && !ecr) {
                json_path = compact_json;
                raw_path = compact_raw;
            }
        }
        fmsh_log("disk cache hit: key=" + net_name +
                 " json=" + json_path.string() +
                 " raw=" + raw_path.string());
        auto network = Network::CreateFromJsonFile(json_path.string());
        network.loadParamsFromFile(raw_path.string());
        return network;
    } catch (const std::exception &) {
        fmsh_log("disk cache miss: key=" + net_name + ", compile required");
    }

    fmsh_build_matmul_onnx(work_dir, net_name, k, n);
    fmsh_write_compile_toml(work_dir, net_name, k, n);

    std::ostringstream cmd;
    cmd << "cd " << fmsh_quote_sh(work_dir.string())
        << " && icraft compile " << fmsh_quote_sh(net_name + ".toml");
    fmsh_run_checked(cmd.str());

    auto [json_path, raw_path] = fmsh_find_generated_artifacts(work_dir, net_name);
    std::error_code ecj;
    std::error_code ecr;
    fs::copy_file(json_path, compact_json, fs::copy_options::overwrite_existing, ecj);
    fs::copy_file(raw_path, compact_raw, fs::copy_options::overwrite_existing, ecr);
    if (!ecj && !ecr) {
        json_path = compact_json;
        raw_path = compact_raw;
    }
    fmsh_log("compile output: key=" + net_name +
             " json=" + json_path.string() +
             " raw=" + raw_path.string());

    auto network = Network::CreateFromJsonFile(json_path.string());
    network.loadParamsFromFile(raw_path.string());
    fmsh_cleanup_compile_workspace(work_dir, net_name);
    return network;
}
