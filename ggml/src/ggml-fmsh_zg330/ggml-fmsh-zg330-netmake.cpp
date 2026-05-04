#include "ggml-fmsh-zg330-netmake.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ggml::fmsh::netmake {

namespace {

struct CachedMatmulEntry {
    std::string net_name;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    icraft::xir::Network network;
    std::filesystem::path json_path;
    std::filesystem::path raw_path;
};

std::mutex g_cache_mutex;
std::unordered_map<std::string, CachedMatmulEntry> g_cache;
std::unordered_set<std::string> g_preloaded_roots_abs;

static void ensure_dir(const std::filesystem::path & p) {
    if (p.empty()) {
        throw std::runtime_error("ensure_dir: empty path");
    }
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("failed to create directory: " + p.string() + " (" + ec.message() + ")");
    }
}

static std::string quote_for_sh(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\"'\"'";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

static std::string quote_for_toml(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

static int run_system_checked(const std::string & cmd) {
    std::cout << "Running command: " << cmd << std::endl;
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("command failed (rc=" + std::to_string(rc) + "): " + cmd);
    }
    return rc;
}

static bool run_system_ok(const std::string & cmd) {
    std::cout << "Running command: " << cmd << std::endl;
    return std::system(cmd.c_str()) == 0;
}

static std::string make_root_net_key(const std::filesystem::path & root_abs, const std::string & net_name) {
    return root_abs.string() + "::" + net_name;
}

static std::string make_matmul_net_name(int64_t m, int64_t k, int64_t n) {
    return "matmul_" + std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
}

static std::string elementwise_op_name(ElementwiseZgOp op) {
    switch (op) {
        case ElementwiseZgOp::ADD: return "add";
        case ElementwiseZgOp::MUL: return "mul";
        case ElementwiseZgOp::SCALE: return "scale";
        case ElementwiseZgOp::CPY: return "cpy";
        case ElementwiseZgOp::DUP: return "dup";
        case ElementwiseZgOp::SOFT_MAX: return "softmax";
        case ElementwiseZgOp::RMS_NORM: return "rmsnorm";
        case ElementwiseZgOp::BF16_BRIDGE: return "bf16_bridge";
        default: return "unknown";
    }
}

static std::string make_elementwise_net_name(ElementwiseZgOp op, int64_t rows, int64_t cols) {
    return elementwise_op_name(op) + "_bf16_" + std::to_string(rows) + "x" + std::to_string(cols);
}

static std::string make_bf16_bridge_net_name(int64_t rows, int64_t cols) {
    return "bf16_bridge_bf16_" + std::to_string(rows) + "x" + std::to_string(cols);
}

static std::string make_flash_attn_net_name(
    int64_t head_dim,
    int64_t value_dim,
    int64_t q_len,
    int64_t kv_len,
    int64_t softmax_cols,
    bool use_logit_softcap) {
    return "flashattn_d" + std::to_string(head_dim) +
           "_dv" + std::to_string(value_dim) +
           "_q" + std::to_string(q_len) +
           "_kv" + std::to_string(kv_len) +
           "_scol" + std::to_string(softmax_cols) +
           "_sc" + std::to_string(use_logit_softcap ? 1 : 0);
}

static std::string shape_to_toml_array(const std::vector<int64_t> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) out += ",";
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

static std::string shape_layout(const std::vector<int64_t> & shape) {
    if (shape.size() == 4) {
        return "NHWC";
    }
    return "FD";
}

static bool parse_matmul_dims_from_net_name(const std::string & net_name, int64_t * m, int64_t * k, int64_t * n) {
    if (net_name.rfind("matmul_", 0) != 0) {
        return false;
    }
    const std::string dims = net_name.substr(std::string("matmul_").size());
    const size_t p1 = dims.find('x');
    if (p1 == std::string::npos) return false;
    const size_t p2 = dims.find('x', p1 + 1);
    if (p2 == std::string::npos) return false;

    try {
        const int64_t mm = std::stoll(dims.substr(0, p1));
        const int64_t kk = std::stoll(dims.substr(p1 + 1, p2 - p1 - 1));
        const int64_t nn = std::stoll(dims.substr(p2 + 1));
        if (mm <= 0 || kk <= 0 || nn <= 0) return false;
        *m = mm;
        *k = kk;
        *n = nn;
        return true;
    } catch (...) {
        return false;
    }
}

static OnnxMatmulModel build_matmul_onnx(const std::filesystem::path & onnx_path, int64_t m, int64_t k, int64_t n) {
    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::runtime_error("build_matmul_onnx: invalid shapes");
    }

    ensure_dir(onnx_path.parent_path());

    OnnxMatmulModel model;
    model.onnx_path = onnx_path;
    model.input_name = "A";
    model.weight_name = "W";
    model.output_name = "Y";
    model.m = m;
    model.k = k;
    model.n = n;

    const std::string py_script = R"PY(
import sys
import onnx
from onnx import helper, TensorProto

m = int(sys.argv[1]); k = int(sys.argv[2]); n = int(sys.argv[3]); out_path = sys.argv[4]

A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])

node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"], name="MatMul_0")
graph = helper.make_graph([node], "matmul_graph", [A, B], [Y])
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8

onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }

    const std::filesystem::path script_path = onnx_path.parent_path() / "make_matmul_onnx.py";
    std::ofstream py(script_path);
    if (!py) {
        throw std::runtime_error("failed to write python script: " + script_path.string());
    }
    py << py_script;
    py.close();

    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            std::to_string(m) + " " + std::to_string(k) + " " + std::to_string(n) + " " +
                            quote_for_sh(onnx_path.string());
    run_system_checked(cmd);
    return model;
}

static void build_elementwise_onnx(
    const std::filesystem::path & onnx_path,
    ElementwiseZgOp op,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("build_elementwise_onnx: invalid shape");
    }

    ensure_dir(onnx_path.parent_path());

    const std::string py_script = R"PY(
import sys
import onnx
from onnx import helper, TensorProto

op = sys.argv[1]
rows = int(sys.argv[2])
cols = int(sys.argv[3])
out_path = sys.argv[4]

# FD: [rows, cols]
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows, cols])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows, cols])
inputs = [X]
nodes = []

if op in ("add", "mul"):
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [rows, cols])
    inputs.append(B)
    node_type = "Add" if op == "add" else "Mul"
    nodes.append(helper.make_node(node_type, inputs=["X", "B"], outputs=["Y"], name=f"{node_type}_0"))
elif op == "scale":
    S = helper.make_tensor_value_info("S", TensorProto.FLOAT, [1, 1])
    inputs.append(S)
    nodes.append(helper.make_node("Mul", inputs=["X", "S"], outputs=["Y"], name="Mul_0"))
elif op in ("cpy", "dup"):
    S = helper.make_tensor_value_info("S", TensorProto.FLOAT, [1, 1])
    inputs.append(S)
    nodes.append(helper.make_node("Mul", inputs=["X", "S"], outputs=["Y"], name="Mul_0"))
elif op == "bf16_bridge":
    import numpy as np
    one_init = helper.make_tensor("ONE", TensorProto.FLOAT, [1, 1], np.array([1.0], dtype=np.float32).tobytes())
    nodes.append(helper.make_node("Mul", inputs=["X", "ONE"], outputs=["Y"], name="Mul_Bridge"))
    graph = helper.make_graph(nodes, f"{op}_graph", inputs, [Y], initializer=[one_init])
    model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    sys.exit(0)
elif op == "softmax":
    nodes.append(helper.make_node("Softmax", inputs=["X"], outputs=["Y"], axis=1, name="Softmax_0"))
elif op == "rmsnorm":
    E = helper.make_tensor_value_info("E", TensorProto.FLOAT, [1, 1])
    inputs.append(E)
    x2 = helper.make_node("Mul", inputs=["X", "X"], outputs=["X2"], name="Mul_X2")
    mean = helper.make_node("ReduceMean", inputs=["X2"], outputs=["M"], axes=[1], keepdims=1, name="ReduceMean_0")
    add_eps = helper.make_node("Add", inputs=["M", "E"], outputs=["ME"], name="Add_EPS")
    rms = helper.make_node("Sqrt", inputs=["ME"], outputs=["R"], name="Sqrt_0")
    div = helper.make_node("Div", inputs=["X", "R"], outputs=["Y"], name="Div_0")
    nodes.extend([x2, mean, add_eps, rms, div])
else:
    raise RuntimeError(f"unsupported op: {op}")

graph = helper.make_graph(nodes, f"{op}_graph", inputs, [Y])
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }

    const std::filesystem::path script_path = onnx_path.parent_path() / "make_elementwise_onnx.py";
    std::ofstream py(script_path);
    if (!py) {
        throw std::runtime_error("failed to write python script: " + script_path.string());
    }
    py << py_script;
    py.close();

    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            quote_for_sh(elementwise_op_name(op)) + " " +
                            std::to_string(rows) + " " + std::to_string(cols) + " " +
                            quote_for_sh(onnx_path.string());
    run_system_checked(cmd);
}

static void build_flash_attn_onnx(
    const std::filesystem::path & onnx_path,
    int64_t head_dim,
    int64_t value_dim,
    int64_t q_len,
    int64_t kv_len,
    int64_t softmax_cols,
    bool use_logit_softcap) {
    if (head_dim <= 0 || value_dim <= 0 || q_len <= 0 || kv_len <= 0 || softmax_cols < q_len) {
        throw std::runtime_error("build_flash_attn_onnx: invalid shape");
    }

    ensure_dir(onnx_path.parent_path());

    const std::string py_script = R"PY(
import sys
import onnx
from onnx import helper, TensorProto

head_dim = int(sys.argv[1])
value_dim = int(sys.argv[2])
q_len = int(sys.argv[3])
kv_len = int(sys.argv[4])
softmax_cols = int(sys.argv[5])
use_softcap = int(sys.argv[6]) != 0
out_path = sys.argv[7]

if softmax_cols < q_len:
    raise RuntimeError("softmax_cols must be >= q_len")

Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [q_len, head_dim])
K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [head_dim, kv_len])
V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [kv_len, value_dim])
SCALE = helper.make_tensor_value_info("SCALE", TensorProto.FLOAT, [1, 1])
MASK = helper.make_tensor_value_info("MASK", TensorProto.FLOAT, [q_len, kv_len])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [q_len, value_dim])

inputs = [Q, K, V, SCALE, MASK]
nodes = []

nodes.append(helper.make_node("MatMul", inputs=["Q", "K"], outputs=["QK"], name="QK_MatMul"))
nodes.append(helper.make_node("Mul", inputs=["QK", "SCALE"], outputs=["QKS"], name="QK_Scale"))

logits_name = "QKS"
if use_softcap:
    SOFTCAP = helper.make_tensor_value_info("SOFTCAP", TensorProto.FLOAT, [1, 1])
    inputs.append(SOFTCAP)
    nodes.append(helper.make_node("Tanh", inputs=[logits_name], outputs=["QKTanh"], name="QK_Tanh"))
    nodes.append(helper.make_node("Mul", inputs=["QKTanh", "SOFTCAP"], outputs=["QKSoftcap"], name="QK_SoftcapMul"))
    logits_name = "QKSoftcap"

nodes.append(helper.make_node("Add", inputs=[logits_name, "MASK"], outputs=["QKMask"], name="QK_AddMask"))
nodes.append(helper.make_node("Transpose", inputs=["QKMask"], outputs=["QKT"], perm=[1, 0], name="Softmax_PreTranspose"))

softmax_input = "QKT"
if softmax_cols > q_len:
    pad_const = helper.make_tensor(
        "PadConst", TensorProto.INT64, [4], [0, 0, 0, softmax_cols - q_len]
    )
    pad_val = helper.make_tensor("PadVal", TensorProto.FLOAT, [1], [0.0])
    nodes.append(
        helper.make_node(
            "Pad",
            inputs=["QKT", "PadConst", "PadVal"],
            outputs=["QKTPad"],
            name="Softmax_Pad",
            mode="constant",
        )
    )
    softmax_input = "QKTPad"

nodes.append(helper.make_node("Softmax", inputs=[softmax_input], outputs=["ProbT"], axis=0, name="Softmax_0"))

softmax_output = "ProbT"
if softmax_cols > q_len:
    starts = helper.make_tensor("SliceStarts", TensorProto.INT64, [2], [0, 0])
    ends = helper.make_tensor("SliceEnds", TensorProto.INT64, [2], [kv_len, q_len])
    axes = helper.make_tensor("SliceAxes", TensorProto.INT64, [2], [0, 1])
    steps = helper.make_tensor("SliceSteps", TensorProto.INT64, [2], [1, 1])
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ProbT", "SliceStarts", "SliceEnds", "SliceAxes", "SliceSteps"],
            outputs=["ProbTSlice"],
            name="Softmax_SliceBack",
        )
    )
    softmax_output = "ProbTSlice"

nodes.append(helper.make_node("Transpose", inputs=[softmax_output], outputs=["Prob"], perm=[1, 0], name="Softmax_PostTranspose"))
nodes.append(helper.make_node("MatMul", inputs=["Prob", "V"], outputs=["Y"], name="PV_MatMul"))

initializers = []
if softmax_cols > q_len:
    initializers.extend([pad_const, pad_val, starts, ends, axes, steps])

graph = helper.make_graph(nodes, "flash_attn_graph", inputs, [Y], initializer=initializers)
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }

    const std::filesystem::path script_path = onnx_path.parent_path() / "make_flash_attn_onnx.py";
    std::ofstream py(script_path);
    if (!py) {
        throw std::runtime_error("failed to write python script: " + script_path.string());
    }
    py << py_script;
    py.close();

    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            std::to_string(head_dim) + " " +
                            std::to_string(value_dim) + " " +
                            std::to_string(q_len) + " " +
                            std::to_string(kv_len) + " " +
                            std::to_string(softmax_cols) + " " +
                            std::to_string(use_logit_softcap ? 1 : 0) + " " +
                            quote_for_sh(onnx_path.string());
    run_system_checked(cmd);
}

static IcraftArtifacts write_icraft_compile_toml_for_zg_matmul(
    const std::filesystem::path & work_dir,
    const std::string & net_name,
    const std::filesystem::path & onnx_path,
    int64_t m,
    int64_t k,
    int64_t n) {
    (void) n;
    ensure_dir(work_dir);

    const std::filesystem::path toml_path = work_dir / (net_name + ".toml");
    const std::string jr = "./.cache/" + net_name + "_ZG/";

    const auto parsed_json = jr + net_name + "_parsed.json";
    const auto parsed_raw  = jr + net_name + "_parsed.raw";
    const auto opt_json    = jr + net_name + "_optimized.json";
    const auto opt_raw     = jr + net_name + "_optimized.raw";
    const auto quant_json  = jr + net_name + "_quantized.json";
    const auto quant_raw   = jr + net_name + "_quantized.raw";
    const auto adap_json   = jr + net_name + "_adapted.json";
    const auto adap_raw    = jr + net_name + "_adapted.raw";

    std::ofstream ofs(toml_path);
    if (!ofs) {
        throw std::runtime_error("failed to write toml: " + toml_path.string());
    }

    ofs << "[parse]\n"
        << "net_name = " << quote_for_toml(net_name) << "\n"
        << "framework = \"Onnx\"\n"
        << "network = " << quote_for_toml(onnx_path.filename().string()) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n"
        << "target = \"zhuge\"\n"
        << "inputs = [[" << m << "," << k << "],[" << k << "," << n << "]]\n"
        << "inputs_layout = \"FD;FD\"\n"
        << "inputs_dtype = \"fp32;fp32\"\n"
        << "pre_method = \"nop;nop\"\n"
        << "pre_mean = \"nop;nop\"\n"
        << "pre_scale = \"nop;nop\"\n"
        << "channel_swap = \"nop;nop\"\n\n"
        << "[optimize]\n"
        << "json = " << quote_for_toml(parsed_json) << "\n"
        << "raw = " << quote_for_toml(parsed_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n\n"
        << "[quantize]\n"
        << "json = " << quote_for_toml(opt_json) << "\n"
        << "raw = " << quote_for_toml(opt_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n"
        << "target = \"zhuge\"\n"
        << "qdtype = \"tf32\"\n"
        << "forward_mode = \"image\"\n\n"
        << "[adapt]\n"
        << "json = " << quote_for_toml(quant_json) << "\n"
        << "raw = " << quote_for_toml(quant_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n\n"
        << "[generate]\n"
        << "json = " << quote_for_toml(adap_json) << "\n"
        << "raw = " << quote_for_toml(adap_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n";

    IcraftArtifacts a;
    a.work_dir = work_dir;
    a.toml_path = toml_path;
    a.json_path = work_dir / ".cache" / (net_name + "_ZG.json");
    a.raw_path  = work_dir / ".cache" / (net_name + "_ZG.raw");
    return a;
}

static IcraftArtifacts write_icraft_compile_toml_for_zg_elementwise(
    const std::filesystem::path & work_dir,
    const std::string & net_name,
    const std::filesystem::path & onnx_path,
    const std::vector<std::vector<int64_t>> & input_shapes,
    bool bf16 = true) {
    ensure_dir(work_dir);

    const std::filesystem::path toml_path = work_dir / (net_name + ".toml");
    const std::string jr = "./.cache/" + net_name + "_ZG/";

    const auto parsed_json = jr + net_name + "_parsed.json";
    const auto parsed_raw  = jr + net_name + "_parsed.raw";
    const auto opt_json    = jr + net_name + "_optimized.json";
    const auto opt_raw     = jr + net_name + "_optimized.raw";
    const auto quant_json  = jr + net_name + "_quantized.json";
    const auto quant_raw   = jr + net_name + "_quantized.raw";
    const auto adap_json   = jr + net_name + "_adapted.json";
    const auto adap_raw    = jr + net_name + "_adapted.raw";

    std::string inputs_toml = "[";
    std::string layouts;
    std::string dtypes;
    std::string nop_method;
    std::string nop_mean;
    std::string nop_scale;
    std::string nop_swap;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        if (i != 0) {
            inputs_toml += ",";
            layouts += ";";
            dtypes += ";";
            nop_method += ";";
            nop_mean += ";";
            nop_scale += ";";
            nop_swap += ";";
        }
        inputs_toml += shape_to_toml_array(input_shapes[i]);
        layouts += shape_layout(input_shapes[i]);
        dtypes += (bf16 ? "bf16" : "fp32");
        nop_method += "nop";
        nop_mean += "nop";
        nop_scale += "nop";
        nop_swap += "nop";
    }
    inputs_toml += "]";

    std::ofstream ofs(toml_path);
    if (!ofs) {
        throw std::runtime_error("failed to write toml: " + toml_path.string());
    }

    ofs << "[parse]\n"
        << "net_name = " << quote_for_toml(net_name) << "\n"
        << "framework = \"Onnx\"\n"
        << "network = " << quote_for_toml(onnx_path.filename().string()) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n"
        << "target = \"zhuge\"\n"
        << "inputs = " << inputs_toml << "\n"
        << "inputs_layout = " << quote_for_toml(layouts) << "\n"
        << "inputs_dtype = " << quote_for_toml(dtypes) << "\n"
        << "pre_method = " << quote_for_toml(nop_method) << "\n"
        << "pre_mean = " << quote_for_toml(nop_mean) << "\n"
        << "pre_scale = " << quote_for_toml(nop_scale) << "\n"
        << "channel_swap = " << quote_for_toml(nop_swap) << "\n\n"
        << "[optimize]\n"
        << "json = " << quote_for_toml(parsed_json) << "\n"
        << "raw = " << quote_for_toml(parsed_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n\n"
        << "[quantize]\n"
        << "json = " << quote_for_toml(opt_json) << "\n"
        << "raw = " << quote_for_toml(opt_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n"
        << "target = \"zhuge\"\n"
        << "qdtype = " << (bf16 ? "\"bf16\"" : "\"tf32\"") << "\n"
        << "forward_mode = \"image\"\n\n"
        << "[adapt]\n"
        << "json = " << quote_for_toml(quant_json) << "\n"
        << "raw = " << quote_for_toml(quant_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n\n"
        << "[generate]\n"
        << "json = " << quote_for_toml(adap_json) << "\n"
        << "raw = " << quote_for_toml(adap_raw) << "\n"
        << "jr_path = " << quote_for_toml(jr) << "\n";

    IcraftArtifacts a;
    a.work_dir = work_dir;
    a.toml_path = toml_path;
    a.json_path = work_dir / ".cache" / (net_name + "_ZG.json");
    a.raw_path  = work_dir / ".cache" / (net_name + "_ZG.raw");
    return a;
}

static void run_icraft_compile(const IcraftArtifacts & artifacts) {
#if defined(__aarch64__) || defined(_M_ARM64)
    std::cout<< "No, you cannot compile with icraft on ARM platform" << std::endl;
    throw std::runtime_error("icraft compile not supported on ARM");
#else
    const std::string cmd =
        "cd " + quote_for_sh(artifacts.work_dir.string()) +
        " && icraft compile " + quote_for_sh(artifacts.toml_path.filename().string());
    run_system_checked(cmd);
#endif
}

static std::pair<std::filesystem::path, std::filesystem::path> find_generated_zg_json_raw(
    const std::filesystem::path & work_dir,
    const std::string & net_name) {
    const std::filesystem::path cache_root = work_dir / ".cache";
    const std::filesystem::path cand_json = cache_root / (net_name + "_ZG.json");
    const std::filesystem::path cand_raw  = cache_root / (net_name + "_ZG.raw");
    if (std::filesystem::exists(cand_json) && std::filesystem::exists(cand_raw)) {
        return {cand_json, cand_raw};
    }

    if (std::filesystem::exists(cache_root)) {
        std::filesystem::path found_json;
        std::filesystem::path found_raw;
        for (const auto & de : std::filesystem::recursive_directory_iterator(cache_root)) {
            if (!de.is_regular_file()) {
                continue;
            }
            const auto fn = de.path().filename().string();
            if (fn == net_name + "_ZG.json") {
                found_json = de.path();
            } else if (fn == net_name + "_ZG.raw") {
                found_raw = de.path();
            }
        }
        if (!found_json.empty() && !found_raw.empty()) {
            return {found_json, found_raw};
        }
    }

    throw std::runtime_error("cannot find generated *_ZG.json/raw under: " + cache_root.string());
}

} // namespace

void preload_zg_cache(const std::filesystem::path & work_root) {
    ensure_dir(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto root_key = root_abs.string();

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        if (g_preloaded_roots_abs.find(root_key) != g_preloaded_roots_abs.end()) {
            return;
        }
    }

    std::unordered_map<std::string, CachedMatmulEntry> loaded;
    for (const auto & de : std::filesystem::recursive_directory_iterator(root_abs)) {
        if (!de.is_regular_file()) {
            continue;
        }
        const auto fn = de.path().filename().string();
        const std::string suffix = "_ZG.json";
        if (fn.size() <= suffix.size() || fn.substr(fn.size() - suffix.size()) != suffix) {
            continue;
        }

        const auto net_name = fn.substr(0, fn.size() - suffix.size());
        // Parse matmul dims when available; other net types (elementwise, flash_attn) keep m=k=n=0.
        // All compiled binaries are preloaded so g_cache is warm before the first session creation.
        int64_t m = 0, k = 0, n = 0;
        parse_matmul_dims_from_net_name(net_name, &m, &k, &n);

        const auto raw_path = de.path().parent_path() / (net_name + "_ZG.raw");
        if (!std::filesystem::exists(raw_path)) {
            continue;
        }

        try {
            auto network = icraft::xir::Network::CreateFromJsonFile(de.path().string());
            network.loadParamsFromFile(raw_path.string());

            CachedMatmulEntry entry;
            entry.net_name = net_name;
            entry.m = m;
            entry.k = k;
            entry.n = n;
            entry.network = std::move(network);
            entry.json_path = de.path();
            entry.raw_path = raw_path;

            loaded[make_root_net_key(root_abs, net_name)] = std::move(entry);
        } catch (...) {
            // ignore broken cache
        }
    }

    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for (auto & kv : loaded) {
        g_cache[kv.first] = std::move(kv.second);
    }
    g_preloaded_roots_abs.insert(root_key);
}

MatmulZgNetworkBundle get_or_compile_matmul_zg_network(
    const std::filesystem::path & work_root,
    int64_t m,
    int64_t k,
    int64_t n) {
    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::runtime_error("get_or_compile_matmul_zg_network: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name = make_matmul_net_name(m, k, n);
    const auto cache_key = make_root_net_key(root_abs, net_name);

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cache_key);
        if (it != g_cache.end()) {
            MatmulZgNetworkBundle out;
            out.net_name = it->second.net_name;
            out.network = it->second.network;
            out.ram_cache_hit = true;
            out.compiled_now = false;
            return out;
        }
    }

    const std::filesystem::path work_dir = root_abs / net_name;
    ensure_dir(work_dir);

    const auto onnx_path = work_dir / (net_name + ".onnx");
    (void) build_matmul_onnx(onnx_path, m, k, n);
    const auto artifacts = write_icraft_compile_toml_for_zg_matmul(work_dir, net_name, onnx_path, m, k, n);
    run_icraft_compile(artifacts);
    auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);

    auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
    network.loadParamsFromFile(raw_path.string());

    CachedMatmulEntry entry;
    entry.net_name = net_name;
    entry.m = m;
    entry.k = k;
    entry.n = n;
    entry.network = network;
    entry.json_path = json_path;
    entry.raw_path = raw_path;

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cache_key] = entry;
    }

    MatmulZgNetworkBundle out;
    out.net_name = net_name;
    out.network = std::move(network);
    out.ram_cache_hit = false;
    out.compiled_now = true;
    return out;
}

ElementwiseZgNetworkBundle get_or_compile_elementwise_zg_network(
    const std::filesystem::path & work_root,
    ElementwiseZgOp op,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("get_or_compile_elementwise_zg_network: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name = make_elementwise_net_name(op, rows, cols);
    const auto cache_key = make_root_net_key(root_abs, net_name);

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cache_key);
        if (it != g_cache.end()) {
            ElementwiseZgNetworkBundle out;
            out.net_name = it->second.net_name;
            out.network = it->second.network;
            out.ram_cache_hit = true;
            out.compiled_now = false;
            return out;
        }
    }

    const std::filesystem::path work_dir = root_abs / net_name;
    ensure_dir(work_dir);

    // Disk cache hit path: reuse previously generated json/raw across process runs.
    // This avoids repeating icraft compile every new llama-cli process.
    try {
        auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);
        auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
        network.loadParamsFromFile(raw_path.string());

        CachedMatmulEntry entry;
        entry.net_name = net_name;
        entry.m = rows;
        entry.k = cols;
        entry.n = 1;
        entry.network = network;
        entry.json_path = json_path;
        entry.raw_path = raw_path;
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[cache_key] = entry;
        }

        ElementwiseZgNetworkBundle out;
        out.net_name = net_name;
        out.network = std::move(network);
        out.ram_cache_hit = false;
        out.compiled_now = false;
        return out;
    } catch (...) {
        // cache miss or broken cache; fall through to compile.
    }

    const auto onnx_path = work_dir / (net_name + ".onnx");
    build_elementwise_onnx(onnx_path, op, rows, cols);

    std::vector<std::vector<int64_t>> input_shapes = {{rows, cols}};
    if (op == ElementwiseZgOp::ADD || op == ElementwiseZgOp::MUL) {
        input_shapes.push_back({rows, cols});
    } else if (op == ElementwiseZgOp::SCALE || op == ElementwiseZgOp::RMS_NORM ||
               op == ElementwiseZgOp::CPY || op == ElementwiseZgOp::DUP) {
        input_shapes.push_back({1, 1});
    }

    const auto artifacts = write_icraft_compile_toml_for_zg_elementwise(work_dir, net_name, onnx_path, input_shapes);
    run_icraft_compile(artifacts);
    auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);

    auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
    network.loadParamsFromFile(raw_path.string());

    CachedMatmulEntry entry;
    entry.net_name = net_name;
    entry.m = rows;
    entry.k = cols;
    entry.n = 1;
    entry.network = network;
    entry.json_path = json_path;
    entry.raw_path = raw_path;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cache_key] = entry;
    }

    ElementwiseZgNetworkBundle out;
    out.net_name = net_name;
    out.network = std::move(network);
    out.ram_cache_hit = false;
    out.compiled_now = true;
    return out;
}

FlashAttnZgNetworkBundle get_or_compile_flash_attn_zg_network(
    const std::filesystem::path & work_root,
    int64_t head_dim,
    int64_t value_dim,
    int64_t q_len,
    int64_t kv_len,
    int64_t softmax_cols,
    bool use_logit_softcap) {
    if (head_dim <= 0 || value_dim <= 0 || q_len <= 0 || kv_len <= 0 || softmax_cols < q_len) {
        throw std::runtime_error("get_or_compile_flash_attn_zg_network: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name = make_flash_attn_net_name(head_dim, value_dim, q_len, kv_len, softmax_cols, use_logit_softcap);
    const auto cache_key = make_root_net_key(root_abs, net_name);

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cache_key);
        if (it != g_cache.end()) {
            FlashAttnZgNetworkBundle out;
            out.net_name = it->second.net_name;
            out.network = it->second.network;
            out.ram_cache_hit = true;
            out.compiled_now = false;
            return out;
        }
    }

    const std::filesystem::path work_dir = root_abs / net_name;
    ensure_dir(work_dir);

    // Disk cache hit path: reuse previously generated json/raw across process runs.
    try {
        auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);
        auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
        network.loadParamsFromFile(raw_path.string());

        CachedMatmulEntry entry;
        entry.net_name = net_name;
        entry.m = q_len;
        entry.k = head_dim;
        entry.n = kv_len;
        entry.network = network;
        entry.json_path = json_path;
        entry.raw_path = raw_path;
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[cache_key] = entry;
        }

        FlashAttnZgNetworkBundle out;
        out.net_name = net_name;
        out.network = std::move(network);
        out.ram_cache_hit = false;
        out.compiled_now = false;
        return out;
    } catch (...) {
        // cache miss or broken cache; fall through to compile.
    }

    const auto onnx_path = work_dir / (net_name + ".onnx");
    build_flash_attn_onnx(onnx_path, head_dim, value_dim, q_len, kv_len, softmax_cols, use_logit_softcap);

    std::vector<std::vector<int64_t>> input_shapes = {
        {q_len, head_dim},
        {head_dim, kv_len},
        {kv_len, value_dim},
        {1, 1},          // scale
        {q_len, kv_len}, // mask
    };
    if (use_logit_softcap) {
        input_shapes.push_back({1, 1});
    }

    const auto artifacts = write_icraft_compile_toml_for_zg_elementwise(work_dir, net_name, onnx_path, input_shapes, false);
    run_icraft_compile(artifacts);
    auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);

    auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
    network.loadParamsFromFile(raw_path.string());

    CachedMatmulEntry entry;
    entry.net_name = net_name;
    entry.m = q_len;
    entry.k = head_dim;
    entry.n = kv_len;
    entry.network = network;
    entry.json_path = json_path;
    entry.raw_path = raw_path;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cache_key] = entry;
    }

    FlashAttnZgNetworkBundle out;
    out.net_name = net_name;
    out.network = std::move(network);
    out.ram_cache_hit = false;
    out.compiled_now = true;
    return out;
}

ElementwiseZgNetworkBundle get_or_compile_bf16_bridge_zg_network(
    const std::filesystem::path & work_root,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("get_or_compile_bf16_bridge_zg_network: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name = make_bf16_bridge_net_name(rows, cols);
    const auto cache_key = make_root_net_key(root_abs, net_name);

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cache_key);
        if (it != g_cache.end()) {
            ElementwiseZgNetworkBundle out;
            out.net_name = it->second.net_name;
            out.network = it->second.network;
            out.ram_cache_hit = true;
            out.compiled_now = false;
            return out;
        }
    }

    const std::filesystem::path work_dir = root_abs / net_name;
    ensure_dir(work_dir);

    // Disk cache hit path
    try {
        auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);
        auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
        network.loadParamsFromFile(raw_path.string());

        CachedMatmulEntry entry;
        entry.net_name = net_name;
        entry.m = rows;
        entry.k = cols;
        entry.n = 1;
        entry.network = network;
        entry.json_path = json_path;
        entry.raw_path = raw_path;
        {
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[cache_key] = entry;
        }

        ElementwiseZgNetworkBundle out;
        out.net_name = net_name;
        out.network = std::move(network);
        out.ram_cache_hit = false;
        out.compiled_now = false;
        return out;
    } catch (...) {
        // cache miss; fall through to compile
    }

    const auto onnx_path = work_dir / (net_name + ".onnx");
    build_elementwise_onnx(onnx_path, ElementwiseZgOp::BF16_BRIDGE, rows, cols);

    // Bridge has single BF16 input (ONE is baked as ONNX initializer)
    std::vector<std::vector<int64_t>> input_shapes = {{rows, cols}};
    const auto artifacts = write_icraft_compile_toml_for_zg_elementwise(work_dir, net_name, onnx_path, input_shapes);
    run_icraft_compile(artifacts);
    auto [json_path, raw_path] = find_generated_zg_json_raw(work_dir, net_name);

    auto network = icraft::xir::Network::CreateFromJsonFile(json_path.string());
    network.loadParamsFromFile(raw_path.string());

    CachedMatmulEntry entry;
    entry.net_name = net_name;
    entry.m = rows;
    entry.k = cols;
    entry.n = 1;
    entry.network = network;
    entry.json_path = json_path;
    entry.raw_path = raw_path;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cache_key] = entry;
    }

    ElementwiseZgNetworkBundle out;
    out.net_name = net_name;
    out.network = std::move(network);
    out.ram_cache_hit = false;
    out.compiled_now = true;
    return out;
}

} // namespace ggml::fmsh::netmake
