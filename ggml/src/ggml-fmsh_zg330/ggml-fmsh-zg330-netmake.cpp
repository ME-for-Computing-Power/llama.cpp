#include "ggml-fmsh-zg330-netmake.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
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
std::mutex g_compile_mutex;
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

static std::string make_matmul_net_name(int64_t m, int64_t k, int64_t n, bool bf16 = false) {
    return std::string("matmul_") + (bf16 ? "bf16_" : "") +
           std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
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

static std::string make_fused_ew_net_name(
    const std::vector<ElementwiseZgOp> & ops,
    int64_t rows,
    int64_t cols) {
    std::string name = "fused";
    for (auto op : ops) {
        name += "_" + elementwise_op_name(op);
    }
    name += "_bf16_" + std::to_string(rows) + "x" + std::to_string(cols);
    return name;
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
    one_init = helper.make_tensor("ONE", TensorProto.FLOAT, [1, 1], [1.0])
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
    # ZG330 Mean/Sum only support non-channel (non-last) dimensions.
    # In FD layout [rows, cols], cols is the channel axis — cannot ReduceMean on it directly.
    # Workaround per icraft docs: Transpose to move cols to axis-0, ReduceSum on axis-0
    # (non-channel in the transposed view), Transpose back, then scale by 1/cols.
    E = helper.make_tensor_value_info("E", TensorProto.FLOAT, [1, 1])
    inputs.append(E)
    inv_n = helper.make_tensor("INV_N", TensorProto.FLOAT, [1, 1], [1.0 / cols])
    x2   = helper.make_node("Mul",       ["X",   "X"],    ["X2"],  name="Mul_X2")
    tp0  = helper.make_node("Transpose", ["X2"],           ["X2T"], name="Tp0", perm=[1, 0])
    rsum = helper.make_node("ReduceSum", ["X2T"],          ["ST"],  name="RSum", axes=[0], keepdims=1)
    tp1  = helper.make_node("Transpose", ["ST"],           ["S"],   name="Tp1", perm=[1, 0])
    scl  = helper.make_node("Mul",       ["S",  "INV_N"], ["M"],   name="ScaleMean")
    aeps = helper.make_node("Add",       ["M",   "E"],    ["ME"],  name="Add_EPS")
    sqt  = helper.make_node("Sqrt",      ["ME"],           ["R"],   name="Sqrt_0")
    div  = helper.make_node("Div",       ["X",   "R"],    ["Y"],   name="Div_0")
    nodes.extend([x2, tp0, rsum, tp1, scl, aeps, sqt, div])
    graph = helper.make_graph(nodes, f"{op}_graph", inputs, [Y], initializer=[inv_n])
    model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    sys.exit(0)
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

static void build_fused_elementwise_onnx(
    const std::filesystem::path & onnx_path,
    const std::vector<ElementwiseZgOp> & ops,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0 || ops.empty()) {
        throw std::runtime_error("build_fused_elementwise_onnx: invalid args");
    }

    ensure_dir(onnx_path.parent_path());

    std::string ops_csv;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i != 0) ops_csv += ",";
        ops_csv += elementwise_op_name(ops[i]);
    }

    const std::string py_script = R"PY(
import sys
import onnx
from onnx import helper, TensorProto

ops = sys.argv[1].split(',')
rows = int(sys.argv[2])
cols = int(sys.argv[3])
out_path = sys.argv[4]

X_val = helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows, cols])
Y_val = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows, cols])
inputs = [X_val]
initializers = []
nodes = []
current = "X"
secondary_idx = 0

for k, op in enumerate(ops):
    out_name = "mid_" + str(k) if k < len(ops) - 1 else "Y"
    if op in ("add", "mul"):
        b_name = "B_" + str(secondary_idx)
        secondary_idx += 1
        b_val = helper.make_tensor_value_info(b_name, TensorProto.FLOAT, [rows, cols])
        inputs.append(b_val)
        node_type = "Add" if op == "add" else "Mul"
        nodes.append(helper.make_node(node_type, inputs=[current, b_name], outputs=[out_name], name=f"{node_type}_{k}"))
        current = out_name
    elif op == "scale":
        b_name = "B_" + str(secondary_idx)
        secondary_idx += 1
        b_val = helper.make_tensor_value_info(b_name, TensorProto.FLOAT, [1, 1])
        inputs.append(b_val)
        nodes.append(helper.make_node("Mul", inputs=[current, b_name], outputs=[out_name], name=f"Mul_{k}"))
        current = out_name
    elif op == "rmsnorm":
        eps_name = "B_" + str(secondary_idx)
        secondary_idx += 1
        eps_val = helper.make_tensor_value_info(eps_name, TensorProto.FLOAT, [1, 1])
        inputs.append(eps_val)
        inv_n_name = "INV_N_" + str(k)
        inv_n = helper.make_tensor(inv_n_name, TensorProto.FLOAT, [1, 1], [1.0 / cols])
        initializers.append(inv_n)
        m_n   = f"m_{k}"
        me_n  = f"me_{k}"
        r_n   = f"r_{k}"
        x2_n  = f"x2_{k}"
        x2t_n = f"x2t_{k}"
        st_n  = f"st_{k}"
        s_n   = f"s_{k}"
        # Network input X has explicit FD layout; Transpose workaround is safe.
        # RMS_NORM is only ever at k==0 (enforced by chain builder).
        nodes.extend([
            helper.make_node("Mul",       [current, current],    [x2_n],     name=f"Mul_X2_{k}"),
            helper.make_node("Transpose", [x2_n],                [x2t_n],    name=f"Tp0_{k}", perm=[1, 0]),
            helper.make_node("ReduceSum", [x2t_n],               [st_n],     name=f"RSum_{k}", axes=[0], keepdims=1),
            helper.make_node("Transpose", [st_n],                [s_n],      name=f"Tp1_{k}", perm=[1, 0]),
            helper.make_node("Mul",       [s_n, inv_n_name],     [m_n],      name=f"ScaleMean_{k}"),
            helper.make_node("Add",       [m_n, eps_name],       [me_n],     name=f"Add_EPS_{k}"),
            helper.make_node("Sqrt",      [me_n],                [r_n],      name=f"Sqrt_{k}"),
            helper.make_node("Div",       [current, r_n],        [out_name], name=f"Div_{k}"),
        ])
        current = out_name
    else:
        raise RuntimeError(f"unsupported op in fused chain: {op}")

graph = helper.make_graph(nodes, "fused_ew_graph", inputs, [Y_val], initializer=initializers)
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }

    const std::filesystem::path script_path = onnx_path.parent_path() / "make_fused_ew_onnx.py";
    std::ofstream py(script_path);
    if (!py) {
        throw std::runtime_error("failed to write python script: " + script_path.string());
    }
    py << py_script;
    py.close();

    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            quote_for_sh(ops_csv) + " " +
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
    int64_t n,
    bool bf16 = false) {
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
        << "inputs_dtype = " << (bf16 ? "\"bf16;bf16\"" : "\"fp32;fp32\"") << "\n"
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
    auto try_find_once = [&]() -> std::pair<std::filesystem::path, std::filesystem::path> {
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
        return {};
    };

    for (int i = 0; i < 40; ++i) {
        auto found = try_find_once();
        if (!found.first.empty() && !found.second.empty()) {
            return found;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
    int64_t n,
    bool bf16) {
    if (m <= 0 || k <= 0 || n <= 0) {
        throw std::runtime_error("get_or_compile_matmul_zg_network: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name = make_matmul_net_name(m, k, n, bf16);
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
    (void) build_matmul_onnx(onnx_path, m, k, n);  // ONNX stays FLOAT; icraft toml drives bf16 quant
    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
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
    const auto artifacts = write_icraft_compile_toml_for_zg_matmul(work_dir, net_name, onnx_path, m, k, n, bf16);
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
    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
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

ElementwiseZgNetworkBundle get_or_compile_fused_ew_zg_network(
    const std::filesystem::path & work_root,
    const std::vector<ElementwiseZgOp> & ops,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0 || ops.empty()) {
        throw std::runtime_error("get_or_compile_fused_ew_zg_network: invalid args");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name = make_fused_ew_net_name(ops, rows, cols);
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
    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
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

    // Disk cache hit: reuse previously compiled json/raw.
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
    build_fused_elementwise_onnx(onnx_path, ops, rows, cols);

    std::vector<std::vector<int64_t>> input_shapes = {{rows, cols}};
    for (auto op : ops) {
        if (op == ElementwiseZgOp::ADD || op == ElementwiseZgOp::MUL) {
            input_shapes.push_back({rows, cols});
        } else if (op == ElementwiseZgOp::SCALE || op == ElementwiseZgOp::RMS_NORM) {
            input_shapes.push_back({1, 1});
        }
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

    ElementwiseZgNetworkBundle fused_out;
    fused_out.net_name = net_name;
    fused_out.network = std::move(network);
    fused_out.ram_cache_hit = false;
    fused_out.compiled_now = true;
    return fused_out;
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
    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
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
    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
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

// ── RMS_NORM split: pre (reduce to r²) ──────────────────────────────────────
// ONNX: X[R,C], eps[1,1] → Mul(X,X) → Transpose → ReduceSum → Transpose → Mul(1/C) → Add(eps) → r_sq[R,1]
static void build_rmsnorm_pre_onnx(
    const std::filesystem::path & onnx_path,
    int64_t rows,
    int64_t cols) {
    ensure_dir(onnx_path.parent_path());

    const std::string py_script = R"PY(
import sys, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
rows = int(sys.argv[1]); cols = int(sys.argv[2]); out_path = sys.argv[3]
inv_n_val = float(1.0 / cols)
X   = helper.make_tensor_value_info("X",   TensorProto.FLOAT, [rows, cols])
EPS = helper.make_tensor_value_info("EPS", TensorProto.FLOAT, [1, 1])
Y   = helper.make_tensor_value_info("Y",   TensorProto.FLOAT, [rows, 1])
inv_n = numpy_helper.from_array(np.array([[inv_n_val]], dtype=np.float32), name="INV_N")
nodes = [
    helper.make_node("Mul",       ["X", "X"],       ["x2"],   name="Mul_X2"),
    helper.make_node("Transpose", ["x2"],            ["x2t"],  name="Tp0",    perm=[1, 0]),
    helper.make_node("ReduceSum", ["x2t"],           ["st"],   name="RSum",   axes=[0], keepdims=1),
    helper.make_node("Transpose", ["st"],            ["s"],    name="Tp1",    perm=[1, 0]),
    helper.make_node("Mul",       ["s", "INV_N"],   ["m"],    name="Scale"),
    helper.make_node("Add",       ["m", "EPS"],     ["Y"],    name="AddEPS"),
]
graph = helper.make_graph(nodes, "rmsnorm_pre_graph", [X, EPS], [Y], initializer=[inv_n])
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }
    const std::filesystem::path script_path = onnx_path.parent_path() / "make_rmsnorm_pre_onnx.py";
    std::ofstream py(script_path);
    if (!py) { throw std::runtime_error("failed to write python script: " + script_path.string()); }
    py << py_script; py.close();
    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            std::to_string(rows) + " " + std::to_string(cols) + " " +
                            quote_for_sh(onnx_path.string());
    run_system_checked(cmd);
}

// ── RMS_NORM split: post (multiply X by r_inv per row) ──────────────────────
// ONNX: X[R,C], r_inv[R,1] → Mul(X, r_inv) → Y[R,C]
static void build_rmsnorm_post_onnx(
    const std::filesystem::path & onnx_path,
    int64_t rows,
    int64_t cols) {
    ensure_dir(onnx_path.parent_path());

    const std::string py_script = R"PY(
import sys, onnx
from onnx import helper, TensorProto
rows = int(sys.argv[1]); cols = int(sys.argv[2]); out_path = sys.argv[3]
X     = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [rows, cols])
R_INV = helper.make_tensor_value_info("R_INV", TensorProto.FLOAT, [rows, 1])
Y     = helper.make_tensor_value_info("Y",     TensorProto.FLOAT, [rows, cols])
nodes = [
    helper.make_node("Mul", ["X", "R_INV"], ["Y"], name="Mul_RInv"),
]
graph = helper.make_graph(nodes, "rmsnorm_post_graph", [X, R_INV], [Y])
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }
    const std::filesystem::path script_path = onnx_path.parent_path() / "make_rmsnorm_post_onnx.py";
    std::ofstream py(script_path);
    if (!py) { throw std::runtime_error("failed to write python script: " + script_path.string()); }
    py << py_script; py.close();
    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            std::to_string(rows) + " " + std::to_string(cols) + " " +
                            quote_for_sh(onnx_path.string());
    run_system_checked(cmd);
}

RmsNormSplitNetworkBundle get_or_compile_rmsnorm_split_zg_networks(
    const std::filesystem::path & work_root,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("get_or_compile_rmsnorm_split_zg_networks: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    const auto net_name_pre  = "rmsnorm_pre_bf16_"  + std::to_string(rows) + "x" + std::to_string(cols);
    const auto net_name_post = "rmsnorm_post_bf16_" + std::to_string(rows) + "x" + std::to_string(cols);
    const auto key_pre  = make_root_net_key(root_abs, net_name_pre);
    const auto key_post = make_root_net_key(root_abs, net_name_post);

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it_pre  = g_cache.find(key_pre);
        const auto it_post = g_cache.find(key_post);
        if (it_pre != g_cache.end() && it_post != g_cache.end()) {
            RmsNormSplitNetworkBundle out;
            out.net_name_pre  = it_pre->second.net_name;
            out.net_name_post = it_post->second.net_name;
            out.network_pre   = it_pre->second.network;
            out.network_post  = it_post->second.network;
            out.ram_cache_hit = true;
            out.compiled_now  = false;
            return out;
        }
    }

    const auto work_dir_pre  = root_abs / net_name_pre;
    const auto work_dir_post = root_abs / net_name_post;
    ensure_dir(work_dir_pre);
    ensure_dir(work_dir_post);

    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it_pre  = g_cache.find(key_pre);
        const auto it_post = g_cache.find(key_post);
        if (it_pre != g_cache.end() && it_post != g_cache.end()) {
            RmsNormSplitNetworkBundle out;
            out.net_name_pre  = it_pre->second.net_name;
            out.net_name_post = it_post->second.net_name;
            out.network_pre   = it_pre->second.network;
            out.network_post  = it_post->second.network;
            out.ram_cache_hit = true;
            out.compiled_now  = false;
            return out;
        }
    }

    // Compile pre network.
    icraft::xir::Network network_pre;
    {
        try {
            auto [jp, rp] = find_generated_zg_json_raw(work_dir_pre, net_name_pre);
            network_pre = icraft::xir::Network::CreateFromJsonFile(jp.string());
            network_pre.loadParamsFromFile(rp.string());
            CachedMatmulEntry e; e.net_name = net_name_pre; e.network = network_pre;
            e.json_path = jp; e.raw_path = rp;
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[key_pre] = e;
        } catch (...) {
            const auto onnx_pre = work_dir_pre / (net_name_pre + ".onnx");
            build_rmsnorm_pre_onnx(onnx_pre, rows, cols);
            // pre: inputs X[R,C] and EPS[1,1]
            const auto art = write_icraft_compile_toml_for_zg_elementwise(
                work_dir_pre, net_name_pre, onnx_pre, {{rows, cols}, {1, 1}});
            run_icraft_compile(art);
            auto [jp, rp] = find_generated_zg_json_raw(work_dir_pre, net_name_pre);
            network_pre = icraft::xir::Network::CreateFromJsonFile(jp.string());
            network_pre.loadParamsFromFile(rp.string());
            CachedMatmulEntry e; e.net_name = net_name_pre; e.network = network_pre;
            e.m = rows; e.k = cols; e.n = 1; e.json_path = jp; e.raw_path = rp;
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[key_pre] = e;
        }
    }

    // Compile post network.
    icraft::xir::Network network_post;
    {
        try {
            auto [jp, rp] = find_generated_zg_json_raw(work_dir_post, net_name_post);
            network_post = icraft::xir::Network::CreateFromJsonFile(jp.string());
            network_post.loadParamsFromFile(rp.string());
            CachedMatmulEntry e; e.net_name = net_name_post; e.network = network_post;
            e.json_path = jp; e.raw_path = rp;
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[key_post] = e;
        } catch (...) {
            const auto onnx_post = work_dir_post / (net_name_post + ".onnx");
            build_rmsnorm_post_onnx(onnx_post, rows, cols);
            // post: inputs X[R,C] and R_INV[R,1]
            const auto art = write_icraft_compile_toml_for_zg_elementwise(
                work_dir_post, net_name_post, onnx_post, {{rows, cols}, {rows, 1}});
            run_icraft_compile(art);
            auto [jp, rp] = find_generated_zg_json_raw(work_dir_post, net_name_post);
            network_post = icraft::xir::Network::CreateFromJsonFile(jp.string());
            network_post.loadParamsFromFile(rp.string());
            CachedMatmulEntry e; e.net_name = net_name_post; e.network = network_post;
            e.m = rows; e.k = cols; e.n = 1; e.json_path = jp; e.raw_path = rp;
            std::lock_guard<std::mutex> lock(g_cache_mutex);
            g_cache[key_post] = e;
        }
    }

    RmsNormSplitNetworkBundle out;
    out.net_name_pre  = net_name_pre;
    out.net_name_post = net_name_post;
    out.network_pre   = std::move(network_pre);
    out.network_post  = std::move(network_post);
    out.ram_cache_hit = false;
    out.compiled_now  = true;
    return out;
}

// ── ROPE NeoX: X[N,d], THETA[N,d/2] → Y[N,d] ─────────────────────────────
// THETA contains rotation angles (pre-reduced to [-π,π] by host fmod).
// cos and sin are computed on ZG330 via degree-12 Horner polynomials:
//   cos(θ) = Horner(θ², [1/479001600, -1/3628800, 1/40320, -1/720, 1/24, -1/2, 1])
//   sin(θ) = θ · Horner(θ², [1/6227020800, -1/39916800, 1/362880, -1/5040, 1/120, -1/6, 1])
// All ops (Mul/Add/Sub/Slice/Concat) lower to ZG330 HardOps — no HostBackend needed.
static void build_rope_onnx(
    const std::filesystem::path & onnx_path,
    int64_t rows,
    int64_t cols) {
    if (cols % 2 != 0) {
        throw std::runtime_error("build_rope_onnx: cols must be even (got " + std::to_string(cols) + ")");
    }
    ensure_dir(onnx_path.parent_path());

    const std::string py_script = R"PY(
import sys, onnx, numpy as np
from onnx import helper, TensorProto, numpy_helper
rows = int(sys.argv[1]); cols = int(sys.argv[2]); out_path = sys.argv[3]
half = cols // 2

X     = helper.make_tensor_value_info("X",     TensorProto.FLOAT, [rows, cols])
THETA = helper.make_tensor_value_info("THETA", TensorProto.FLOAT, [rows, half])
Y     = helper.make_tensor_value_info("Y",     TensorProto.FLOAT, [rows, cols])

def scalar(name, val):
    return numpy_helper.from_array(np.array([val], dtype=np.float32), name=name)

# Horner coefficients: cos(x) = Horner(x², cc), degree-12 polynomial
# cc[0]*x^12 + cc[1]*x^10 + ... + cc[6]*x^0
cc = [1.0/479001600.0, -1.0/3628800.0, 1.0/40320.0, -1.0/720.0, 1.0/24.0, -0.5, 1.0]
# sin(x) = x * Horner(x², sc), inner polynomial
sc = [1.0/6227020800.0, -1.0/39916800.0, 1.0/362880.0, -1.0/5040.0, 1.0/120.0, -1.0/6.0, 1.0]

inits = [scalar(f"cc{i}", v) for i, v in enumerate(cc)]
inits += [scalar(f"sc{i}", v) for i, v in enumerate(sc)]

starts0 = numpy_helper.from_array(np.array([0, 0],       dtype=np.int64), name="starts0")
ends0   = numpy_helper.from_array(np.array([rows, half], dtype=np.int64), name="ends0")
starts1 = numpy_helper.from_array(np.array([0, half],    dtype=np.int64), name="starts1")
ends1   = numpy_helper.from_array(np.array([rows, cols], dtype=np.int64), name="ends1")
axes_01 = numpy_helper.from_array(np.array([0, 1],       dtype=np.int64), name="axes_01")
steps_1 = numpy_helper.from_array(np.array([1, 1],       dtype=np.int64), name="steps_1")
inits += [starts0, ends0, starts1, ends1, axes_01, steps_1]

nodes = []

# th2 = THETA * THETA  (shared by cos and sin Horner)
nodes.append(helper.make_node("Mul", ["THETA", "THETA"], ["th2"]))

# cos Horner: ct = (((((cc0*th2 + cc1)*th2 + cc2)*th2 + cc3)*th2 + cc4)*th2 + cc5)*th2 + cc6
nodes.append(helper.make_node("Mul", ["cc0",  "th2"], ["ct0"]))
nodes.append(helper.make_node("Add", ["ct0",  "cc1"], ["ct1"]))
nodes.append(helper.make_node("Mul", ["ct1",  "th2"], ["ct2"]))
nodes.append(helper.make_node("Add", ["ct2",  "cc2"], ["ct3"]))
nodes.append(helper.make_node("Mul", ["ct3",  "th2"], ["ct4"]))
nodes.append(helper.make_node("Add", ["ct4",  "cc3"], ["ct5"]))
nodes.append(helper.make_node("Mul", ["ct5",  "th2"], ["ct6"]))
nodes.append(helper.make_node("Add", ["ct6",  "cc4"], ["ct7"]))
nodes.append(helper.make_node("Mul", ["ct7",  "th2"], ["ct8"]))
nodes.append(helper.make_node("Add", ["ct8",  "cc5"], ["ct9"]))
nodes.append(helper.make_node("Mul", ["ct9",  "th2"], ["ct10"]))
nodes.append(helper.make_node("Add", ["ct10", "cc6"], ["cos_t"]))

# sin Horner: st_inner = (((((sc0*th2+sc1)*th2+sc2)*th2+sc3)*th2+sc4)*th2+sc5)*th2+sc6
#             sin_t = THETA * st_inner
nodes.append(helper.make_node("Mul", ["sc0",    "th2"], ["st0"]))
nodes.append(helper.make_node("Add", ["st0",    "sc1"], ["st1"]))
nodes.append(helper.make_node("Mul", ["st1",    "th2"], ["st2"]))
nodes.append(helper.make_node("Add", ["st2",    "sc2"], ["st3"]))
nodes.append(helper.make_node("Mul", ["st3",    "th2"], ["st4"]))
nodes.append(helper.make_node("Add", ["st4",    "sc3"], ["st5"]))
nodes.append(helper.make_node("Mul", ["st5",    "th2"], ["st6"]))
nodes.append(helper.make_node("Add", ["st6",    "sc4"], ["st7"]))
nodes.append(helper.make_node("Mul", ["st7",    "th2"], ["st8"]))
nodes.append(helper.make_node("Add", ["st8",    "sc5"], ["st9"]))
nodes.append(helper.make_node("Mul", ["st9",    "th2"], ["st10"]))
nodes.append(helper.make_node("Add", ["st10",   "sc6"], ["st_inner"]))
nodes.append(helper.make_node("Mul", ["THETA",  "st_inner"], ["sin_t"]))

# NeoX rotation: x0*cos - x1*sin, x0*sin + x1*cos
nodes.append(helper.make_node("Slice", ["X", "starts0", "ends0", "axes_01", "steps_1"], ["x0"]))
nodes.append(helper.make_node("Slice", ["X", "starts1", "ends1", "axes_01", "steps_1"], ["x1"]))
nodes.append(helper.make_node("Mul",   ["x0", "cos_t"], ["x0cos"]))
nodes.append(helper.make_node("Mul",   ["x1", "sin_t"], ["x1sin"]))
nodes.append(helper.make_node("Mul",   ["x0", "sin_t"], ["x0sin"]))
nodes.append(helper.make_node("Mul",   ["x1", "cos_t"], ["x1cos"]))
nodes.append(helper.make_node("Sub",   ["x0cos", "x1sin"], ["out0"]))
nodes.append(helper.make_node("Add",   ["x0sin", "x1cos"], ["out1"]))
nodes.append(helper.make_node("Concat", ["out0", "out1"], ["Y"], axis=1))

graph = helper.make_graph(nodes, "rope_neox_graph", [X, THETA], [Y], initializer=inits)
model = helper.make_model(graph, producer_name="ggml_fmsh_netmake", opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out_path)
)PY";

    if (!run_system_ok("python3 -c " + quote_for_sh("import onnx") + " >/dev/null 2>&1")) {
        throw std::runtime_error("python package missing: onnx");
    }
    const std::filesystem::path script_path = onnx_path.parent_path() / "make_rope_onnx.py";
    std::ofstream py(script_path);
    if (!py) { throw std::runtime_error("failed to write python script: " + script_path.string()); }
    py << py_script; py.close();
    const std::string cmd = "python3 " + quote_for_sh(script_path.string()) + " " +
                            std::to_string(rows) + " " + std::to_string(cols) + " " +
                            quote_for_sh(onnx_path.string());
    run_system_checked(cmd);
}

RopeZgNetworkBundle get_or_compile_rope_zg_network(
    const std::filesystem::path & work_root,
    int64_t rows,
    int64_t cols) {
    if (rows <= 0 || cols <= 0 || cols % 2 != 0) {
        throw std::runtime_error("get_or_compile_rope_zg_network: invalid dims");
    }

    preload_zg_cache(work_root);
    const auto root_abs = std::filesystem::weakly_canonical(work_root);
    // v3: inputs X[rows,cols] + THETA[rows,cols/2]; cos/sin computed on ZG330 via Horner polynomial
    const auto net_name = "rope_neox_v3_bf16_" + std::to_string(rows) + "x" + std::to_string(cols);
    const auto cache_key = make_root_net_key(root_abs, net_name);

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cache_key);
        if (it != g_cache.end()) {
            RopeZgNetworkBundle out;
            out.net_name      = it->second.net_name;
            out.network       = it->second.network;
            out.ram_cache_hit = true;
            out.compiled_now  = false;
            return out;
        }
    }

    const auto work_dir = root_abs / net_name;
    ensure_dir(work_dir);
    const std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        const auto it = g_cache.find(cache_key);
        if (it != g_cache.end()) {
            RopeZgNetworkBundle out;
            out.net_name      = it->second.net_name;
            out.network       = it->second.network;
            out.ram_cache_hit = true;
            out.compiled_now  = false;
            return out;
        }
    }

    icraft::xir::Network network;
    try {
        auto [jp, rp] = find_generated_zg_json_raw(work_dir, net_name);
        network = icraft::xir::Network::CreateFromJsonFile(jp.string());
        network.loadParamsFromFile(rp.string());
        CachedMatmulEntry e; e.net_name = net_name; e.network = network;
        e.m = rows; e.k = cols; e.n = 1; e.json_path = jp; e.raw_path = rp;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cache_key] = e;
    } catch (...) {
        const auto onnx_path = work_dir / (net_name + ".onnx");
        build_rope_onnx(onnx_path, rows, cols);
        // inputs: X[rows, cols], THETA[rows, cols/2]
        const auto art = write_icraft_compile_toml_for_zg_elementwise(
            work_dir, net_name, onnx_path, {{rows, cols}, {rows, cols / 2}});
        run_icraft_compile(art);
        auto [jp, rp] = find_generated_zg_json_raw(work_dir, net_name);
        network = icraft::xir::Network::CreateFromJsonFile(jp.string());
        network.loadParamsFromFile(rp.string());
        CachedMatmulEntry e; e.net_name = net_name; e.network = network;
        e.m = rows; e.k = cols; e.n = 1; e.json_path = jp; e.raw_path = rp;
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_cache[cache_key] = e;
    }

    RopeZgNetworkBundle out;
    out.net_name     = net_name;
    out.network      = std::move(network);
    out.ram_cache_hit = false;
    out.compiled_now  = true;
    return out;
}

} // namespace ggml::fmsh::netmake
