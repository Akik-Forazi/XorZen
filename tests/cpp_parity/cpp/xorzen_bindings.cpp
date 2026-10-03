// xorzen_bindings.cpp — pybind11 bindings for the REAL XorzenModel.
//
// Exposes the C++ XorzenModelImpl to Python as `xorzen_cpp.XorzenModel`.
// This is NOT a reimplementation — it wraps the actual production C++ model.
//
// API:
//   import xorzen_cpp
//   model = xorzen_cpp.XorzenModel(config_dict)
//   model.load_from_dir("path/to/checkpoint_dir")
//   logits = model.forward(input_ids)
//   tokens = model.generate(prompt, max_length=10, temperature=0.5)
// Use torch/python.h for automatic tensor interop between Python and C++
#include <torch/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "xorzen/model.h"
#include "xorzen/variants.h"

#include <fstream>
#include <sstream>
#include <filesystem>

namespace py = pybind11;
using namespace xorzen;

// Convert a Python dict to a ModelConfig
ModelConfig dict_to_config(const py::dict& d) {
    ModelConfig cfg;
    if (d.contains("vocab_size")) cfg.vocab_size = d["vocab_size"].cast<int64_t>();
    if (d.contains("hidden_size")) cfg.hidden_size = d["hidden_size"].cast<int64_t>();
    if (d.contains("num_layers")) cfg.num_layers = d["num_layers"].cast<int64_t>();
    if (d.contains("num_attention_heads")) cfg.num_attention_heads = d["num_attention_heads"].cast<int64_t>();
    if (d.contains("expert_count")) cfg.expert_count = d["expert_count"].cast<int64_t>();
    if (d.contains("top_k_experts")) cfg.top_k_experts = d["top_k_experts"].cast<int64_t>();
    if (d.contains("context_length")) cfg.context_length = d["context_length"].cast<int64_t>();
    if (d.contains("ssm_state_dim")) cfg.ssm_state_dim = d["ssm_state_dim"].cast<int64_t>();
    if (d.contains("ssm_kernel_size")) cfg.ssm_kernel_size = d["ssm_kernel_size"].cast<int64_t>();
    if (d.contains("pad_token_id")) cfg.pad_token_id = d["pad_token_id"].cast<int64_t>();
    if (d.contains("tie_word_embeddings")) cfg.tie_word_embeddings = d["tie_word_embeddings"].cast<bool>();
    if (d.contains("router_temperature")) cfg.router_temperature = d["router_temperature"].cast<float>();
    if (d.contains("eval_routing_noise")) cfg.eval_routing_noise = d["eval_routing_noise"].cast<float>();
    cfg.normalize();
    return cfg;
}

// Wrapper class that holds the XorzenModel and exposes it to Python
class PyXorzenModel {
public:
    XorzenModel model_{nullptr};
    bool test_mode_ = true;

    PyXorzenModel(const py::dict& config, bool test_mode = true)
        : test_mode_(test_mode) {
        auto cfg = dict_to_config(config);
        model_ = XorzenModel(cfg, test_mode);
    }

    PyXorzenModel(const ModelConfig& cfg, bool test_mode = true)
        : test_mode_(test_mode) {
        model_ = XorzenModel(cfg, test_mode);
    }

    // Load from per-tensor .bin directory
    py::dict load_from_dir(const std::string& dir, bool strict = false) {
        // Load tensor map from directory
        std::unordered_map<std::string, torch::Tensor> sd;
        std::ifstream f(dir + "/state_dict_manifest.txt");
        if (!f) throw std::runtime_error("no state_dict_manifest.txt in " + dir);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string tok; ss >> tok;
            if (tok != "tensor") continue;
            std::string name, dtype, file, sc;
            ss >> name >> dtype >> file >> sc;
            std::vector<int64_t> shape;
            std::stringstream s2(sc); std::string item;
            while (std::getline(s2, item, ',')) if (!item.empty()) shape.push_back(std::stoll(item));

            torch::ScalarType dt;
            if (dtype == "float32") dt = torch::kFloat32;
            else if (dtype == "int64") dt = torch::kInt64;
            else continue;
            auto t = torch::empty(shape, torch::TensorOptions().dtype(dt));
            std::ifstream tf(dir + "/" + file, std::ios::binary);
            tf.read(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
            sd[name] = t;
        }
        auto result = model_->load_from_tensor_map(sd, strict);
        py::dict r;
        r["matched"] = result.matched;
        r["missing"] = result.missing;
        r["unexpected"] = result.unexpected;
        r["shape_mismatches"] = result.shape_mismatches;
        return r;
    }

    void save_to_dir(const std::string& dir) {
        model_->save_to_tensor_map(dir);
    }

    void eval() { model_->eval(); }
    void train() { model_->train(); }

    // Forward pass: input_ids [B, T] → logits [B, T, V]
    py::object forward(const torch::Tensor& input_ids,
                       const py::object& labels_obj = py::none()) {
        torch::Tensor labels;
        if (!labels_obj.is_none()) {
            labels = labels_obj.cast<torch::Tensor>();
        }
        auto out = model_->forward(input_ids, {}, {}, labels);
        py::dict result;
        result["logits"] = out.logits;
        result["loss"] = out.loss;
        result["lm_loss"] = out.lm_loss;
        result["routing_loss"] = out.routing_loss;
        result["load_balance_loss"] = out.load_balance_loss;
        return result;
    }

    // Generation
    torch::Tensor generate(const torch::Tensor& prompt, int64_t max_length,
                           double temperature, int64_t top_k, double top_p,
                           double repetition_penalty, int64_t eos_token_id,
                           bool do_sample, int64_t num_beams) {
        GenerationConfig gc;
        gc.max_length = max_length;
        gc.temperature = temperature;
        gc.top_k = top_k;
        gc.top_p = top_p;
        gc.repetition_penalty = repetition_penalty;
        gc.eos_token_id = eos_token_id;
        gc.do_sample = do_sample;
        gc.num_beams = num_beams;
        return model_->generate(prompt, gc);
    }

    int64_t param_count() { return model_->count_parameters(); }
};

PYBIND11_MODULE(xorzen_cpp, m) {
    m.doc() = "XorZen C++ backend — real production model via pybind11";

    py::class_<PyXorzenModel>(m, "XorzenModel")
        .def(py::init<const py::dict&, bool>(), py::arg("config"), py::arg("test_mode") = true)
        .def("load_from_dir", &PyXorzenModel::load_from_dir,
             py::arg("dir"), py::arg("strict") = false,
             "Load checkpoint from per-tensor .bin directory")
        .def("save_to_dir", &PyXorzenModel::save_to_dir,
             py::arg("dir"),
             "Save checkpoint to per-tensor .bin directory")
        .def("eval", &PyXorzenModel::eval, "Set eval mode")
        .def("train", &PyXorzenModel::train, "Set train mode")
        .def("forward", &PyXorzenModel::forward,
             py::arg("input_ids"), py::arg("labels") = py::none(),
             "Forward pass: [B,T] input_ids → logits [B,T,V]")
        .def("generate", &PyXorzenModel::generate,
             py::arg("prompt"), py::arg("max_length") = 10,
             py::arg("temperature") = 1.0, py::arg("top_k") = 0,
             py::arg("top_p") = 1.0, py::arg("repetition_penalty") = 1.0,
             py::arg("eos_token_id") = -1, py::arg("do_sample") = false,
             py::arg("num_beams") = 1,
             "Generate tokens from prompt")
        .def_property_readonly("param_count", &PyXorzenModel::param_count,
                               "Total parameter count");

    m.attr("__version__") = "0.2.5";

    // Factory: create from preset name (e.g. "tiny_23k")
    m.def("from_preset", [](const std::string& name, bool test_mode) -> PyXorzenModel* {
        ModelConfig cfg;
        if (name == "tiny_23k") cfg = ConfigFactory::get_config(ModelSize::TINY_23K);
        else if (name == "nano_1m") cfg = ConfigFactory::get_config(ModelSize::NANO_1M);
        else throw std::runtime_error("Unknown preset: " + name);
        return new PyXorzenModel(cfg, test_mode);
    }, py::arg("name"), py::arg("test_mode") = true,
       "Create model from preset name (tiny_23k, nano_1m)");
}
