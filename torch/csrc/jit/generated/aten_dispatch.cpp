#include "Python.h"
#include "aten_dispatch.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/utils/functional.h"

#include <unordered_map>
#include <cstring>
#include <tuple>

// generated from tools/autograd/templates/aten_dispatch.cpp

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::Tensor;
using at::IntList;
using at::TensorList;
using operator_constructor = std::function<TensorOp(jit::Node*)>;

namespace {

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template<typename T>
void pack(Stack & stack, T&& v) {
  stack.push_back(as_tensor(std::move(v)));
}
template<>
void pack(Stack & stack, Tensor&& v) {
  stack.push_back(std::move(v));
}
template<>
void pack(Stack & stack, std::vector<Tensor>&& ts) {
  for(auto& t : ts) {
    stack.push_back(std::move(t));
  }
}

template<std::size_t remaining, typename... Args>
struct TuplePacker
{
  // NB: *Not* a universal reference.
  static void execute(Stack & stack, std::tuple<Args...> && t)
  {
    // NB: The move here does not "destroy" the entire tuple, that is
    // not what std::move does; only the particular tuple index
    // processed here gets stolen.
    pack(stack, std::get<sizeof...(Args) - remaining>(std::move(t)));
    TuplePacker<remaining - 1, Args...>::execute(stack, std::move(t));
  }
};

template<typename... Args>
struct TuplePacker<0, Args...>
{
  static void execute(Stack & stack, std::tuple<Args...> && t) {};
};

template<typename... Args>
void pack(Stack & stack, std::tuple<Args...> && t) {
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

int deviceForInputs(Stack & stack, size_t N) {
  if(N == 0)
    return -1;
  auto & t = *(stack.end() - N);
  return t.type().is_cuda() ? (int) t.get_device() : -1;
}

// A list of functions taking TensorList arguments (where we can't use
// the number of inputs to choose an overload).
std::unordered_set<Symbol> tensor_vararg_fns = {
  kcat,
};

template<size_t N>
std::array<bool, N> as_bool_array(const std::vector<int64_t>& vec) {
  std::array<bool, N> res;
  JIT_ASSERT(vec.size() == N);
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}

std::unordered_map<std::string, operator_constructor> constructors = {
  {"RoiPooling2d_backward-4-pooledHeight-pooledWidth-spatialScale", [](Node *node) {
    auto pooledHeight = int64_t(node->i(Symbol("pooledHeight")));
    auto pooledWidth = int64_t(node->i(Symbol("pooledWidth")));
    auto spatialScale = double(node->f(Symbol("spatialScale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
  
      
      auto result = at::RoiPooling2d_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), pooledHeight, pooledWidth, spatialScale, std::move(fromLast(stack, -1)), std::move(fromLast(stack, -2)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_backward", 4);
  }},
  {"RoiPooling2d_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto pooledHeight = tensor_as<int64_t>(std::move(fromLast(stack, 4)));
      auto pooledWidth = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto spatialScale = tensor_as<double>(std::move(fromLast(stack, 2)));
      
      auto result = at::RoiPooling2d_backward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), pooledHeight, pooledWidth, spatialScale, std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_backward", 7);
  }},
  {"RoiPooling2d_forward-2-pooledHeight-pooledWidth-spatialScale", [](Node *node) {
    auto pooledHeight = int64_t(node->i(Symbol("pooledHeight")));
    auto pooledWidth = int64_t(node->i(Symbol("pooledWidth")));
    auto spatialScale = double(node->f(Symbol("spatialScale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::RoiPooling2d_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), pooledHeight, pooledWidth, spatialScale);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_forward", 2);
  }},
  {"RoiPooling2d_forward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto pooledHeight = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto pooledWidth = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto spatialScale = tensor_as<double>(std::move(fromLast(stack, 0)));
      
      auto result = at::RoiPooling2d_forward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), pooledHeight, pooledWidth, spatialScale);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_forward", 5);
  }},
  {"_addmv-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addmv");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::_addmv(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_addmv", 3);
  }},
  {"_addmv-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addmv");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::_addmv(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "_addmv", 5);
  }},
  {"_addr-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addr");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::_addr(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_addr", 3);
  }},
  {"_addr-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addr");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::_addr(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "_addr", 5);
  }},
  {"_cat-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 1));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      drop(stack, 1);
      auto result = at::_cat(last(stack, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cat", varargs_length);
  }},
  {"_cat-*-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 0));
  
      
      auto result = at::_cat(last(stack, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cat", varargs_length);
  }},
  {"_convolution-12", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 12 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 8)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 7)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto transposed = tensor_as<bool>(std::move(fromLast(stack, 5)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto cudnn_enabled = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::_convolution(std::move(fromLast(stack, 12)), std::move(fromLast(stack, 11)), std::move(fromLast(stack, 10)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
      drop(stack, 12);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution", 12);
  }},
  {"_convolution-3-benchmark-cudnn_enabled-deterministic-dilation-groups-output_padding-padding-stride-transposed", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto transposed = bool(node->i(Symbol("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    auto cudnn_enabled = bool(node->i(Symbol("cudnn_enabled")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::_convolution(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution", 3);
  }},
  {"_convolution_double_backward-16", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_double_backward");
      AutoGPU device_guard(deviceForInputs(stack, 16 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 9)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 8)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 7)));
      auto transposed = tensor_as<bool>(std::move(fromLast(stack, 6)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 4)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 3)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto cudnn_enabled = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::_convolution_double_backward(std::move(fromLast(stack, 16)), std::move(fromLast(stack, 15)), std::move(fromLast(stack, 14)), std::move(fromLast(stack, 13)), std::move(fromLast(stack, 12)), std::move(fromLast(stack, 11)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
      drop(stack, 16);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_double_backward", 16);
  }},
  {"_convolution_double_backward-6-benchmark-cudnn_enabled-deterministic-dilation-groups-output_mask-output_padding-padding-stride-transposed", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto transposed = bool(node->i(Symbol("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    auto cudnn_enabled = bool(node->i(Symbol("cudnn_enabled")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_double_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
  
      
      auto result = at::_convolution_double_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_double_backward", 6);
  }},
  {"_convolution_nogroup-3-dilation-output_padding-padding-stride-transposed", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto transposed = bool(node->i(Symbol("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_nogroup");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::_convolution_nogroup(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, transposed, output_padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_nogroup", 3);
  }},
  {"_convolution_nogroup-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_nogroup");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto transposed = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::_convolution_nogroup(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), stride, padding, dilation, transposed, output_padding);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_nogroup", 8);
  }},
  {"_cudnn_rnn_flatten_weight-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cudnn_rnn_flatten_weight");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 7));
      auto weight_stride0 = tensor_as<int64_t>(std::move(fromLast(stack, 6)));
      auto input_size = tensor_as<int64_t>(std::move(fromLast(stack, 5)));
      auto mode = tensor_as<int64_t>(std::move(fromLast(stack, 4)));
      auto hidden_size = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto num_layers = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto batch_first = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto bidirectional = tensor_as<bool>(std::move(fromLast(stack, 0)));
      drop(stack, 7);
      auto result = at::_cudnn_rnn_flatten_weight(last(stack, varargs_length), weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cudnn_rnn_flatten_weight", varargs_length);
  }},
  {"_cudnn_rnn_flatten_weight-*-batch_first-bidirectional-hidden_size-input_size-mode-num_layers-weight_stride0", [](Node *node) {
    auto weight_stride0 = int64_t(node->i(Symbol("weight_stride0")));
    auto input_size = int64_t(node->i(Symbol("input_size")));
    auto mode = int64_t(node->i(Symbol("mode")));
    auto hidden_size = int64_t(node->i(Symbol("hidden_size")));
    auto num_layers = int64_t(node->i(Symbol("num_layers")));
    auto batch_first = bool(node->i(Symbol("batch_first")));
    auto bidirectional = bool(node->i(Symbol("bidirectional")));
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cudnn_rnn_flatten_weight");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 0));
  
      
      auto result = at::_cudnn_rnn_flatten_weight(last(stack, varargs_length), weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cudnn_rnn_flatten_weight", varargs_length);
  }},
  {"_det_with_svd-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_det_with_svd");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::_det_with_svd(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_det_with_svd", 1);
  }},
  {"_dimI-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dimI");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1)))._dimI();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_dimI", 1);
  }},
  {"_dimV-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dimV");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1)))._dimV();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_dimV", 1);
  }},
  {"_dirichlet_grad-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dirichlet_grad");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::_dirichlet_grad(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_dirichlet_grad", 3);
  }},
  {"_dot-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dot");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_dot(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_dot", 2);
  }},
  {"_ger-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_ger");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_ger(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_ger", 2);
  }},
  {"_indices-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_indices");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1)))._indices();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_indices", 1);
  }},
  {"_mm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_mm");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_mm(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_mm", 2);
  }},
  {"_mv-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_mv");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_mv(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_mv", 2);
  }},
  {"_nnz-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_nnz");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1)))._nnz();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_nnz", 1);
  }},
  {"_s_where-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_s_where");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::_s_where(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_s_where", 3);
  }},
  {"_sigmoid-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sigmoid");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::_sigmoid(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sigmoid", 1);
  }},
  {"_sigmoid_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sigmoid_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_sigmoid_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_sigmoid_backward", 2);
  }},
  {"_sigmoid_forward-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sigmoid_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::_sigmoid_forward(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sigmoid_forward", 1);
  }},
  {"_standard_gamma_grad-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_standard_gamma_grad");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_standard_gamma_grad(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_standard_gamma_grad", 2);
  }},
  {"_tanh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tanh");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::_tanh(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_tanh", 1);
  }},
  {"_tanh_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tanh_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::_tanh_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_tanh_backward", 2);
  }},
  {"_tanh_forward-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tanh_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::_tanh_forward(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_tanh_forward", 1);
  }},
  {"_unsafe_view-1-size", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_unsafe_view");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::_unsafe_view(std::move(fromLast(stack, 1)), size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_unsafe_view", 1);
  }},
  {"_unsafe_view-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_unsafe_view");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::_unsafe_view(std::move(fromLast(stack, 2)), size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_unsafe_view", 2);
  }},
  {"_values-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_values");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1)))._values();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_values", 1);
  }},
  {"abs-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("abs");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::abs(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "abs", 1);
  }},
  {"acos-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("acos");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::acos(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "acos", 1);
  }},
  {"adaptive_avg_pool1d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_avg_pool1d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool1d", 1);
  }},
  {"adaptive_avg_pool1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_avg_pool1d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool1d", 2);
  }},
  {"adaptive_avg_pool2d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_avg_pool2d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d", 1);
  }},
  {"adaptive_avg_pool2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_avg_pool2d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d", 2);
  }},
  {"adaptive_avg_pool2d_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::adaptive_avg_pool2d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d_backward", 2);
  }},
  {"adaptive_avg_pool2d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_avg_pool2d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d_forward", 1);
  }},
  {"adaptive_avg_pool2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_avg_pool2d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d_forward", 2);
  }},
  {"adaptive_avg_pool3d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_avg_pool3d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d", 1);
  }},
  {"adaptive_avg_pool3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_avg_pool3d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d", 2);
  }},
  {"adaptive_avg_pool3d_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::adaptive_avg_pool3d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d_backward", 2);
  }},
  {"adaptive_avg_pool3d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_avg_pool3d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d_forward", 1);
  }},
  {"adaptive_avg_pool3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_avg_pool3d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d_forward", 2);
  }},
  {"adaptive_max_pool1d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_max_pool1d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool1d", 1);
  }},
  {"adaptive_max_pool1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_max_pool1d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool1d", 2);
  }},
  {"adaptive_max_pool2d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_max_pool2d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d", 1);
  }},
  {"adaptive_max_pool2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_max_pool2d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d", 2);
  }},
  {"adaptive_max_pool2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::adaptive_max_pool2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d_backward", 3);
  }},
  {"adaptive_max_pool2d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_max_pool2d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d_forward", 1);
  }},
  {"adaptive_max_pool2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_max_pool2d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d_forward", 2);
  }},
  {"adaptive_max_pool3d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_max_pool3d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d", 1);
  }},
  {"adaptive_max_pool3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_max_pool3d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d", 2);
  }},
  {"adaptive_max_pool3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::adaptive_max_pool3d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d_backward", 3);
  }},
  {"adaptive_max_pool3d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::adaptive_max_pool3d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d_forward", 1);
  }},
  {"adaptive_max_pool3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::adaptive_max_pool3d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d_forward", 2);
  }},
  {"add-1-alpha-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("add");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::add(std::move(fromLast(stack, 1)), other, alpha);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "add", 1);
  }},
  {"add-2-alpha", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("add");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::add(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), alpha);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "add", 2);
  }},
  {"add-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("add");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::add(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "add", 3);
  }},
  {"addbmm-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addbmm");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::addbmm(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addbmm", 3);
  }},
  {"addbmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addbmm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::addbmm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addbmm", 5);
  }},
  {"addcdiv-3-value", [](Node *node) {
    auto value = Scalar(node->t(Symbol("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcdiv");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::addcdiv(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addcdiv", 3);
  }},
  {"addcdiv-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcdiv");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto value = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::addcdiv(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), value);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "addcdiv", 4);
  }},
  {"addcmul-3-value", [](Node *node) {
    auto value = Scalar(node->t(Symbol("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcmul");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::addcmul(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addcmul", 3);
  }},
  {"addcmul-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcmul");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto value = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::addcmul(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), value);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "addcmul", 4);
  }},
  {"addmm-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmm");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::addmm(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addmm", 3);
  }},
  {"addmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::addmm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addmm", 5);
  }},
  {"addmv-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmv");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::addmv(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addmv", 3);
  }},
  {"addmv-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmv");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::addmv(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addmv", 5);
  }},
  {"addr-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addr");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::addr(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addr", 3);
  }},
  {"addr-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addr");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::addr(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addr", 5);
  }},
  {"alias-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("alias");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::alias(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "alias", 1);
  }},
  {"all-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("all");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).all();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "all", 1);
  }},
  {"allclose-2-atol-rtol", [](Node *node) {
    auto rtol = double(node->f(Symbol("rtol")));
    auto atol = double(node->f(Symbol("atol")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("allclose");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::allclose(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), rtol, atol);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "allclose", 2);
  }},
  {"allclose-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("allclose");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto rtol = tensor_as<double>(std::move(fromLast(stack, 1)));
      auto atol = tensor_as<double>(std::move(fromLast(stack, 0)));
      
      auto result = at::allclose(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), rtol, atol);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "allclose", 4);
  }},
  {"any-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("any");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).any();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "any", 1);
  }},
  {"as_strided-1-size-storage_offset-stride", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol("size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto storage_offset = int64_t(node->i(Symbol("storage_offset")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("as_strided");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::as_strided(std::move(fromLast(stack, 1)), size, stride, storage_offset);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "as_strided", 1);
  }},
  {"as_strided-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("as_strided");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto storage_offset = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::as_strided(std::move(fromLast(stack, 4)), size, stride, storage_offset);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "as_strided", 4);
  }},
  {"asin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("asin");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::asin(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "asin", 1);
  }},
  {"atan-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("atan");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::atan(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "atan", 1);
  }},
  {"atan2-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("atan2");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::atan2(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "atan2", 2);
  }},
  {"avg_pool2d-1-ceil_mode-count_include_pad-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::avg_pool2d(std::move(fromLast(stack, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d", 1);
  }},
  {"avg_pool2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto count_include_pad = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::avg_pool2d(std::move(fromLast(stack, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d", 6);
  }},
  {"avg_pool2d_backward-2-ceil_mode-count_include_pad-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::avg_pool2d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_backward", 2);
  }},
  {"avg_pool2d_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto count_include_pad = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::avg_pool2d_backward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_backward", 7);
  }},
  {"avg_pool2d_forward-1-ceil_mode-count_include_pad-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::avg_pool2d_forward(std::move(fromLast(stack, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_forward", 1);
  }},
  {"avg_pool2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto count_include_pad = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::avg_pool2d_forward(std::move(fromLast(stack, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_forward", 6);
  }},
  {"avg_pool3d-1-ceil_mode-count_include_pad-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::avg_pool3d(std::move(fromLast(stack, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d", 1);
  }},
  {"avg_pool3d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto count_include_pad = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::avg_pool3d(std::move(fromLast(stack, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d", 6);
  }},
  {"avg_pool3d_backward-2-ceil_mode-count_include_pad-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::avg_pool3d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_backward", 2);
  }},
  {"avg_pool3d_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto count_include_pad = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::avg_pool3d_backward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_backward", 7);
  }},
  {"avg_pool3d_forward-1-ceil_mode-count_include_pad-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::avg_pool3d_forward(std::move(fromLast(stack, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_forward", 1);
  }},
  {"avg_pool3d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto count_include_pad = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::avg_pool3d_forward(std::move(fromLast(stack, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_forward", 6);
  }},
  {"baddbmm-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("baddbmm");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::baddbmm(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "baddbmm", 3);
  }},
  {"baddbmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("baddbmm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::baddbmm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "baddbmm", 5);
  }},
  {"batch_norm-5-cudnn_enabled-eps-momentum-training", [](Node *node) {
    auto training = bool(node->i(Symbol("training")));
    auto momentum = double(node->f(Symbol("momentum")));
    auto eps = double(node->f(Symbol("eps")));
    auto cudnn_enabled = bool(node->i(Symbol("cudnn_enabled")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::batch_norm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), training, momentum, eps, cudnn_enabled);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "batch_norm", 5);
  }},
  {"batch_norm-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto training = tensor_as<bool>(std::move(fromLast(stack, 3)));
      auto momentum = tensor_as<double>(std::move(fromLast(stack, 2)));
      auto eps = tensor_as<double>(std::move(fromLast(stack, 1)));
      auto cudnn_enabled = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::batch_norm(std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), training, momentum, eps, cudnn_enabled);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "batch_norm", 9);
  }},
  {"binary_cross_entropy-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::binary_cross_entropy(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy", 3);
  }},
  {"binary_cross_entropy-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::binary_cross_entropy(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy", 5);
  }},
  {"binary_cross_entropy_backward-4-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
  
      
      auto result = at::binary_cross_entropy_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_backward", 4);
  }},
  {"binary_cross_entropy_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::binary_cross_entropy_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_backward", 6);
  }},
  {"binary_cross_entropy_forward-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::binary_cross_entropy_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_forward", 3);
  }},
  {"binary_cross_entropy_forward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::binary_cross_entropy_forward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_forward", 5);
  }},
  {"bmm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("bmm");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::bmm(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "bmm", 2);
  }},
  {"btrifact-1-pivot", [](Node *node) {
    auto pivot = bool(node->i(Symbol("pivot")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::btrifact(std::move(fromLast(stack, 1)), pivot);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact", 1);
  }},
  {"btrifact-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto pivot = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::btrifact(std::move(fromLast(stack, 2)), pivot);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact", 2);
  }},
  {"btrifact_with_info-1-pivot", [](Node *node) {
    auto pivot = bool(node->i(Symbol("pivot")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact_with_info");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::btrifact_with_info(std::move(fromLast(stack, 1)), pivot);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact_with_info", 1);
  }},
  {"btrifact_with_info-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact_with_info");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto pivot = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::btrifact_with_info(std::move(fromLast(stack, 2)), pivot);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact_with_info", 2);
  }},
  {"btrisolve-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrisolve");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::btrisolve(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "btrisolve", 3);
  }},
  {"cat-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 1));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      drop(stack, 1);
      auto result = at::cat(last(stack, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "cat", varargs_length);
  }},
  {"cat-*-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 0));
  
      
      auto result = at::cat(last(stack, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "cat", varargs_length);
  }},
  {"ceil-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ceil");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::ceil(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ceil", 1);
  }},
  {"chunk-1-chunks-dim", [](Node *node) {
    auto chunks = int64_t(node->i(Symbol("chunks")));
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("chunk");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::chunk(std::move(fromLast(stack, 1)), chunks, dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "chunk", 1);
  }},
  {"chunk-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("chunk");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto chunks = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::chunk(std::move(fromLast(stack, 3)), chunks, dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "chunk", 3);
  }},
  {"clamp-1-max-min", [](Node *node) {
    auto min = Scalar(node->t(Symbol("min")));
    auto max = Scalar(node->t(Symbol("max")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::clamp(std::move(fromLast(stack, 1)), min, max);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clamp", 1);
  }},
  {"clamp-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto min = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto max = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::clamp(std::move(fromLast(stack, 3)), min, max);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "clamp", 3);
  }},
  {"clamp_max-1-max", [](Node *node) {
    auto max = Scalar(node->t(Symbol("max")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_max");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::clamp_max(std::move(fromLast(stack, 1)), max);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_max", 1);
  }},
  {"clamp_max-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_max");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto max = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::clamp_max(std::move(fromLast(stack, 2)), max);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_max", 2);
  }},
  {"clamp_min-1-min", [](Node *node) {
    auto min = Scalar(node->t(Symbol("min")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_min");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::clamp_min(std::move(fromLast(stack, 1)), min);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_min", 1);
  }},
  {"clamp_min-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_min");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto min = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::clamp_min(std::move(fromLast(stack, 2)), min);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_min", 2);
  }},
  {"clone-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clone");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).clone();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clone", 1);
  }},
  {"coalesce-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("coalesce");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).coalesce();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "coalesce", 1);
  }},
  {"contiguous-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("contiguous");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).contiguous();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "contiguous", 1);
  }},
  {"conv1d-3-dilation-groups-padding-stride", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv1d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv1d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv1d", 3);
  }},
  {"conv1d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv1d");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv1d(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), stride, padding, dilation, groups);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "conv1d", 7);
  }},
  {"conv2d-3-dilation-groups-padding-stride", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv2d", 3);
  }},
  {"conv2d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv2d(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), stride, padding, dilation, groups);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "conv2d", 7);
  }},
  {"conv3d-3-dilation-groups-padding-stride", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv3d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv3d", 3);
  }},
  {"conv3d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv3d(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), stride, padding, dilation, groups);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "conv3d", 7);
  }},
  {"conv_tbc-3-pad", [](Node *node) {
    auto pad = int64_t(node->i(Symbol("pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv_tbc(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), pad);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc", 3);
  }},
  {"conv_tbc-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto pad = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv_tbc(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), pad);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc", 4);
  }},
  {"conv_tbc_backward-4-pad", [](Node *node) {
    auto pad = int64_t(node->i(Symbol("pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
  
      
      auto result = at::conv_tbc_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), pad);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc_backward", 4);
  }},
  {"conv_tbc_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto pad = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv_tbc_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), pad);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc_backward", 5);
  }},
  {"conv_transpose1d-3-dilation-groups-output_padding-padding-stride", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose1d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv_transpose1d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, output_padding, groups, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose1d", 3);
  }},
  {"conv_transpose1d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose1d");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv_transpose1d(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), stride, padding, output_padding, groups, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose1d", 8);
  }},
  {"conv_transpose2d-3-dilation-groups-output_padding-padding-stride", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv_transpose2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, output_padding, groups, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose2d", 3);
  }},
  {"conv_transpose2d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv_transpose2d(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), stride, padding, output_padding, groups, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose2d", 8);
  }},
  {"conv_transpose3d-3-dilation-groups-output_padding-padding-stride", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::conv_transpose3d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, output_padding, groups, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose3d", 3);
  }},
  {"conv_transpose3d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::conv_transpose3d(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), stride, padding, output_padding, groups, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose3d", 8);
  }},
  {"convolution-3-dilation-groups-output_padding-padding-stride-transposed", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto transposed = bool(node->i(Symbol("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto groups = int64_t(node->i(Symbol("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::convolution(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), stride, padding, dilation, transposed, output_padding, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "convolution", 3);
  }},
  {"convolution-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("convolution");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto transposed = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::convolution(std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), stride, padding, dilation, transposed, output_padding, groups);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "convolution", 9);
  }},
  {"cos-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cos");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cos(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cos", 1);
  }},
  {"cosh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cosh");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cosh(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cosh", 1);
  }},
  {"cross-2-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cross");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::cross(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cross", 2);
  }},
  {"cross-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cross");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::cross(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cross", 3);
  }},
  {"cudnn_affine_grid_generator-1-C-H-N-W", [](Node *node) {
    auto N = int64_t(node->i(Symbol("N")));
    auto C = int64_t(node->i(Symbol("C")));
    auto H = int64_t(node->i(Symbol("H")));
    auto W = int64_t(node->i(Symbol("W")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cudnn_affine_grid_generator(std::move(fromLast(stack, 1)), N, C, H, W);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator", 1);
  }},
  {"cudnn_affine_grid_generator-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto N = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto C = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto H = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto W = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_affine_grid_generator(std::move(fromLast(stack, 5)), N, C, H, W);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator", 5);
  }},
  {"cudnn_affine_grid_generator_backward-1-C-H-N-W", [](Node *node) {
    auto N = int64_t(node->i(Symbol("N")));
    auto C = int64_t(node->i(Symbol("C")));
    auto H = int64_t(node->i(Symbol("H")));
    auto W = int64_t(node->i(Symbol("W")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cudnn_affine_grid_generator_backward(std::move(fromLast(stack, 1)), N, C, H, W);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator_backward", 1);
  }},
  {"cudnn_affine_grid_generator_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto N = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto C = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto H = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto W = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_affine_grid_generator_backward(std::move(fromLast(stack, 5)), N, C, H, W);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator_backward", 5);
  }},
  {"cudnn_batch_norm-5-epsilon-exponential_average_factor-training", [](Node *node) {
    auto training = bool(node->i(Symbol("training")));
    auto exponential_average_factor = double(node->f(Symbol("exponential_average_factor")));
    auto epsilon = double(node->f(Symbol("epsilon")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::cudnn_batch_norm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), training, exponential_average_factor, epsilon);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm", 5);
  }},
  {"cudnn_batch_norm-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto training = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto exponential_average_factor = tensor_as<double>(std::move(fromLast(stack, 1)));
      auto epsilon = tensor_as<double>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_batch_norm(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), training, exponential_average_factor, epsilon);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm", 8);
  }},
  {"cudnn_batch_norm_backward-7-epsilon", [](Node *node) {
    auto epsilon = double(node->f(Symbol("epsilon")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
  
      
      auto result = at::cudnn_batch_norm_backward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), epsilon);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm_backward", 7);
  }},
  {"cudnn_batch_norm_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto epsilon = tensor_as<double>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_batch_norm_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), epsilon);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm_backward", 8);
  }},
  {"cudnn_convolution-3-benchmark-deterministic-dilation-groups-padding-stride", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::cudnn_convolution(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution", 3);
  }},
  {"cudnn_convolution-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution(std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution", 9);
  }},
  {"cudnn_convolution_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_backward(std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward", 10);
  }},
  {"cudnn_convolution_backward-3-benchmark-deterministic-dilation-groups-output_mask-padding-stride", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::cudnn_convolution_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward", 3);
  }},
  {"cudnn_convolution_backward_bias-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_bias");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cudnn_convolution_backward_bias(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_bias", 1);
  }},
  {"cudnn_convolution_backward_input-2-benchmark-deterministic-dilation-groups-padding-self_size-stride", [](Node *node) {
    auto self_size = std::vector<int64_t>(node->is(Symbol("self_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::cudnn_convolution_backward_input(self_size, std::move(fromLast(stack, 1)), std::move(fromLast(stack, 0)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_input", 2);
  }},
  {"cudnn_convolution_backward_input-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto self_size = tensor_as<IntList>(std::move(fromLast(stack, 8)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_backward_input(self_size, std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_input", 9);
  }},
  {"cudnn_convolution_backward_weight-2-benchmark-deterministic-dilation-groups-padding-stride-weight_size", [](Node *node) {
    auto weight_size = std::vector<int64_t>(node->is(Symbol("weight_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::cudnn_convolution_backward_weight(weight_size, std::move(fromLast(stack, 1)), std::move(fromLast(stack, 0)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_weight", 2);
  }},
  {"cudnn_convolution_backward_weight-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto weight_size = tensor_as<IntList>(std::move(fromLast(stack, 8)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_backward_weight(weight_size, std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_weight", 9);
  }},
  {"cudnn_convolution_transpose-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose");
      AutoGPU device_guard(deviceForInputs(stack, 10 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_transpose(std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose", 10);
  }},
  {"cudnn_convolution_transpose-3-benchmark-deterministic-dilation-groups-output_padding-padding-stride", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::cudnn_convolution_transpose(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose", 3);
  }},
  {"cudnn_convolution_transpose_backward-11", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward");
      AutoGPU device_guard(deviceForInputs(stack, 11 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 7)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_transpose_backward(std::move(fromLast(stack, 11)), std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 11);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward", 11);
  }},
  {"cudnn_convolution_transpose_backward-3-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::cudnn_convolution_transpose_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward", 3);
  }},
  {"cudnn_convolution_transpose_backward_bias-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_bias");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cudnn_convolution_transpose_backward_bias(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_bias", 1);
  }},
  {"cudnn_convolution_transpose_backward_input-2-benchmark-deterministic-dilation-groups-padding-stride", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::cudnn_convolution_transpose_backward_input(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_input", 2);
  }},
  {"cudnn_convolution_transpose_backward_input-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_transpose_backward_input(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_input", 8);
  }},
  {"cudnn_convolution_transpose_backward_weight-2-benchmark-deterministic-dilation-groups-padding-stride-weight_size", [](Node *node) {
    auto weight_size = std::vector<int64_t>(node->is(Symbol("weight_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto groups = int64_t(node->i(Symbol("groups")));
    auto benchmark = bool(node->i(Symbol("benchmark")));
    auto deterministic = bool(node->i(Symbol("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::cudnn_convolution_transpose_backward_weight(weight_size, std::move(fromLast(stack, 1)), std::move(fromLast(stack, 0)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_weight", 2);
  }},
  {"cudnn_convolution_transpose_backward_weight-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto weight_size = tensor_as<IntList>(std::move(fromLast(stack, 8)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto groups = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto benchmark = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto deterministic = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::cudnn_convolution_transpose_backward_weight(weight_size, std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_weight", 9);
  }},
  {"cudnn_grid_sampler-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_grid_sampler");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::cudnn_grid_sampler(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_grid_sampler", 2);
  }},
  {"cudnn_grid_sampler_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_grid_sampler_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::cudnn_grid_sampler_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_grid_sampler_backward", 3);
  }},
  {"cudnn_is_acceptable-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_is_acceptable");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cudnn_is_acceptable(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_is_acceptable", 1);
  }},
  {"cumprod-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumprod");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cumprod(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cumprod", 1);
  }},
  {"cumprod-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumprod");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::cumprod(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cumprod", 2);
  }},
  {"cumsum-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumsum");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::cumsum(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cumsum", 1);
  }},
  {"cumsum-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumsum");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::cumsum(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cumsum", 2);
  }},
  {"data_ptr-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("data_ptr");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).data_ptr();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "data_ptr", 1);
  }},
  {"det-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("det");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::det(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "det", 1);
  }},
  {"diag-1-diagonal", [](Node *node) {
    auto diagonal = int64_t(node->i(Symbol("diagonal")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diag");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::diag(std::move(fromLast(stack, 1)), diagonal);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "diag", 1);
  }},
  {"diag-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diag");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto diagonal = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::diag(std::move(fromLast(stack, 2)), diagonal);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "diag", 2);
  }},
  {"digamma-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("digamma");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::digamma(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "digamma", 1);
  }},
  {"dim-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dim");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).dim();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "dim", 1);
  }},
  {"dist-2-p", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dist");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::dist(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), p);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "dist", 2);
  }},
  {"dist-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dist");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::dist(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), p);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "dist", 3);
  }},
  {"div-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("div");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::div(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "div", 1);
  }},
  {"div-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("div");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::div(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "div", 2);
  }},
  {"dot-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dot");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::dot(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "dot", 2);
  }},
  {"eig-1-eigenvectors", [](Node *node) {
    auto eigenvectors = bool(node->i(Symbol("eigenvectors")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eig");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::eig(std::move(fromLast(stack, 1)), eigenvectors);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "eig", 1);
  }},
  {"eig-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eig");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto eigenvectors = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::eig(std::move(fromLast(stack, 2)), eigenvectors);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "eig", 2);
  }},
  {"elu-1-alpha-scale", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol("alpha")));
    auto scale = Scalar(node->t(Symbol("scale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::elu(std::move(fromLast(stack, 1)), alpha, scale);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "elu", 1);
  }},
  {"elu-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto scale = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::elu(std::move(fromLast(stack, 3)), alpha, scale);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "elu", 3);
  }},
  {"elu_backward-2-alpha-scale", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol("alpha")));
    auto scale = Scalar(node->t(Symbol("scale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::elu_backward(std::move(fromLast(stack, 2)), alpha, scale, std::move(fromLast(stack, -1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "elu_backward", 2);
  }},
  {"elu_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto scale = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      
      auto result = at::elu_backward(std::move(fromLast(stack, 4)), alpha, scale, std::move(fromLast(stack, 1)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "elu_backward", 4);
  }},
  {"elu_forward-1-alpha-scale", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol("alpha")));
    auto scale = Scalar(node->t(Symbol("scale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::elu_forward(std::move(fromLast(stack, 1)), alpha, scale);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "elu_forward", 1);
  }},
  {"elu_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto scale = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::elu_forward(std::move(fromLast(stack, 3)), alpha, scale);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "elu_forward", 3);
  }},
  {"embedding-2-padding_idx-scale_grad_by_freq-sparse", [](Node *node) {
    auto padding_idx = int64_t(node->i(Symbol("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    auto sparse = bool(node->i(Symbol("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::embedding(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding", 2);
  }},
  {"embedding-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto padding_idx = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto sparse = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding", 5);
  }},
  {"embedding_backward-2-num_weights-padding_idx-scale_grad_by_freq-sparse", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol("num_weights")));
    auto padding_idx = int64_t(node->i(Symbol("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    auto sparse = bool(node->i(Symbol("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::embedding_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), num_weights, padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_backward", 2);
  }},
  {"embedding_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto num_weights = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto padding_idx = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto sparse = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), num_weights, padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_backward", 6);
  }},
  {"embedding_bag-3-mode-scale_grad_by_freq-sparse", [](Node *node) {
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol("mode")));
    auto sparse = bool(node->i(Symbol("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::embedding_bag(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), scale_grad_by_freq, mode, sparse);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag", 3);
  }},
  {"embedding_bag-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto mode = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto sparse = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_bag(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), scale_grad_by_freq, mode, sparse);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag", 6);
  }},
  {"embedding_bag_backward-5-mode-num_weights-scale_grad_by_freq-sparse", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol("num_weights")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol("mode")));
    auto sparse = bool(node->i(Symbol("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::embedding_bag_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), num_weights, scale_grad_by_freq, mode, sparse);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_backward", 5);
  }},
  {"embedding_bag_backward-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_backward");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto num_weights = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto mode = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto sparse = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_bag_backward(std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), num_weights, scale_grad_by_freq, mode, sparse);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_backward", 9);
  }},
  {"embedding_bag_dense_backward-5-mode-num_weights-scale_grad_by_freq", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol("num_weights")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol("mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::embedding_bag_dense_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_dense_backward", 5);
  }},
  {"embedding_bag_dense_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto num_weights = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto mode = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_bag_dense_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_dense_backward", 8);
  }},
  {"embedding_bag_sparse_backward-5-mode-num_weights-scale_grad_by_freq", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol("num_weights")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol("mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::embedding_bag_sparse_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_sparse_backward", 5);
  }},
  {"embedding_bag_sparse_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto num_weights = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto mode = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_bag_sparse_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_sparse_backward", 8);
  }},
  {"embedding_dense_backward-2-num_weights-padding_idx-scale_grad_by_freq", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol("num_weights")));
    auto padding_idx = int64_t(node->i(Symbol("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::embedding_dense_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_dense_backward", 2);
  }},
  {"embedding_dense_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto num_weights = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto padding_idx = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_dense_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_dense_backward", 5);
  }},
  {"embedding_sparse_backward-2-num_weights-padding_idx-scale_grad_by_freq", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol("num_weights")));
    auto padding_idx = int64_t(node->i(Symbol("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol("scale_grad_by_freq")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::embedding_sparse_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_sparse_backward", 2);
  }},
  {"embedding_sparse_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto num_weights = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto padding_idx = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::embedding_sparse_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_sparse_backward", 5);
  }},
  {"empty_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("empty_like");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::empty_like(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "empty_like", 1);
  }},
  {"eq-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eq");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::eq(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "eq", 1);
  }},
  {"eq-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eq");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::eq(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "eq", 2);
  }},
  {"equal-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("equal");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::equal(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "equal", 2);
  }},
  {"erf-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("erf");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::erf(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "erf", 1);
  }},
  {"erfinv-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("erfinv");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::erfinv(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "erfinv", 1);
  }},
  {"exp-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("exp");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::exp(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "exp", 1);
  }},
  {"expand-1-size", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expand");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).expand(size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "expand", 1);
  }},
  {"expand-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expand");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = (std::move(fromLast(stack, 2))).expand(size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "expand", 2);
  }},
  {"expand_as-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expand_as");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = (std::move(fromLast(stack, 2))).expand_as(std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "expand_as", 2);
  }},
  {"expm1-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expm1");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::expm1(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "expm1", 1);
  }},
  {"floor-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("floor");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::floor(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "floor", 1);
  }},
  {"fmod-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fmod");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::fmod(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "fmod", 1);
  }},
  {"fmod-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fmod");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::fmod(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "fmod", 2);
  }},
  {"frac-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("frac");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::frac(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "frac", 1);
  }},
  {"fractional_max_pool2d-2-kernel_size-output_size", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::fractional_max_pool2d(std::move(fromLast(stack, 2)), kernel_size, output_size, std::move(fromLast(stack, -1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d", 2);
  }},
  {"fractional_max_pool2d-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      
      auto result = at::fractional_max_pool2d(std::move(fromLast(stack, 4)), kernel_size, output_size, std::move(fromLast(stack, 1)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d", 4);
  }},
  {"fractional_max_pool2d_backward-3-kernel_size-output_size", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::fractional_max_pool2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, output_size, std::move(fromLast(stack, -1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_backward", 3);
  }},
  {"fractional_max_pool2d_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      
      auto result = at::fractional_max_pool2d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), kernel_size, output_size, std::move(fromLast(stack, 1)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_backward", 5);
  }},
  {"fractional_max_pool2d_forward-2-kernel_size-output_size", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::fractional_max_pool2d_forward(std::move(fromLast(stack, 2)), kernel_size, output_size, std::move(fromLast(stack, -1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_forward", 2);
  }},
  {"fractional_max_pool2d_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      
      auto result = at::fractional_max_pool2d_forward(std::move(fromLast(stack, 4)), kernel_size, output_size, std::move(fromLast(stack, 1)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_forward", 4);
  }},
  {"gather-2-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gather");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::gather(std::move(fromLast(stack, 2)), dim, std::move(fromLast(stack, 0)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gather", 2);
  }},
  {"gather-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gather");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      
      auto result = at::gather(std::move(fromLast(stack, 3)), dim, std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "gather", 3);
  }},
  {"ge-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ge");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::ge(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ge", 1);
  }},
  {"ge-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ge");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::ge(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "ge", 2);
  }},
  {"gels-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gels");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::gels(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gels", 2);
  }},
  {"geqrf-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("geqrf");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::geqrf(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "geqrf", 1);
  }},
  {"ger-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ger");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::ger(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "ger", 2);
  }},
  {"gesv-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gesv");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::gesv(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gesv", 2);
  }},
  {"get_device-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("get_device");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).get_device();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "get_device", 1);
  }},
  {"glu-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::glu(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "glu", 1);
  }},
  {"glu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::glu(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "glu", 2);
  }},
  {"glu_backward-2-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::glu_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "glu_backward", 2);
  }},
  {"glu_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::glu_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "glu_backward", 3);
  }},
  {"glu_forward-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::glu_forward(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "glu_forward", 1);
  }},
  {"glu_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::glu_forward(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "glu_forward", 2);
  }},
  {"gt-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gt");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::gt(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "gt", 1);
  }},
  {"gt-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gt");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::gt(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gt", 2);
  }},
  {"hardshrink-1-lambd", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::hardshrink(std::move(fromLast(stack, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink", 1);
  }},
  {"hardshrink-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto lambd = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::hardshrink(std::move(fromLast(stack, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink", 2);
  }},
  {"hardshrink_backward-2-lambd", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::hardshrink_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_backward", 2);
  }},
  {"hardshrink_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto lambd = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::hardshrink_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), lambd);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_backward", 3);
  }},
  {"hardshrink_forward-1-lambd", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::hardshrink_forward(std::move(fromLast(stack, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_forward", 1);
  }},
  {"hardshrink_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto lambd = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::hardshrink_forward(std::move(fromLast(stack, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_forward", 2);
  }},
  {"hardtanh-1-max_val-min_val", [](Node *node) {
    auto min_val = Scalar(node->t(Symbol("min_val")));
    auto max_val = Scalar(node->t(Symbol("max_val")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::hardtanh(std::move(fromLast(stack, 1)), min_val, max_val);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh", 1);
  }},
  {"hardtanh-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto min_val = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto max_val = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::hardtanh(std::move(fromLast(stack, 3)), min_val, max_val);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh", 3);
  }},
  {"hardtanh_backward-2-max_val-min_val", [](Node *node) {
    auto min_val = Scalar(node->t(Symbol("min_val")));
    auto max_val = Scalar(node->t(Symbol("max_val")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::hardtanh_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), min_val, max_val);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_backward", 2);
  }},
  {"hardtanh_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto min_val = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto max_val = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::hardtanh_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), min_val, max_val);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_backward", 4);
  }},
  {"hardtanh_forward-1-max_val-min_val", [](Node *node) {
    auto min_val = Scalar(node->t(Symbol("min_val")));
    auto max_val = Scalar(node->t(Symbol("max_val")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::hardtanh_forward(std::move(fromLast(stack, 1)), min_val, max_val);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_forward", 1);
  }},
  {"hardtanh_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto min_val = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto max_val = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::hardtanh_forward(std::move(fromLast(stack, 3)), min_val, max_val);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_forward", 3);
  }},
  {"hinge_embedding_loss-2-margin-reduce-size_average", [](Node *node) {
    auto margin = double(node->f(Symbol("margin")));
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hinge_embedding_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::hinge_embedding_loss(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), margin, size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hinge_embedding_loss", 2);
  }},
  {"hinge_embedding_loss-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hinge_embedding_loss");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto margin = tensor_as<double>(std::move(fromLast(stack, 2)));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::hinge_embedding_loss(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), margin, size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "hinge_embedding_loss", 5);
  }},
  {"histc-1-bins-max-min", [](Node *node) {
    auto bins = int64_t(node->i(Symbol("bins")));
    auto min = Scalar(node->t(Symbol("min")));
    auto max = Scalar(node->t(Symbol("max")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("histc");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::histc(std::move(fromLast(stack, 1)), bins, min, max);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "histc", 1);
  }},
  {"histc-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("histc");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto bins = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto min = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto max = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::histc(std::move(fromLast(stack, 4)), bins, min, max);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "histc", 4);
  }},
  {"hspmm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hspmm");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::hspmm(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hspmm", 2);
  }},
  {"index_select-2-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("index_select");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::index_select(std::move(fromLast(stack, 2)), dim, std::move(fromLast(stack, 0)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "index_select", 2);
  }},
  {"index_select-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("index_select");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      
      auto result = at::index_select(std::move(fromLast(stack, 3)), dim, std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "index_select", 3);
  }},
  {"inverse-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("inverse");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::inverse(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "inverse", 1);
  }},
  {"is_coalesced-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_coalesced");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).is_coalesced();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_coalesced", 1);
  }},
  {"is_contiguous-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_contiguous");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).is_contiguous();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_contiguous", 1);
  }},
  {"is_cuda-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_cuda");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::is_cuda(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_cuda", 1);
  }},
  {"is_distributed-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_distributed");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::is_distributed(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_distributed", 1);
  }},
  {"is_floating_point-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_floating_point");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::is_floating_point(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_floating_point", 1);
  }},
  {"is_nonzero-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_nonzero");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::is_nonzero(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_nonzero", 1);
  }},
  {"is_same_size-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_same_size");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::is_same_size(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "is_same_size", 2);
  }},
  {"is_set_to-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_set_to");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = (std::move(fromLast(stack, 2))).is_set_to(std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "is_set_to", 2);
  }},
  {"is_signed-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_signed");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::is_signed(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_signed", 1);
  }},
  {"is_sparse-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_sparse");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::is_sparse(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_sparse", 1);
  }},
  {"kl_div-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::kl_div(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div", 2);
  }},
  {"kl_div-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::kl_div(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div", 4);
  }},
  {"kl_div_backward-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::kl_div_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_backward", 3);
  }},
  {"kl_div_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::kl_div_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_backward", 5);
  }},
  {"kl_div_forward-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::kl_div_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_forward", 2);
  }},
  {"kl_div_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::kl_div_forward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_forward", 4);
  }},
  {"kthvalue-1-dim-k-keepdim", [](Node *node) {
    auto k = int64_t(node->i(Symbol("k")));
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kthvalue");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::kthvalue(std::move(fromLast(stack, 1)), k, dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "kthvalue", 1);
  }},
  {"kthvalue-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kthvalue");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto k = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::kthvalue(std::move(fromLast(stack, 4)), k, dim, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "kthvalue", 4);
  }},
  {"l1_loss-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::l1_loss(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss", 2);
  }},
  {"l1_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::l1_loss(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss", 4);
  }},
  {"l1_loss_backward-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::l1_loss_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_backward", 3);
  }},
  {"l1_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::l1_loss_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_backward", 5);
  }},
  {"l1_loss_forward-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::l1_loss_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_forward", 2);
  }},
  {"l1_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::l1_loss_forward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_forward", 4);
  }},
  {"le-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("le");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::le(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "le", 1);
  }},
  {"le-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("le");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::le(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "le", 2);
  }},
  {"leaky_relu-1-negative_slope", [](Node *node) {
    auto negative_slope = Scalar(node->t(Symbol("negative_slope")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::leaky_relu(std::move(fromLast(stack, 1)), negative_slope);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu", 1);
  }},
  {"leaky_relu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto negative_slope = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::leaky_relu(std::move(fromLast(stack, 2)), negative_slope);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu", 2);
  }},
  {"leaky_relu_backward-2-negative_slope", [](Node *node) {
    auto negative_slope = Scalar(node->t(Symbol("negative_slope")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::leaky_relu_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), negative_slope);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_backward", 2);
  }},
  {"leaky_relu_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto negative_slope = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::leaky_relu_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), negative_slope);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_backward", 3);
  }},
  {"leaky_relu_forward-1-negative_slope", [](Node *node) {
    auto negative_slope = Scalar(node->t(Symbol("negative_slope")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::leaky_relu_forward(std::move(fromLast(stack, 1)), negative_slope);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_forward", 1);
  }},
  {"leaky_relu_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto negative_slope = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::leaky_relu_forward(std::move(fromLast(stack, 2)), negative_slope);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_forward", 2);
  }},
  {"lerp-2-weight", [](Node *node) {
    auto weight = Scalar(node->t(Symbol("weight")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lerp");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::lerp(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), weight);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "lerp", 2);
  }},
  {"lerp-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lerp");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto weight = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::lerp(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), weight);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "lerp", 3);
  }},
  {"lgamma-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lgamma");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::lgamma(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "lgamma", 1);
  }},
  {"log-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::log(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log", 1);
  }},
  {"log1p-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log1p");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::log1p(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log1p", 1);
  }},
  {"log_sigmoid-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_sigmoid");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::log_sigmoid(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_sigmoid", 1);
  }},
  {"log_sigmoid_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_sigmoid_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::log_sigmoid_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "log_sigmoid_backward", 3);
  }},
  {"log_sigmoid_forward-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_sigmoid_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::log_sigmoid_forward(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_sigmoid_forward", 1);
  }},
  {"log_softmax-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::log_softmax(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax", 1);
  }},
  {"log_softmax-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::log_softmax(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax", 2);
  }},
  {"log_softmax_backward-3-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::log_softmax_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), dim, std::move(fromLast(stack, 0)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax_backward", 3);
  }},
  {"log_softmax_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      
      auto result = at::log_softmax_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), dim, std::move(fromLast(stack, 1)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax_backward", 4);
  }},
  {"log_softmax_forward-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::log_softmax_forward(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax_forward", 1);
  }},
  {"log_softmax_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::log_softmax_forward(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax_forward", 2);
  }},
  {"lt-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lt");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::lt(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "lt", 1);
  }},
  {"lt-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lt");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::lt(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "lt", 2);
  }},
  {"masked_select-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("masked_select");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::masked_select(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "masked_select", 2);
  }},
  {"matmul-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("matmul");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::matmul(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "matmul", 2);
  }},
  {"max-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max", 1);
  }},
  {"max-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max", 1);
  }},
  {"max-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::max(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max", 2);
  }},
  {"max-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::max(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max", 3);
  }},
  {"max_pool1d-1-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max_pool1d(std::move(fromLast(stack, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool1d", 1);
  }},
  {"max_pool1d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_pool1d(std::move(fromLast(stack, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool1d", 6);
  }},
  {"max_pool2d-1-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max_pool2d(std::move(fromLast(stack, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d", 1);
  }},
  {"max_pool2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_pool2d(std::move(fromLast(stack, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d", 6);
  }},
  {"max_pool2d_backward-3-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::max_pool2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, stride, padding, dilation, ceil_mode, std::move(fromLast(stack, -4)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_backward", 3);
  }},
  {"max_pool2d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      
      auto result = at::max_pool2d_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, stride, padding, dilation, ceil_mode, std::move(fromLast(stack, 1)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_backward", 8);
  }},
  {"max_pool2d_forward-1-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max_pool2d_forward(std::move(fromLast(stack, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_forward", 1);
  }},
  {"max_pool2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_pool2d_forward(std::move(fromLast(stack, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_forward", 6);
  }},
  {"max_pool3d-1-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max_pool3d(std::move(fromLast(stack, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d", 1);
  }},
  {"max_pool3d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_pool3d(std::move(fromLast(stack, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d", 6);
  }},
  {"max_pool3d_backward-3-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::max_pool3d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, stride, padding, dilation, ceil_mode, std::move(fromLast(stack, -4)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_backward", 3);
  }},
  {"max_pool3d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 1)));
      
      auto result = at::max_pool3d_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, stride, padding, dilation, ceil_mode, std::move(fromLast(stack, 1)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_backward", 8);
  }},
  {"max_pool3d_forward-1-ceil_mode-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto ceil_mode = bool(node->i(Symbol("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::max_pool3d_forward(std::move(fromLast(stack, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_forward", 1);
  }},
  {"max_pool3d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto ceil_mode = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_pool3d_forward(std::move(fromLast(stack, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_forward", 6);
  }},
  {"max_unpool2d-2-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::max_unpool2d(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d", 2);
  }},
  {"max_unpool2d-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_unpool2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d", 3);
  }},
  {"max_unpool2d_backward-3-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::max_unpool2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_backward", 3);
  }},
  {"max_unpool2d_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_unpool2d_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_size);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_backward", 4);
  }},
  {"max_unpool2d_forward-2-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::max_unpool2d_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_forward", 2);
  }},
  {"max_unpool2d_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_unpool2d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_forward", 3);
  }},
  {"max_unpool3d-2-output_size-padding-stride", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::max_unpool3d(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_size, stride, padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d", 2);
  }},
  {"max_unpool3d-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_unpool3d(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), output_size, stride, padding);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d", 5);
  }},
  {"max_unpool3d_backward-3-output_size-padding-stride", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::max_unpool3d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_size, stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_backward", 3);
  }},
  {"max_unpool3d_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_unpool3d_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), output_size, stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_backward", 6);
  }},
  {"max_unpool3d_forward-2-output_size-padding-stride", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::max_unpool3d_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_size, stride, padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_forward", 2);
  }},
  {"max_unpool3d_forward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::max_unpool3d_forward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), output_size, stride, padding);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_forward", 5);
  }},
  {"mean-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mean");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::mean(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mean", 1);
  }},
  {"mean-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mean");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::mean(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mean", 1);
  }},
  {"mean-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mean");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::mean(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mean", 3);
  }},
  {"median-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("median");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::median(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "median", 1);
  }},
  {"median-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("median");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::median(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "median", 1);
  }},
  {"median-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("median");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::median(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "median", 3);
  }},
  {"min-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::min(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "min", 1);
  }},
  {"min-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::min(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "min", 1);
  }},
  {"min-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::min(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "min", 2);
  }},
  {"min-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::min(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "min", 3);
  }},
  {"mm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mm");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::mm(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mm", 2);
  }},
  {"mode-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mode");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::mode(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mode", 1);
  }},
  {"mode-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mode");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::mode(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mode", 3);
  }},
  {"mse_loss-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::mse_loss(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss", 2);
  }},
  {"mse_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::mse_loss(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss", 4);
  }},
  {"mse_loss_backward-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::mse_loss_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_backward", 3);
  }},
  {"mse_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::mse_loss_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_backward", 5);
  }},
  {"mse_loss_forward-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::mse_loss_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_forward", 2);
  }},
  {"mse_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::mse_loss_forward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_forward", 4);
  }},
  {"mul-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mul");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::mul(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mul", 1);
  }},
  {"mul-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mul");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::mul(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mul", 2);
  }},
  {"multi_margin_loss-3-margin-p-size_average", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    auto margin = Scalar(node->t(Symbol("margin")));
    auto size_average = bool(node->i(Symbol("size_average")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::multi_margin_loss(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), p, margin, std::move(fromLast(stack, -1)), size_average);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss", 3);
  }},
  {"multi_margin_loss-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 3)));
      auto margin = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::multi_margin_loss(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), p, margin, std::move(fromLast(stack, 2)), size_average);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss", 6);
  }},
  {"multi_margin_loss_backward-3-margin-p-size_average", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    auto margin = Scalar(node->t(Symbol("margin")));
    auto size_average = bool(node->i(Symbol("size_average")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::multi_margin_loss_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), p, margin, std::move(fromLast(stack, -1)), size_average);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_backward", 3);
  }},
  {"multi_margin_loss_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 3)));
      auto margin = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::multi_margin_loss_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), p, margin, std::move(fromLast(stack, 2)), size_average);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_backward", 6);
  }},
  {"multi_margin_loss_forward-3-margin-p-size_average", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    auto margin = Scalar(node->t(Symbol("margin")));
    auto size_average = bool(node->i(Symbol("size_average")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::multi_margin_loss_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), p, margin, std::move(fromLast(stack, -1)), size_average);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_forward", 3);
  }},
  {"multi_margin_loss_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 3)));
      auto margin = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::multi_margin_loss_forward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), p, margin, std::move(fromLast(stack, 2)), size_average);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_forward", 6);
  }},
  {"multilabel_margin_loss-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::multilabel_margin_loss(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss", 2);
  }},
  {"multilabel_margin_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::multilabel_margin_loss(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss", 4);
  }},
  {"multilabel_margin_loss_backward-4-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
  
      
      auto result = at::multilabel_margin_loss_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), size_average, reduce, std::move(fromLast(stack, -1)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_backward", 4);
  }},
  {"multilabel_margin_loss_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 1)));
      
      auto result = at::multilabel_margin_loss_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), size_average, reduce, std::move(fromLast(stack, 1)));
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_backward", 6);
  }},
  {"multilabel_margin_loss_forward-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::multilabel_margin_loss_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_forward", 2);
  }},
  {"multilabel_margin_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::multilabel_margin_loss_forward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_forward", 4);
  }},
  {"mv-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mv");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::mv(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mv", 2);
  }},
  {"narrow-1-dim-length-start", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto start = int64_t(node->i(Symbol("start")));
    auto length = int64_t(node->i(Symbol("length")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("narrow");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::narrow(std::move(fromLast(stack, 1)), dim, start, length);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "narrow", 1);
  }},
  {"narrow-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("narrow");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto start = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto length = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::narrow(std::move(fromLast(stack, 4)), dim, start, length);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "narrow", 4);
  }},
  {"ne-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ne");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::ne(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ne", 1);
  }},
  {"ne-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ne");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::ne(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "ne", 2);
  }},
  {"neg-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("neg");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::neg(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "neg", 1);
  }},
  {"nll_loss-3-ignore_index-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto ignore_index = int64_t(node->i(Symbol("ignore_index")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nll_loss(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss", 3);
  }},
  {"nll_loss-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto ignore_index = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::nll_loss(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss", 6);
  }},
  {"nll_loss2d-3-ignore_index-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto ignore_index = int64_t(node->i(Symbol("ignore_index")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nll_loss2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d", 3);
  }},
  {"nll_loss2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto ignore_index = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::nll_loss2d(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d", 6);
  }},
  {"nll_loss2d_backward-5-ignore_index-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto ignore_index = int64_t(node->i(Symbol("ignore_index")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::nll_loss2d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), size_average, ignore_index, reduce, std::move(fromLast(stack, -2)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_backward", 5);
  }},
  {"nll_loss2d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 3)));
      auto ignore_index = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 1)));
      
      auto result = at::nll_loss2d_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), size_average, ignore_index, reduce, std::move(fromLast(stack, 1)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_backward", 8);
  }},
  {"nll_loss2d_forward-3-ignore_index-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto ignore_index = int64_t(node->i(Symbol("ignore_index")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nll_loss2d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_forward", 3);
  }},
  {"nll_loss2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto ignore_index = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::nll_loss2d_forward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_forward", 6);
  }},
  {"nll_loss_backward-5-ignore_index-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto ignore_index = int64_t(node->i(Symbol("ignore_index")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::nll_loss_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), size_average, ignore_index, reduce, std::move(fromLast(stack, -2)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_backward", 5);
  }},
  {"nll_loss_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 3)));
      auto ignore_index = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 1)));
      
      auto result = at::nll_loss_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), size_average, ignore_index, reduce, std::move(fromLast(stack, 1)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_backward", 8);
  }},
  {"nll_loss_forward-3-ignore_index-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto ignore_index = int64_t(node->i(Symbol("ignore_index")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nll_loss_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_forward", 3);
  }},
  {"nll_loss_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto ignore_index = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::nll_loss_forward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_forward", 6);
  }},
  {"nnpack_spatial_convolution-3-kH-kW-padH-padW", [](Node *node) {
    auto kW = int64_t(node->i(Symbol("kW")));
    auto kH = int64_t(node->i(Symbol("kH")));
    auto padW = int64_t(node->i(Symbol("padW")));
    auto padH = int64_t(node->i(Symbol("padH")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nnpack_spatial_convolution(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), kW, kH, padW, padH);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution", 3);
  }},
  {"nnpack_spatial_convolution-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kW = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto kH = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto padW = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto padH = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::nnpack_spatial_convolution(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), kW, kH, padW, padH);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution", 7);
  }},
  {"nnpack_spatial_convolution_backward-3-kH-kW-output_mask-padH-padW", [](Node *node) {
    auto kW = int64_t(node->i(Symbol("kW")));
    auto kH = int64_t(node->i(Symbol("kH")));
    auto padW = int64_t(node->i(Symbol("padW")));
    auto padH = int64_t(node->i(Symbol("padH")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nnpack_spatial_convolution_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), kW, kH, padW, padH, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution_backward", 3);
  }},
  {"nnpack_spatial_convolution_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kW = tensor_as<int64_t>(std::move(fromLast(stack, 4)));
      auto kH = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto padW = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto padH = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::nnpack_spatial_convolution_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kW, kH, padW, padH, output_mask);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution_backward", 8);
  }},
  {"nnpack_spatial_convolution_backward_input-3-kH-kW-padH-padW", [](Node *node) {
    auto kW = int64_t(node->i(Symbol("kW")));
    auto kH = int64_t(node->i(Symbol("kH")));
    auto padW = int64_t(node->i(Symbol("padW")));
    auto padH = int64_t(node->i(Symbol("padH")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::nnpack_spatial_convolution_backward_input(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), kW, kH, padW, padH);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution_backward_input", 3);
  }},
  {"nnpack_spatial_convolution_backward_input-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kW = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto kH = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto padW = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto padH = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::nnpack_spatial_convolution_backward_input(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), kW, kH, padW, padH);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution_backward_input", 7);
  }},
  {"nnpack_spatial_convolution_backward_weight-2-kH-kW-padH-padW-weight_size", [](Node *node) {
    auto weight_size = std::vector<int64_t>(node->is(Symbol("weight_size")));
    auto kW = int64_t(node->i(Symbol("kW")));
    auto kH = int64_t(node->i(Symbol("kH")));
    auto padW = int64_t(node->i(Symbol("padW")));
    auto padH = int64_t(node->i(Symbol("padH")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::nnpack_spatial_convolution_backward_weight(std::move(fromLast(stack, 2)), weight_size, std::move(fromLast(stack, 0)), kW, kH, padW, padH);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution_backward_weight", 2);
  }},
  {"nnpack_spatial_convolution_backward_weight-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nnpack_spatial_convolution_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto weight_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto kW = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto kH = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto padW = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto padH = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::nnpack_spatial_convolution_backward_weight(std::move(fromLast(stack, 7)), weight_size, std::move(fromLast(stack, 5)), kW, kH, padW, padH);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "nnpack_spatial_convolution_backward_weight", 7);
  }},
  {"nonzero-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nonzero");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::nonzero(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "nonzero", 1);
  }},
  {"norm-1-dim-keepdim-p", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::norm(std::move(fromLast(stack, 1)), p, dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 1);
  }},
  {"norm-1-p", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::norm(std::move(fromLast(stack, 1)), p);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 1);
  }},
  {"norm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::norm(std::move(fromLast(stack, 2)), p);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 2);
  }},
  {"norm-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::norm(std::move(fromLast(stack, 4)), p, dim, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 4);
  }},
  {"numel-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("numel");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::numel(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "numel", 1);
  }},
  {"ones_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ones_like");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::ones_like(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ones_like", 1);
  }},
  {"orgqr-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("orgqr");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::orgqr(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "orgqr", 2);
  }},
  {"ormqr-3-left-transpose", [](Node *node) {
    auto left = bool(node->i(Symbol("left")));
    auto transpose = bool(node->i(Symbol("transpose")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ormqr");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::ormqr(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), left, transpose);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "ormqr", 3);
  }},
  {"ormqr-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ormqr");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto left = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto transpose = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::ormqr(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), left, transpose);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "ormqr", 5);
  }},
  {"permute-1-dims", [](Node *node) {
    auto dims = std::vector<int64_t>(node->is(Symbol("dims")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("permute");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).permute(dims);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "permute", 1);
  }},
  {"permute-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("permute");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dims = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = (std::move(fromLast(stack, 2))).permute(dims);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "permute", 2);
  }},
  {"pin_memory-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pin_memory");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::pin_memory(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pin_memory", 1);
  }},
  {"polygamma-1-n", [](Node *node) {
    auto n = int64_t(node->i(Symbol("n")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("polygamma");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::polygamma(n, std::move(fromLast(stack, 0)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "polygamma", 1);
  }},
  {"polygamma-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("polygamma");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto n = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      
      auto result = at::polygamma(n, std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "polygamma", 2);
  }},
  {"potrf-1-upper", [](Node *node) {
    auto upper = bool(node->i(Symbol("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrf");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::potrf(std::move(fromLast(stack, 1)), upper);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "potrf", 1);
  }},
  {"potrf-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrf");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto upper = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::potrf(std::move(fromLast(stack, 2)), upper);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "potrf", 2);
  }},
  {"potri-1-upper", [](Node *node) {
    auto upper = bool(node->i(Symbol("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potri");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::potri(std::move(fromLast(stack, 1)), upper);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "potri", 1);
  }},
  {"potri-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potri");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto upper = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::potri(std::move(fromLast(stack, 2)), upper);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "potri", 2);
  }},
  {"potrs-2-upper", [](Node *node) {
    auto upper = bool(node->i(Symbol("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrs");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::potrs(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), upper);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "potrs", 2);
  }},
  {"potrs-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrs");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto upper = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::potrs(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), upper);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "potrs", 3);
  }},
  {"pow-1-base", [](Node *node) {
    auto base = Scalar(node->t(Symbol("base")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pow");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::pow(base, std::move(fromLast(stack, 0)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pow", 1);
  }},
  {"pow-1-exponent", [](Node *node) {
    auto exponent = Scalar(node->t(Symbol("exponent")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pow");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::pow(std::move(fromLast(stack, 1)), exponent);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pow", 1);
  }},
  {"pow-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pow");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::pow(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "pow", 2);
  }},
  {"prelu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::prelu(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "prelu", 2);
  }},
  {"prelu_backward-3-output_mask", [](Node *node) {
    auto output_mask = as_bool_array<2>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::prelu_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "prelu_backward", 3);
  }},
  {"prelu_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto output_mask = tensor_as<std::array<bool,2>>(std::move(fromLast(stack, 0)));
      
      auto result = at::prelu_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "prelu_backward", 4);
  }},
  {"prelu_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::prelu_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "prelu_forward", 2);
  }},
  {"prod-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prod");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::prod(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "prod", 1);
  }},
  {"prod-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prod");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::prod(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "prod", 1);
  }},
  {"prod-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prod");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::prod(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "prod", 3);
  }},
  {"pstrf-1-tol-upper", [](Node *node) {
    auto upper = bool(node->i(Symbol("upper")));
    auto tol = Scalar(node->t(Symbol("tol")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pstrf");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::pstrf(std::move(fromLast(stack, 1)), upper, tol);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pstrf", 1);
  }},
  {"pstrf-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pstrf");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto upper = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto tol = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::pstrf(std::move(fromLast(stack, 3)), upper, tol);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "pstrf", 3);
  }},
  {"qr-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("qr");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::qr(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "qr", 1);
  }},
  {"rand_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rand_like");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::rand_like(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "rand_like", 1);
  }},
  {"randn_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("randn_like");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::randn_like(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "randn_like", 1);
  }},
  {"reciprocal-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reciprocal");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::reciprocal(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reciprocal", 1);
  }},
  {"reflection_pad1d-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::reflection_pad1d(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d", 1);
  }},
  {"reflection_pad1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::reflection_pad1d(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d", 2);
  }},
  {"reflection_pad1d_backward-2-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::reflection_pad1d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_backward", 2);
  }},
  {"reflection_pad1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::reflection_pad1d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_backward", 3);
  }},
  {"reflection_pad1d_forward-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::reflection_pad1d_forward(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_forward", 1);
  }},
  {"reflection_pad1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::reflection_pad1d_forward(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_forward", 2);
  }},
  {"reflection_pad2d-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::reflection_pad2d(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d", 1);
  }},
  {"reflection_pad2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::reflection_pad2d(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d", 2);
  }},
  {"reflection_pad2d_backward-2-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::reflection_pad2d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_backward", 2);
  }},
  {"reflection_pad2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::reflection_pad2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_backward", 3);
  }},
  {"reflection_pad2d_forward-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::reflection_pad2d_forward(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_forward", 1);
  }},
  {"reflection_pad2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::reflection_pad2d_forward(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_forward", 2);
  }},
  {"remainder-1-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("remainder");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::remainder(std::move(fromLast(stack, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "remainder", 1);
  }},
  {"remainder-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("remainder");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::remainder(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "remainder", 2);
  }},
  {"renorm-1-dim-maxnorm-p", [](Node *node) {
    auto p = Scalar(node->t(Symbol("p")));
    auto dim = int64_t(node->i(Symbol("dim")));
    auto maxnorm = Scalar(node->t(Symbol("maxnorm")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("renorm");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::renorm(std::move(fromLast(stack, 1)), p, dim, maxnorm);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "renorm", 1);
  }},
  {"renorm-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("renorm");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto p = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto maxnorm = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::renorm(std::move(fromLast(stack, 4)), p, dim, maxnorm);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "renorm", 4);
  }},
  {"repeat-1-repeats", [](Node *node) {
    auto repeats = std::vector<int64_t>(node->is(Symbol("repeats")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("repeat");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).repeat(repeats);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "repeat", 1);
  }},
  {"repeat-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("repeat");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto repeats = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = (std::move(fromLast(stack, 2))).repeat(repeats);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "repeat", 2);
  }},
  {"replication_pad1d-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::replication_pad1d(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d", 1);
  }},
  {"replication_pad1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad1d(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d", 2);
  }},
  {"replication_pad1d_backward-2-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::replication_pad1d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_backward", 2);
  }},
  {"replication_pad1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad1d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_backward", 3);
  }},
  {"replication_pad1d_forward-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::replication_pad1d_forward(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_forward", 1);
  }},
  {"replication_pad1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad1d_forward(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_forward", 2);
  }},
  {"replication_pad2d-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::replication_pad2d(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d", 1);
  }},
  {"replication_pad2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad2d(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d", 2);
  }},
  {"replication_pad2d_backward-2-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::replication_pad2d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_backward", 2);
  }},
  {"replication_pad2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_backward", 3);
  }},
  {"replication_pad2d_forward-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::replication_pad2d_forward(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_forward", 1);
  }},
  {"replication_pad2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad2d_forward(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_forward", 2);
  }},
  {"replication_pad3d-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::replication_pad3d(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d", 1);
  }},
  {"replication_pad3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad3d(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d", 2);
  }},
  {"replication_pad3d_backward-2-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::replication_pad3d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_backward", 2);
  }},
  {"replication_pad3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad3d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_backward", 3);
  }},
  {"replication_pad3d_forward-1-padding", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::replication_pad3d_forward(std::move(fromLast(stack, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_forward", 1);
  }},
  {"replication_pad3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::replication_pad3d_forward(std::move(fromLast(stack, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_forward", 2);
  }},
  {"round-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("round");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::round(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "round", 1);
  }},
  {"rrelu_with_noise_backward-3-lower-training-upper", [](Node *node) {
    auto lower = Scalar(node->t(Symbol("lower")));
    auto upper = Scalar(node->t(Symbol("upper")));
    auto training = bool(node->i(Symbol("training")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rrelu_with_noise_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::rrelu_with_noise_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), lower, upper, training);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "rrelu_with_noise_backward", 3);
  }},
  {"rrelu_with_noise_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rrelu_with_noise_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto lower = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto upper = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto training = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::rrelu_with_noise_backward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), lower, upper, training);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "rrelu_with_noise_backward", 6);
  }},
  {"rsqrt-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rsqrt");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::rsqrt(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "rsqrt", 1);
  }},
  {"select-1-dim-index", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto index = int64_t(node->i(Symbol("index")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("select");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::select(std::move(fromLast(stack, 1)), dim, index);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "select", 1);
  }},
  {"select-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("select");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto index = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::select(std::move(fromLast(stack, 3)), dim, index);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "select", 3);
  }},
  {"selu-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("selu");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::selu(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "selu", 1);
  }},
  {"sigmoid-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sigmoid");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sigmoid(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sigmoid", 1);
  }},
  {"sign-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sign");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sign(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sign", 1);
  }},
  {"sin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sin");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sin(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sin", 1);
  }},
  {"sinh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sinh");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sinh(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sinh", 1);
  }},
  {"size-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("size");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::size(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "size", 1);
  }},
  {"size-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("size");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::size(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "size", 2);
  }},
  {"sizes-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sizes");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).sizes();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sizes", 1);
  }},
  {"slice-1-dim-end-start-step", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto start = int64_t(node->i(Symbol("start")));
    auto end = int64_t(node->i(Symbol("end")));
    auto step = int64_t(node->i(Symbol("step")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("slice");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::slice(std::move(fromLast(stack, 1)), dim, start, end, step);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "slice", 1);
  }},
  {"slice-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("slice");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto start = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto end = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto step = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::slice(std::move(fromLast(stack, 5)), dim, start, end, step);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "slice", 5);
  }},
  {"smm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smm");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::smm(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "smm", 2);
  }},
  {"smooth_l1_loss-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::smooth_l1_loss(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss", 2);
  }},
  {"smooth_l1_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::smooth_l1_loss(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss", 4);
  }},
  {"smooth_l1_loss_backward-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::smooth_l1_loss_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_backward", 3);
  }},
  {"smooth_l1_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::smooth_l1_loss_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_backward", 5);
  }},
  {"smooth_l1_loss_forward-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::smooth_l1_loss_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_forward", 2);
  }},
  {"smooth_l1_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::smooth_l1_loss_forward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_forward", 4);
  }},
  {"soft_margin_loss-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::soft_margin_loss(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss", 2);
  }},
  {"soft_margin_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::soft_margin_loss(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss", 4);
  }},
  {"soft_margin_loss_backward-3-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::soft_margin_loss_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_backward", 3);
  }},
  {"soft_margin_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::soft_margin_loss_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_backward", 5);
  }},
  {"soft_margin_loss_forward-2-reduce-size_average", [](Node *node) {
    auto size_average = bool(node->i(Symbol("size_average")));
    auto reduce = bool(node->i(Symbol("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::soft_margin_loss_forward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_forward", 2);
  }},
  {"soft_margin_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto size_average = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto reduce = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::soft_margin_loss_forward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_forward", 4);
  }},
  {"softmax-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::softmax(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softmax", 1);
  }},
  {"softmax-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::softmax(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softmax", 2);
  }},
  {"softmax_backward-3-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::softmax_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), dim, std::move(fromLast(stack, 0)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softmax_backward", 3);
  }},
  {"softmax_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      
      auto result = at::softmax_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), dim, std::move(fromLast(stack, 1)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "softmax_backward", 4);
  }},
  {"softmax_forward-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::softmax_forward(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softmax_forward", 1);
  }},
  {"softmax_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::softmax_forward(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softmax_forward", 2);
  }},
  {"softplus-1-beta-threshold", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto threshold = Scalar(node->t(Symbol("threshold")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::softplus(std::move(fromLast(stack, 1)), beta, threshold);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softplus", 1);
  }},
  {"softplus-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto threshold = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::softplus(std::move(fromLast(stack, 3)), beta, threshold);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softplus", 3);
  }},
  {"softplus_backward-3-beta-threshold", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto threshold = Scalar(node->t(Symbol("threshold")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::softplus_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), beta, threshold, std::move(fromLast(stack, -1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_backward", 3);
  }},
  {"softplus_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 2)));
      auto threshold = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      
      auto result = at::softplus_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), beta, threshold, std::move(fromLast(stack, 1)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_backward", 5);
  }},
  {"softplus_forward-1-beta-threshold", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto threshold = Scalar(node->t(Symbol("threshold")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::softplus_forward(std::move(fromLast(stack, 1)), beta, threshold);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_forward", 1);
  }},
  {"softplus_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto threshold = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::softplus_forward(std::move(fromLast(stack, 3)), beta, threshold);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_forward", 3);
  }},
  {"softshrink-1-lambd", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::softshrink(std::move(fromLast(stack, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink", 1);
  }},
  {"softshrink-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto lambd = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::softshrink(std::move(fromLast(stack, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink", 2);
  }},
  {"softshrink_backward-2-lambd", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::softshrink_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_backward", 2);
  }},
  {"softshrink_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto lambd = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::softshrink_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), lambd);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_backward", 3);
  }},
  {"softshrink_forward-1-lambd", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::softshrink_forward(std::move(fromLast(stack, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_forward", 1);
  }},
  {"softshrink_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto lambd = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::softshrink_forward(std::move(fromLast(stack, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_forward", 2);
  }},
  {"sort-1-descending-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto descending = bool(node->i(Symbol("descending")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sort");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sort(std::move(fromLast(stack, 1)), dim, descending);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sort", 1);
  }},
  {"sort-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sort");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto descending = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::sort(std::move(fromLast(stack, 3)), dim, descending);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sort", 3);
  }},
  {"sparse_coo_tensor-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sparse_coo_tensor");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::sparse_coo_tensor(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "sparse_coo_tensor", 2);
  }},
  {"sparse_coo_tensor-2-size", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sparse_coo_tensor");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::sparse_coo_tensor(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "sparse_coo_tensor", 2);
  }},
  {"sparse_coo_tensor-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sparse_coo_tensor");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::sparse_coo_tensor(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sparse_coo_tensor", 3);
  }},
  {"split-1-dim-split_size", [](Node *node) {
    auto split_size = int64_t(node->i(Symbol("split_size")));
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("split");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::split(std::move(fromLast(stack, 1)), split_size, dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "split", 1);
  }},
  {"split-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("split");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto split_size = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::split(std::move(fromLast(stack, 3)), split_size, dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "split", 3);
  }},
  {"sqrt-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sqrt");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sqrt(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sqrt", 1);
  }},
  {"squeeze-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("squeeze");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::squeeze(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "squeeze", 1);
  }},
  {"squeeze-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("squeeze");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::squeeze(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "squeeze", 1);
  }},
  {"squeeze-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("squeeze");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::squeeze(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "squeeze", 2);
  }},
  {"sspaddmm-3-alpha-beta", [](Node *node) {
    auto beta = Scalar(node->t(Symbol("beta")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sspaddmm");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::sspaddmm(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sspaddmm", 3);
  }},
  {"sspaddmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sspaddmm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto beta = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::sspaddmm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "sspaddmm", 5);
  }},
  {"stack-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stack");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 1));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      drop(stack, 1);
      auto result = at::stack(last(stack, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "stack", varargs_length);
  }},
  {"stack-*-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stack");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length + 0));
  
      
      auto result = at::stack(last(stack, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "stack", varargs_length);
  }},
  {"std-1-dim-keepdim-unbiased", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto unbiased = bool(node->i(Symbol("unbiased")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::std(std::move(fromLast(stack, 1)), dim, unbiased, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "std", 1);
  }},
  {"std-1-unbiased", [](Node *node) {
    auto unbiased = bool(node->i(Symbol("unbiased")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::std(std::move(fromLast(stack, 1)), unbiased);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "std", 1);
  }},
  {"std-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto unbiased = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::std(std::move(fromLast(stack, 2)), unbiased);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "std", 2);
  }},
  {"std-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto unbiased = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::std(std::move(fromLast(stack, 4)), dim, unbiased, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "std", 4);
  }},
  {"stft-2-fft_size-frame_length-hop-pad_end-return_onesided", [](Node *node) {
    auto frame_length = int64_t(node->i(Symbol("frame_length")));
    auto hop = int64_t(node->i(Symbol("hop")));
    auto fft_size = int64_t(node->i(Symbol("fft_size")));
    auto return_onesided = bool(node->i(Symbol("return_onesided")));
    auto pad_end = int64_t(node->i(Symbol("pad_end")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stft");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::stft(std::move(fromLast(stack, 2)), frame_length, hop, fft_size, return_onesided, std::move(fromLast(stack, -3)), pad_end);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "stft", 2);
  }},
  {"stft-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stft");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto frame_length = tensor_as<int64_t>(std::move(fromLast(stack, 5)));
      auto hop = tensor_as<int64_t>(std::move(fromLast(stack, 4)));
      auto fft_size = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto return_onesided = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto pad_end = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::stft(std::move(fromLast(stack, 7)), frame_length, hop, fft_size, return_onesided, std::move(fromLast(stack, 2)), pad_end);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "stft", 7);
  }},
  {"storage_offset-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("storage_offset");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).storage_offset();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "storage_offset", 1);
  }},
  {"stride-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stride");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::stride(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "stride", 1);
  }},
  {"stride-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stride");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::stride(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "stride", 2);
  }},
  {"strides-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("strides");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).strides();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "strides", 1);
  }},
  {"sub-1-alpha-other", [](Node *node) {
    auto other = Scalar(node->t(Symbol("other")));
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sub");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sub(std::move(fromLast(stack, 1)), other, alpha);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sub", 1);
  }},
  {"sub-2-alpha", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sub");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::sub(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), alpha);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "sub", 2);
  }},
  {"sub-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sub");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto alpha = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::sub(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sub", 3);
  }},
  {"sum-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sum");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sum(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sum", 1);
  }},
  {"sum-1-dim-keepdim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sum");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::sum(std::move(fromLast(stack, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sum", 1);
  }},
  {"sum-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sum");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::sum(std::move(fromLast(stack, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sum", 3);
  }},
  {"svd-1-some", [](Node *node) {
    auto some = bool(node->i(Symbol("some")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("svd");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::svd(std::move(fromLast(stack, 1)), some);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "svd", 1);
  }},
  {"svd-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("svd");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto some = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::svd(std::move(fromLast(stack, 2)), some);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "svd", 2);
  }},
  {"symeig-1-eigenvectors-upper", [](Node *node) {
    auto eigenvectors = bool(node->i(Symbol("eigenvectors")));
    auto upper = bool(node->i(Symbol("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("symeig");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::symeig(std::move(fromLast(stack, 1)), eigenvectors, upper);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "symeig", 1);
  }},
  {"symeig-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("symeig");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto eigenvectors = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto upper = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::symeig(std::move(fromLast(stack, 3)), eigenvectors, upper);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "symeig", 3);
  }},
  {"t-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("t");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::t(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "t", 1);
  }},
  {"take-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("take");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::take(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "take", 2);
  }},
  {"tan-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tan");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::tan(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "tan", 1);
  }},
  {"tanh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tanh");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::tanh(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "tanh", 1);
  }},
  {"thnn_batch_norm-5-eps-momentum-training", [](Node *node) {
    auto training = bool(node->i(Symbol("training")));
    auto momentum = double(node->f(Symbol("momentum")));
    auto eps = double(node->f(Symbol("eps")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_batch_norm(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), training, momentum, eps);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm", 5);
  }},
  {"thnn_batch_norm-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto training = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto momentum = tensor_as<double>(std::move(fromLast(stack, 1)));
      auto eps = tensor_as<double>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_batch_norm(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), training, momentum, eps);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm", 8);
  }},
  {"thnn_batch_norm_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10 + 0));
      auto training = tensor_as<bool>(std::move(fromLast(stack, 4)));
      auto eps = tensor_as<double>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_batch_norm_backward(std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), training, eps, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_backward", 10);
  }},
  {"thnn_batch_norm_backward-7-eps-output_mask-training", [](Node *node) {
    auto training = bool(node->i(Symbol("training")));
    auto eps = double(node->f(Symbol("eps")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
  
      
      auto result = at::thnn_batch_norm_backward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), training, eps, std::move(fromLast(stack, 0)), std::move(fromLast(stack, -1)), output_mask);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_backward", 7);
  }},
  {"thnn_batch_norm_forward-5-eps-momentum-training", [](Node *node) {
    auto training = bool(node->i(Symbol("training")));
    auto momentum = double(node->f(Symbol("momentum")));
    auto eps = double(node->f(Symbol("eps")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_batch_norm_forward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), training, momentum, eps);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_forward", 5);
  }},
  {"thnn_batch_norm_forward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_forward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto training = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto momentum = tensor_as<double>(std::move(fromLast(stack, 1)));
      auto eps = tensor_as<double>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_batch_norm_forward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), training, momentum, eps);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_forward", 8);
  }},
  {"thnn_conv2d-3-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d", 3);
  }},
  {"thnn_conv2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv2d(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), kernel_size, std::move(fromLast(stack, 3)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d", 6);
  }},
  {"thnn_conv2d_backward-5-kernel_size-output_mask-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_conv2d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), kernel_size, stride, padding, std::move(fromLast(stack, -1)), std::move(fromLast(stack, -2)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_backward", 5);
  }},
  {"thnn_conv2d_backward-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv2d_backward(std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, stride, padding, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_backward", 9);
  }},
  {"thnn_conv2d_forward-3-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv2d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_forward", 3);
  }},
  {"thnn_conv2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv2d_forward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), kernel_size, std::move(fromLast(stack, 3)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_forward", 6);
  }},
  {"thnn_conv3d-3-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv3d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d", 3);
  }},
  {"thnn_conv3d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv3d(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), kernel_size, std::move(fromLast(stack, 3)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d", 6);
  }},
  {"thnn_conv3d_backward-5-kernel_size-output_mask-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_conv3d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), kernel_size, stride, padding, std::move(fromLast(stack, -1)), std::move(fromLast(stack, -2)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_backward", 5);
  }},
  {"thnn_conv3d_backward-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 9 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv3d_backward(std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, stride, padding, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_backward", 9);
  }},
  {"thnn_conv3d_forward-3-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv3d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_forward", 3);
  }},
  {"thnn_conv3d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv3d_forward(std::move(fromLast(stack, 6)), std::move(fromLast(stack, 5)), kernel_size, std::move(fromLast(stack, 3)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_forward", 6);
  }},
  {"thnn_conv_depthwise2d-3-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_depthwise2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d", 3);
  }},
  {"thnn_conv_depthwise2d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_depthwise2d(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, std::move(fromLast(stack, 4)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d", 7);
  }},
  {"thnn_conv_depthwise2d_backward-3-dilation-kernel_size-output_mask-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto output_mask = as_bool_array<2>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_depthwise2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), kernel_size, stride, padding, dilation, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_backward", 3);
  }},
  {"thnn_conv_depthwise2d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto output_mask = tensor_as<std::array<bool,2>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_depthwise2d_backward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, stride, padding, dilation, output_mask);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_backward", 8);
  }},
  {"thnn_conv_depthwise2d_forward-3-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_depthwise2d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_forward", 3);
  }},
  {"thnn_conv_depthwise2d_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_depthwise2d_forward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, std::move(fromLast(stack, 4)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_forward", 7);
  }},
  {"thnn_conv_dilated2d-3-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_dilated2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d", 3);
  }},
  {"thnn_conv_dilated2d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_dilated2d(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, std::move(fromLast(stack, 4)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d", 7);
  }},
  {"thnn_conv_dilated2d_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_dilated2d_backward(std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), kernel_size, stride, padding, dilation, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_backward", 10);
  }},
  {"thnn_conv_dilated2d_backward-5-dilation-kernel_size-output_mask-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_conv_dilated2d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), kernel_size, stride, padding, dilation, std::move(fromLast(stack, -2)), std::move(fromLast(stack, -3)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_backward", 5);
  }},
  {"thnn_conv_dilated2d_forward-3-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_dilated2d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_forward", 3);
  }},
  {"thnn_conv_dilated2d_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_dilated2d_forward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, std::move(fromLast(stack, 4)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_forward", 7);
  }},
  {"thnn_conv_dilated3d-3-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_dilated3d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d", 3);
  }},
  {"thnn_conv_dilated3d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_dilated3d(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, std::move(fromLast(stack, 4)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d", 7);
  }},
  {"thnn_conv_dilated3d_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_dilated3d_backward(std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), std::move(fromLast(stack, 8)), kernel_size, stride, padding, dilation, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_backward", 10);
  }},
  {"thnn_conv_dilated3d_backward-5-dilation-kernel_size-output_mask-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_conv_dilated3d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), kernel_size, stride, padding, dilation, std::move(fromLast(stack, -2)), std::move(fromLast(stack, -3)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_backward", 5);
  }},
  {"thnn_conv_dilated3d_forward-3-dilation-kernel_size-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_dilated3d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_forward", 3);
  }},
  {"thnn_conv_dilated3d_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_dilated3d_forward(std::move(fromLast(stack, 7)), std::move(fromLast(stack, 6)), kernel_size, std::move(fromLast(stack, 4)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_forward", 7);
  }},
  {"thnn_conv_transpose2d-3-dilation-kernel_size-output_padding-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_transpose2d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d", 3);
  }},
  {"thnn_conv_transpose2d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_transpose2d(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, std::move(fromLast(stack, 5)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d", 8);
  }},
  {"thnn_conv_transpose2d_backward-11", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 11 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 7)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_transpose2d_backward(std::move(fromLast(stack, 11)), std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), kernel_size, stride, padding, output_padding, dilation, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 11);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_backward", 11);
  }},
  {"thnn_conv_transpose2d_backward-5-dilation-kernel_size-output_mask-output_padding-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_conv_transpose2d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), kernel_size, stride, padding, output_padding, dilation, std::move(fromLast(stack, -3)), std::move(fromLast(stack, -4)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_backward", 5);
  }},
  {"thnn_conv_transpose2d_forward-3-dilation-kernel_size-output_padding-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_transpose2d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_forward", 3);
  }},
  {"thnn_conv_transpose2d_forward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_transpose2d_forward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, std::move(fromLast(stack, 5)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_forward", 8);
  }},
  {"thnn_conv_transpose3d-3-dilation-kernel_size-output_padding-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_transpose3d(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d", 3);
  }},
  {"thnn_conv_transpose3d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_transpose3d(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, std::move(fromLast(stack, 5)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d", 8);
  }},
  {"thnn_conv_transpose3d_backward-11", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 11 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 7)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 6)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 4)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_transpose3d_backward(std::move(fromLast(stack, 11)), std::move(fromLast(stack, 10)), std::move(fromLast(stack, 9)), kernel_size, stride, padding, output_padding, dilation, std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), output_mask);
      drop(stack, 11);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_backward", 11);
  }},
  {"thnn_conv_transpose3d_backward-5-dilation-kernel_size-output_mask-output_padding-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
  
      
      auto result = at::thnn_conv_transpose3d_backward(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), kernel_size, stride, padding, output_padding, dilation, std::move(fromLast(stack, -3)), std::move(fromLast(stack, -4)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_backward", 5);
  }},
  {"thnn_conv_transpose3d_forward-3-dilation-kernel_size-output_padding-padding-stride", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::thnn_conv_transpose3d_forward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), kernel_size, std::move(fromLast(stack, 0)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_forward", 3);
  }},
  {"thnn_conv_transpose3d_forward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 8 + 0));
      auto kernel_size = tensor_as<IntList>(std::move(fromLast(stack, 5)));
      auto stride = tensor_as<IntList>(std::move(fromLast(stack, 3)));
      auto padding = tensor_as<IntList>(std::move(fromLast(stack, 2)));
      auto output_padding = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto dilation = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::thnn_conv_transpose3d_forward(std::move(fromLast(stack, 8)), std::move(fromLast(stack, 7)), kernel_size, std::move(fromLast(stack, 5)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_forward", 8);
  }},
  {"threshold-1-threshold-value", [](Node *node) {
    auto threshold = Scalar(node->t(Symbol("threshold")));
    auto value = Scalar(node->t(Symbol("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::threshold(std::move(fromLast(stack, 1)), threshold, value);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "threshold", 1);
  }},
  {"threshold-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto threshold = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto value = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::threshold(std::move(fromLast(stack, 3)), threshold, value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "threshold", 3);
  }},
  {"threshold_backward-2-threshold-value", [](Node *node) {
    auto threshold = Scalar(node->t(Symbol("threshold")));
    auto value = Scalar(node->t(Symbol("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::threshold_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), threshold, value);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_backward", 2);
  }},
  {"threshold_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto threshold = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto value = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::threshold_backward(std::move(fromLast(stack, 4)), std::move(fromLast(stack, 3)), threshold, value);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_backward", 4);
  }},
  {"threshold_forward-1-threshold-value", [](Node *node) {
    auto threshold = Scalar(node->t(Symbol("threshold")));
    auto value = Scalar(node->t(Symbol("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::threshold_forward(std::move(fromLast(stack, 1)), threshold, value);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_forward", 1);
  }},
  {"threshold_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto threshold = tensor_as<Scalar>(std::move(fromLast(stack, 1)));
      auto value = tensor_as<Scalar>(std::move(fromLast(stack, 0)));
      
      auto result = at::threshold_forward(std::move(fromLast(stack, 3)), threshold, value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_forward", 3);
  }},
  {"to_dense-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("to_dense");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).to_dense();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "to_dense", 1);
  }},
  {"topk-1-dim-k-largest-sorted", [](Node *node) {
    auto k = int64_t(node->i(Symbol("k")));
    auto dim = int64_t(node->i(Symbol("dim")));
    auto largest = bool(node->i(Symbol("largest")));
    auto sorted = bool(node->i(Symbol("sorted")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("topk");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::topk(std::move(fromLast(stack, 1)), k, dim, largest, sorted);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "topk", 1);
  }},
  {"topk-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("topk");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto k = tensor_as<int64_t>(std::move(fromLast(stack, 3)));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto largest = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto sorted = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::topk(std::move(fromLast(stack, 5)), k, dim, largest, sorted);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "topk", 5);
  }},
  {"trace-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trace");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::trace(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "trace", 1);
  }},
  {"transpose-1-dim0-dim1", [](Node *node) {
    auto dim0 = int64_t(node->i(Symbol("dim0")));
    auto dim1 = int64_t(node->i(Symbol("dim1")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("transpose");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::transpose(std::move(fromLast(stack, 1)), dim0, dim1);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "transpose", 1);
  }},
  {"transpose-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("transpose");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto dim0 = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto dim1 = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::transpose(std::move(fromLast(stack, 3)), dim0, dim1);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "transpose", 3);
  }},
  {"tril-1-diagonal", [](Node *node) {
    auto diagonal = int64_t(node->i(Symbol("diagonal")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tril");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::tril(std::move(fromLast(stack, 1)), diagonal);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "tril", 1);
  }},
  {"tril-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tril");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto diagonal = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::tril(std::move(fromLast(stack, 2)), diagonal);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "tril", 2);
  }},
  {"triu-1-diagonal", [](Node *node) {
    auto diagonal = int64_t(node->i(Symbol("diagonal")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("triu");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::triu(std::move(fromLast(stack, 1)), diagonal);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "triu", 1);
  }},
  {"triu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("triu");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto diagonal = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::triu(std::move(fromLast(stack, 2)), diagonal);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "triu", 2);
  }},
  {"trtrs-2-transpose-unitriangular-upper", [](Node *node) {
    auto upper = bool(node->i(Symbol("upper")));
    auto transpose = bool(node->i(Symbol("transpose")));
    auto unitriangular = bool(node->i(Symbol("unitriangular")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trtrs");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::trtrs(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), upper, transpose, unitriangular);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "trtrs", 2);
  }},
  {"trtrs-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trtrs");
      AutoGPU device_guard(deviceForInputs(stack, 5 + 0));
      auto upper = tensor_as<bool>(std::move(fromLast(stack, 2)));
      auto transpose = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto unitriangular = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::trtrs(std::move(fromLast(stack, 5)), std::move(fromLast(stack, 4)), upper, transpose, unitriangular);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "trtrs", 5);
  }},
  {"trunc-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trunc");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::trunc(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "trunc", 1);
  }},
  {"type_as-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("type_as");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = (std::move(fromLast(stack, 2))).type_as(std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "type_as", 2);
  }},
  {"unfold-1-dimension-size-step", [](Node *node) {
    auto dimension = int64_t(node->i(Symbol("dimension")));
    auto size = int64_t(node->i(Symbol("size")));
    auto step = int64_t(node->i(Symbol("step")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unfold");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).unfold(dimension, size, step);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "unfold", 1);
  }},
  {"unfold-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unfold");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto dimension = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto size = tensor_as<int64_t>(std::move(fromLast(stack, 1)));
      auto step = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = (std::move(fromLast(stack, 4))).unfold(dimension, size, step);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "unfold", 4);
  }},
  {"unsqueeze-1-dim", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unsqueeze");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::unsqueeze(std::move(fromLast(stack, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "unsqueeze", 1);
  }},
  {"unsqueeze-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unsqueeze");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::unsqueeze(std::move(fromLast(stack, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "unsqueeze", 2);
  }},
  {"upsample_bilinear2d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_bilinear2d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d", 1);
  }},
  {"upsample_bilinear2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_bilinear2d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d", 2);
  }},
  {"upsample_bilinear2d_backward-1-input_size-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    auto input_size = std::vector<int64_t>(node->is(Symbol("input_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_bilinear2d_backward(std::move(fromLast(stack, 1)), output_size, input_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_backward", 1);
  }},
  {"upsample_bilinear2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto input_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_bilinear2d_backward(std::move(fromLast(stack, 3)), output_size, input_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_backward", 3);
  }},
  {"upsample_bilinear2d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_bilinear2d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_forward", 1);
  }},
  {"upsample_bilinear2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_bilinear2d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_forward", 2);
  }},
  {"upsample_linear1d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_linear1d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d", 1);
  }},
  {"upsample_linear1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_linear1d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d", 2);
  }},
  {"upsample_linear1d_backward-1-input_size-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    auto input_size = std::vector<int64_t>(node->is(Symbol("input_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_linear1d_backward(std::move(fromLast(stack, 1)), output_size, input_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_backward", 1);
  }},
  {"upsample_linear1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto input_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_linear1d_backward(std::move(fromLast(stack, 3)), output_size, input_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_backward", 3);
  }},
  {"upsample_linear1d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_linear1d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_forward", 1);
  }},
  {"upsample_linear1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_linear1d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_forward", 2);
  }},
  {"upsample_nearest1d-1-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_nearest1d(std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d", 1);
  }},
  {"upsample_nearest1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest1d(std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d", 2);
  }},
  {"upsample_nearest1d_backward-2-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::upsample_nearest1d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_backward", 2);
  }},
  {"upsample_nearest1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest1d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_backward", 3);
  }},
  {"upsample_nearest1d_forward-1-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_nearest1d_forward(std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_forward", 1);
  }},
  {"upsample_nearest1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest1d_forward(std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_forward", 2);
  }},
  {"upsample_nearest2d-1-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_nearest2d(std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d", 1);
  }},
  {"upsample_nearest2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest2d(std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d", 2);
  }},
  {"upsample_nearest2d_backward-2-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::upsample_nearest2d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_backward", 2);
  }},
  {"upsample_nearest2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest2d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_backward", 3);
  }},
  {"upsample_nearest2d_forward-1-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_nearest2d_forward(std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_forward", 1);
  }},
  {"upsample_nearest2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest2d_forward(std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_forward", 2);
  }},
  {"upsample_nearest3d-1-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_nearest3d(std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d", 1);
  }},
  {"upsample_nearest3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest3d(std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d", 2);
  }},
  {"upsample_nearest3d_backward-2-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = at::upsample_nearest3d_backward(std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_backward", 2);
  }},
  {"upsample_nearest3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest3d_backward(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_backward", 3);
  }},
  {"upsample_nearest3d_forward-1-scale_factor", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_nearest3d_forward(std::move(fromLast(stack, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_forward", 1);
  }},
  {"upsample_nearest3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto scale_factor = tensor_as<int64_t>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_nearest3d_forward(std::move(fromLast(stack, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_forward", 2);
  }},
  {"upsample_trilinear3d-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_trilinear3d(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d", 1);
  }},
  {"upsample_trilinear3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_trilinear3d(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d", 2);
  }},
  {"upsample_trilinear3d_backward-1-input_size-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    auto input_size = std::vector<int64_t>(node->is(Symbol("input_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_trilinear3d_backward(std::move(fromLast(stack, 1)), output_size, input_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_backward", 1);
  }},
  {"upsample_trilinear3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 1)));
      auto input_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_trilinear3d_backward(std::move(fromLast(stack, 3)), output_size, input_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_backward", 3);
  }},
  {"upsample_trilinear3d_forward-1-output_size", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::upsample_trilinear3d_forward(std::move(fromLast(stack, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_forward", 1);
  }},
  {"upsample_trilinear3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto output_size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = at::upsample_trilinear3d_forward(std::move(fromLast(stack, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_forward", 2);
  }},
  {"var-1-dim-keepdim-unbiased", [](Node *node) {
    auto dim = int64_t(node->i(Symbol("dim")));
    auto unbiased = bool(node->i(Symbol("unbiased")));
    auto keepdim = bool(node->i(Symbol("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::var(std::move(fromLast(stack, 1)), dim, unbiased, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "var", 1);
  }},
  {"var-1-unbiased", [](Node *node) {
    auto unbiased = bool(node->i(Symbol("unbiased")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::var(std::move(fromLast(stack, 1)), unbiased);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "var", 1);
  }},
  {"var-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto unbiased = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::var(std::move(fromLast(stack, 2)), unbiased);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "var", 2);
  }},
  {"var-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 4 + 0));
      auto dim = tensor_as<int64_t>(std::move(fromLast(stack, 2)));
      auto unbiased = tensor_as<bool>(std::move(fromLast(stack, 1)));
      auto keepdim = tensor_as<bool>(std::move(fromLast(stack, 0)));
      
      auto result = at::var(std::move(fromLast(stack, 4)), dim, unbiased, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "var", 4);
  }},
  {"view-1-size", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("view");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = (std::move(fromLast(stack, 1))).view(size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "view", 1);
  }},
  {"view-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("view");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
      auto size = tensor_as<IntList>(std::move(fromLast(stack, 0)));
      
      auto result = (std::move(fromLast(stack, 2))).view(size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "view", 2);
  }},
  {"view_as-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("view_as");
      AutoGPU device_guard(deviceForInputs(stack, 2 + 0));
  
      
      auto result = (std::move(fromLast(stack, 2))).view_as(std::move(fromLast(stack, 1)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "view_as", 2);
  }},
  {"where-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("where");
      AutoGPU device_guard(deviceForInputs(stack, 3 + 0));
  
      
      auto result = at::where(std::move(fromLast(stack, 3)), std::move(fromLast(stack, 2)), std::move(fromLast(stack, 1)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "where", 3);
  }},
  {"zeros_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("zeros_like");
      AutoGPU device_guard(deviceForInputs(stack, 1 + 0));
  
      
      auto result = at::zeros_like(std::move(fromLast(stack, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "zeros_like", 1);
  }},
};

std::string getDescriptor(jit::Node* n) {
  std::stringstream s;
  s << n->kind().toString();
  if (tensor_vararg_fns.count(n->kind()) == 0)
    s << "-" << n->inputs().size();
  else
    s << "-*";
  std::vector<const char*> attr_names = fmap(n->attributeNames(), [](Symbol x) { return x.toString(); });
  std::sort(attr_names.begin(), attr_names.end(), [](const char *a, const char *b) {
    return std::strcmp(a, b) < 0;
  });
  for (const auto & name : attr_names)
    s << "-" << name;
  return s.str();
}

} // anonymous namespace

TensorOp getTensorOp(jit::Node* n) {
  auto signature = getDescriptor(n);
  try {
    return constructors.at(signature)(n);
  } catch (std::out_of_range &e) {
    throw std::runtime_error("Unsupported op descriptor: " + signature + ". "
                             "File a bug report.");
  }
};

}} // namespace torch::jit
