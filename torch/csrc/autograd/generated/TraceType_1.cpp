#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

// @generated from tools\autograd\templates/TraceType.cpp

// See the `Tracer` section in `torch/csrc/jit/OVERVIEW.md`.
// NOTE See [Sharded File] comment in VariableType

using namespace at;

namespace torch {

namespace TraceType {

namespace {
Tensor & __irshift___Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__irshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__irshift__", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & __irshift___Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__irshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__irshift__", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor __rshift___Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__rshift__", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor __rshift___Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__rshift__", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, vec1, vec2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_addr");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_addr_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_addr_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, vec1, vec2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _amp_update_scale(Tensor & growth_tracker, const Tensor & current_scale, const Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_amp_update_scale");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "growth_tracker", growth_tracker);
    jit::tracer::addInputs(node, "current_scale", current_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    jit::tracer::addInputs(node, "scale_growth_factor", scale_growth_factor);
    jit::tracer::addInputs(node, "scale_backoff_factor", scale_backoff_factor);
    jit::tracer::addInputs(node, "growth_interval", growth_interval);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_amp_update_scale", "")
      .typed<Tensor (Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Tracer, growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _bmm(const Tensor & self, const Tensor & mat2, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_bmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, mat2, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cast_Byte(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Byte");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Byte", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cast_Half(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Half");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Half", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cast_Int(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Int");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Int", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _cat_out_out(Tensor & out, TensorList tensors, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cat", "out")
      .typed<Tensor & (Tensor &, TensorList, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, int64_t>(op, c10::DispatchKey::Tracer, out, tensors, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor _cdist_backward(const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cdist_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "cdist", cdist);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cdist_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Tracer, grad, x1, x2, p, cdist);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _cholesky_helper(const Tensor & self, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cholesky_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cholesky_helper", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _coalesced_(Tensor & self, bool coalesced) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_coalesced_", "")
      .typed<Tensor & (Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, bool>(op, c10::DispatchKey::Tracer, self, coalesced);
  return self;
}
Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "weight_buf", weight_buf);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
    jit::tracer::addInputs(node, "reserve", reserve);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  std::vector<Tensor> result3;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>)>();
  std::tie(result0, result1, result2, result3) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>>(op, c10::DispatchKey::Tracer, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
int64_t _cufft_get_plan_cache_size(int64_t device_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_get_plan_cache_size", "")
      .typed<int64_t (int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, int64_t>(op, c10::DispatchKey::Tracer, device_index);
  return result;
}
void _cummax_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cummax_helper", "")
      .typed<void (const Tensor &, Tensor &, Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<void, const Tensor &, Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, values, indices, dim);
}
Tensor & _cumprod_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cumprod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumprod", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, out, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor _cumsum(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumsum", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _dim_arange(const Tensor & like, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_dim_arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "like", like);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dim_arange", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, like, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_dirichlet_grad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "total", total);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dirichlet_grad", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, x, alpha, total);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &>(op, c10::DispatchKey::Tracer, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _fused_dropout(const Tensor & self, double p, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_fused_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_fused_dropout", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, double, c10::optional<Generator>)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, p, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _logcumsumexp(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_logcumsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_logcumsumexp", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _make_per_tensor_quantized_tensor(const Tensor & self, double scale, int64_t zero_point) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_make_per_tensor_quantized_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_make_per_tensor_quantized_tensor", "")
      .typed<Tensor (const Tensor &, double, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t>(op, c10::DispatchKey::Tracer, self, scale, zero_point);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _mode(const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mode", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor _nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, input, weight, bias, padding, stride);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_softmax_backward_data");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, output, dim, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_log_softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax", "int")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_log_softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_log_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, dim, half_to_float);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, grad, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_gru_cell_backward(const Tensor & grad_hy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_differentiable_gru_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_differentiable_gru_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  std::tie(result0, result1, result2, result3, result4) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward(const Tensor & grad_hy, const Tensor & workspace, bool has_bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_gru_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  Tensor result4;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_gru_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, bool)>();
  std::tie(result0, result1, result2, result3, result4) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, grad_hy, workspace, has_bias);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
bool _use_cudnn_rnn_flatten_weight() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_use_cudnn_rnn_flatten_weight", "")
      .typed<bool ()>();
  auto result =c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Tracer);
  return result;
}
Tensor _values(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_values", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_cuda_interface_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_w", grad_w);
    jit::tracer::addInputs(node, "saved_v", saved_v);
    jit::tracer::addInputs(node, "saved_g", saved_g);
    jit::tracer::addInputs(node, "saved_norms", saved_norms);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_cuda_interface_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_w, saved_v, saved_g, saved_norms, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & acos_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::acos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acos", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor acosh(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::acosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acosh", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & acosh_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::acosh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::acosh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acosh_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & adaptive_avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, indices);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & add_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, self, other, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, vec1, vec2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addr");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addr_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, vec1, vec2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor affine_grid_generator(const Tensor & theta, IntArrayRef size, bool align_corners) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::affine_grid_generator");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "theta", theta);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::affine_grid_generator", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, theta, size, align_corners);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & arange_out_out(Tensor & out, Scalar end) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "end", end);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arange_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "out")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, end);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & arange_out_start_out(Tensor & out, Scalar start, Scalar end, Scalar step) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arange_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "start_out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, start, end, step);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & asin_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::asin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asin", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor asinh(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::asinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asinh", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & asinh_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::asinh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::asinh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asinh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asinh_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & avg_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::baddbmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, batch1, batch2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::baddbmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::baddbmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::baddbmm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, batch1, batch2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, bool input_g, bool weight_g, bool bias_g) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_backward_reduce");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "input_g", input_g);
    jit::tracer::addInputs(node, "weight_g", weight_g);
    jit::tracer::addInputs(node, "bias_g", bias_g);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_backward_reduce", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool)>();
  std::tie(result0, result1, result2, result3) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Tracer, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor & bernoulli_out_out(Tensor & out, const Tensor & self, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli", "out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, self, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_with_logits");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "pos_weight", pos_weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_with_logits", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, target, weight, pos_weight, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bincount");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weights", weights);
    jit::tracer::addInputs(node, "minlength", minlength);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bincount", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, weights, minlength);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bitwise_and_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bitwise_and_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & bitwise_and__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & bitwise_and__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_and_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor bitwise_not(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & bitwise_not_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_not");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_not_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor bmm(const Tensor & self, const Tensor & mat2) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bmm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, mat2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bucketize_Tensor(const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, boundaries, out_int32, right);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor bucketize_Scalar(Scalar self, const Tensor & boundaries, bool out_int32, bool right) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Scalar")
      .typed<Tensor (Scalar, const Tensor &, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, Scalar, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, boundaries, out_int32, right);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor cartesian_prod(TensorList tensors) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cartesian_prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cartesian_prod", "")
      .typed<Tensor (TensorList)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Tracer, tensors);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & cat_out_out(Tensor & out, TensorList tensors, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cat", "out")
      .typed<Tensor & (Tensor &, TensorList, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, int64_t>(op, c10::DispatchKey::Tracer, out, tensors, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & cat_out_names_out(Tensor & out, TensorList tensors, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cat", "names_out")
      .typed<Tensor & (Tensor &, TensorList, Dimname)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, Dimname>(op, c10::DispatchKey::Tracer, out, tensors, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor chain_matmul(TensorList matrices) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::chain_matmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "matrices", matrices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::chain_matmul", "")
      .typed<Tensor (TensorList)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Tracer, matrices);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor cholesky(const Tensor & self, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor clamp_max(const Tensor & self, Scalar max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & clamp_max_(Tensor & self, Scalar max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp_max");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_max_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor coalesce(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::coalesce");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::coalesce", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & col2im_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::col2im");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("col2im_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, output_size, kernel_size, dilation, padding, stride);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor combinations(const Tensor & self, int64_t r, bool with_replacement) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::combinations");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "r", r);
    jit::tracer::addInputs(node, "with_replacement", with_replacement);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::combinations", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, r, with_replacement);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor conj(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conj");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conj", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conv_tbc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_tbc", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, weight, bias, pad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  return result;
}
Tensor cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosine_similarity");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosine_similarity", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, double)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, double>(op, c10::DispatchKey::Tracer, x1, x2, dim, eps);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor cudnn_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> cummax(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> cummax_dimname(const Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & cumprod_out_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumprod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumprod", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, out, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & cumprod_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumprod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumprod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumprod", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, out, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor cumsum(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor cumsum_dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
int64_t dense_dim(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dense_dim", "")
      .typed<int64_t (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
Tensor diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diagonal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diagonal", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, offset, dim1, dim2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor diagonal_Dimname(const Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diagonal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "outdim", outdim);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diagonal", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, Dimname, Dimname, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, Dimname, Dimname, int64_t>(op, c10::DispatchKey::Tracer, self, outdim, dim1, dim2, offset);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor dist(const Tensor & self, const Tensor & other, Scalar p) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dist", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other, p);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & dot_out_out(Tensor & out, const Tensor & self, const Tensor & tensor) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor", tensor);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dot", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, tensor);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor dropout(const Tensor & input, double p, bool train) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dropout", "")
      .typed<Tensor (const Tensor &, double, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Tracer, input, p, train);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & dropout_(Tensor & self, double p, bool train) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dropout_", "")
      .typed<Tensor & (Tensor &, double, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, bool>(op, c10::DispatchKey::Tracer, self, p, train);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::elu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, alpha, scale, input_scale);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::elu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::elu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_", "")
      .typed<Tensor & (Tensor &, Scalar, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, alpha, scale, input_scale);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & elu_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::elu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, alpha, scale, input_scale, output);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::embedding_renorm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::embedding_renorm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "max_norm", max_norm);
    jit::tracer::addInputs(node, "norm_type", norm_type);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("embedding_renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_renorm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, double, double)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, double, double>(op, c10::DispatchKey::Tracer, self, indices, max_norm, norm_type);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
bool equal(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::equal", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  return result;
}
Tensor & erf_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erf", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor erfc(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfc", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & erfc_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::erfc");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erfc_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfc_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor expm1(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expm1");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & expm1_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::expm1");
    } else {
      op_name = jit::Symbol::fromQualString("aten::expm1_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expm1_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & exponential_(Tensor & self, double lambd, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::exponential");
    } else {
      op_name = jit::Symbol::fromQualString("aten::exponential_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exponential_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exponential_", "")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, lambd, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor fake_quantize_per_channel_affine(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_channel_affine");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_channel_affine", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, scale, zero_point, axis, quant_min, quant_max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fbgemm_linear_fp16_weight_fp32_activation(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight_fp32_activation");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_fp16_weight_fp32_activation", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input, packed_weight, bias);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fbgemm_linear_int8_weight_fp32_activation(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_int8_weight_fp32_activation");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "packed", packed);
    jit::tracer::addInputs(node, "col_offsets", col_offsets);
    jit::tracer::addInputs(node, "weight_scale", weight_scale);
    jit::tracer::addInputs(node, "weight_zero_point", weight_zero_point);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_int8_weight_fp32_activation", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,double,int64_t> fbgemm_linear_quantize_weight(const Tensor & input) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_quantize_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  double result2;
  int64_t result3;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_quantize_weight", "")
      .typed<std::tuple<Tensor,Tensor,double,int64_t> (const Tensor &)>();
  std::tie(result0, result1, result2, result3) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,double,int64_t>, const Tensor &>(op, c10::DispatchKey::Tracer, input);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
Tensor floor(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & floor_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::floor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::floor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & fmod_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & fmod_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fmod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & frac_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frac");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frac_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frac", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out_output(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool2d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Tracer, output, indices, self, kernel_size, output_size, random_samples);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(output, indices);
}
Tensor & frobenius_norm_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frobenius_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frobenius_norm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, out, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full_like", "")
      .typed<Tensor (const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Tracer, self, fill_value, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::group_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "num_groups", num_groups);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::group_norm", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool>(op, c10::DispatchKey::Tracer, input, num_groups, weight, bias, eps, cudnn_enabled);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hamming_window(int64_t window_length, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Tracer, window_length, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hamming_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "periodic")
      .typed<Tensor (int64_t, bool, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, const TensorOptions &>(op, c10::DispatchKey::Tracer, window_length, periodic, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hamming_window_periodic_alpha(int64_t window_length, bool periodic, double alpha, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "periodic_alpha")
      .typed<Tensor (int64_t, bool, double, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, double, const TensorOptions &>(op, c10::DispatchKey::Tracer, window_length, periodic, alpha, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hamming_window_periodic_alpha_beta(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "periodic_alpha_beta")
      .typed<Tensor (int64_t, bool, double, double, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, double, double, const TensorOptions &>(op, c10::DispatchKey::Tracer, window_length, periodic, alpha, beta, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardshrink_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, grad_out, self, lambd);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & hardtanh_out_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, min_val, max_val);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & im2col_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::im2col");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("im2col_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, kernel_size, dilation, padding, stride);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor index_Tensor(const Tensor & self, TensorList indices) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices, true);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index", "Tensor")
      .typed<Tensor (const Tensor &, TensorList)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, TensorList>(op, c10::DispatchKey::Tracer, self, indices);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_put");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices, true);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_put", "")
      .typed<Tensor (const Tensor &, TensorList, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, TensorList, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, indices, values, accumulate);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_put");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_put_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices, true);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_put_", "")
      .typed<Tensor & (Tensor &, TensorList, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, indices, values, accumulate);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_select_dimname(const Tensor & self, Dimname dim, const Tensor & index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
bool is_coalesced(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_coalesced", "")
      .typed<bool (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
bool is_floating_point(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_floating_point", "")
      .typed<bool (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
bool is_vulkan_available() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_vulkan_available", "")
      .typed<bool ()>();
  auto result =c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Tracer);
  return result;
}
Scalar item(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::item", "")
      .typed<Scalar (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & l1_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::l1_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("l1_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & linspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::linspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linspace", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, int64_t>(op, c10::DispatchKey::Tracer, out, start, end, steps);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & log2_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log2", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & log_normal_(Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log_normal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log_normal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_normal_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, mean, std, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & log_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, buffer);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out_output(Tensor & output, Tensor & buffer, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "buffer", buffer);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, output, buffer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  #endif
  return std::forward_as_tuple(output, buffer);
}
Tensor & logaddexp2_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logaddexp2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logaddexp2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logaddexp2", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & logaddexp_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logaddexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logaddexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logaddexp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor logcumsumexp(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor logcumsumexp_dimname(const Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "dimname")
      .typed<Tensor (const Tensor &, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor logical_or(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & logical_or_(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::logical_or");
    } else {
      op_name = jit::Symbol::fromQualString("aten::logical_or_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or_", "")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor logical_xor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logical_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & logical_xor_(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::logical_xor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::logical_xor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor_", "")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logspace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logspace", "")
      .typed<Tensor (Scalar, Scalar, int64_t, double, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, int64_t, double, const TensorOptions &>(op, c10::DispatchKey::Tracer, start, end, steps, base, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor logsumexp(const Tensor & self, IntArrayRef dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor logsumexp_names(const Tensor & self, DimnameList dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logsumexp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "names")
      .typed<Tensor (const Tensor &, DimnameList, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> lstsq_out_X(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lstsq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "qr", qr);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "X", X);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lstsq_out", X);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstsq", "X")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, X, qr, self, A);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, X);
    jit::tracer::addOutput(node, qr);
  }
  #endif
  return std::forward_as_tuple(X, qr);
}
Tensor matmul(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matmul", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> max_dim(const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> max_names_dim(const Tensor & self, Dimname dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor max_other(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "other")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor max(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool1d_with_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool1d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, self, kernel_size, stride, padding, dilation, ceil_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_with_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(out, indices);
}
Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_output, self, indices, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, self, indices, output_size, stride, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & max_unpool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, indices, output_size, stride, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & mean_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mean_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, out, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & mean_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mean_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, out, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor &,Tensor &> median_out_dim_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "dim_values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, values, indices, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> median_out_names_dim_values(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "names_dim_values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, values, indices, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::vector<Tensor> meshgrid(TensorList tensors) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::meshgrid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::meshgrid", "")
      .typed<std::vector<Tensor> (TensorList)>();
  auto result =c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList>(op, c10::DispatchKey::Tracer, tensors);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "exponential_average_factor", exponential_average_factor);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Tracer, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Tracer, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor miopen_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward_input", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor miopen_depthwise_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_depthwise_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, std::array<bool,3>)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Tracer, self, grad_output, weight, padding, stride, dilation, groups, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor mkldnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution_backward_input", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool>(op, c10::DispatchKey::Tracer, self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> mode_dimname(const Tensor & self, Dimname dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor & multi_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multi_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multi_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, out, self, target, p, margin, weight, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor output;
  Tensor is_target;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t)>();
  std::tie(output, is_target) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  #endif
  return std::make_tuple(std::move(output), std::move(is_target));
}
Tensor & mv_out_out(Tensor & out, const Tensor & self, const Tensor & vec) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec", vec);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, vec);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_invstd", save_invstd);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>>(op, c10::DispatchKey::Tracer, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor native_norm(const Tensor & self, Scalar p) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_norm", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, p);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor ne_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor ne_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ne");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & ne__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ne");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & ne__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ne");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out_output(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, output, total_weight, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  #endif
  return std::forward_as_tuple(output, total_weight);
}
Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> nll_loss_forward_out_output(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, output, total_weight, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  #endif
  return std::forward_as_tuple(output, total_weight);
}
Tensor & ones_out_out(Tensor & out, IntArrayRef size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ones_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones", "out")
      .typed<Tensor & (Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, out, size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ormqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "input3", input3);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "transpose", transpose);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ormqr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, input2, input3, left, transpose);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pairwise_distance");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pairwise_distance", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, double, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, double, bool>(op, c10::DispatchKey::Tracer, x1, x2, p, eps, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor pinverse(const Tensor & self, double rcond) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pinverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pinverse", "")
      .typed<Tensor (const Tensor &, double)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Tracer, self, rcond);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & polygamma_out_out(Tensor & out, int64_t n, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::polygamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polygamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::polygamma", "out")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, out, n, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & pow_out_Tensor_Scalar_out(Tensor & out, const Tensor & self, Scalar exponent) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Tensor_Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, self, exponent);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & pow_out_Tensor_Tensor_out(Tensor & out, const Tensor & self, const Tensor & exponent) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Tensor_Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, exponent);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & pow_out_Scalar_out(Tensor & out, Scalar self, const Tensor & exponent) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Scalar_out")
      .typed<Tensor & (Tensor &, Scalar, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, exponent);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & prod_out_int_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "int_out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool, c10::optional<ScalarType>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, out, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & prod_out_Dimname_out(Tensor & out, const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "Dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, bool, c10::optional<ScalarType>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, out, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
int64_t q_per_channel_axis(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_axis", "")
      .typed<int64_t (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
Tensor q_per_channel_zero_points(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::q_per_channel_zero_points");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_zero_points", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> qr_out_Q(Tensor & Q, Tensor & R, const Tensor & self, bool some) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::qr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "R", R);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "Q", Q);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("qr_out", Q);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::qr", "Q")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, Q, R, self, some);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  #endif
  return std::forward_as_tuple(Q, R);
}
Tensor quantized_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_gru_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_gru_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor quantized_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_rnn_relu_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_rnn_relu_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rand_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, names, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rand_generator_with_names(IntArrayRef size, c10::optional<Generator> generator, c10::optional<DimnameList> names, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "generator_with_names")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, generator, names, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rand(IntArrayRef size, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rand_generator(IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "generator")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, generator, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & randint_out_out(Tensor & out, int64_t high, IntArrayRef size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "out")
      .typed<Tensor & (Tensor &, int64_t, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, IntArrayRef>(op, c10::DispatchKey::Tracer, out, high, size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randint_out_generator_out(Tensor & out, int64_t high, IntArrayRef size, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "generator_out")
      .typed<Tensor & (Tensor &, int64_t, IntArrayRef, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, high, size, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randint_out_low_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "low_out")
      .typed<Tensor & (Tensor &, int64_t, int64_t, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t, IntArrayRef>(op, c10::DispatchKey::Tracer, out, low, high, size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randint_out_low_generator_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randint");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "low_generator_out")
      .typed<Tensor & (Tensor &, int64_t, int64_t, IntArrayRef, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, low, high, size, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor randn(IntArrayRef size, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor randn_generator(IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "generator")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, generator, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor randn_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, names, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor randn_generator_with_names(IntArrayRef size, c10::optional<Generator> generator, c10::optional<DimnameList> names, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "generator_with_names")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, generator, names, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & random__from(Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::random");
    } else {
      op_name = jit::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::random_", "from")
      .typed<Tensor & (Tensor &, int64_t, c10::optional<int64_t>, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, c10::optional<int64_t>, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, from, to, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & random__to(Tensor & self, int64_t to, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::random");
    } else {
      op_name = jit::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::random_", "to")
      .typed<Tensor & (Tensor &, int64_t, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, to, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & random_(Tensor & self, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::random");
    } else {
      op_name = jit::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::random_", "")
      .typed<Tensor & (Tensor &, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & randperm_out_out(Tensor & out, int64_t n) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randperm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randperm", "out")
      .typed<Tensor & (Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Tracer, out, n);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randperm_out_generator_out(Tensor & out, int64_t n, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randperm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randperm", "generator_out")
      .typed<Tensor & (Tensor &, int64_t, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, n, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_output, self, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor reflection_pad2d(const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & reflection_pad2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor remainder_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor remainder_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::remainder");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & remainder__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::remainder");
    } else {
      op_name = jit::Symbol::fromQualString("aten::remainder_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & remainder__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::remainder");
    } else {
      op_name = jit::Symbol::fromQualString("aten::remainder_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor repeat(const Tensor & self, IntArrayRef repeats) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::repeat");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::repeat", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, repeats);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor replication_pad1d(const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & replication_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & replication_pad2d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor reshape_as(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reshape_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reshape_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
ScalarType result_type_Tensor(const Tensor & tensor, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Tensor")
      .typed<ScalarType (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<ScalarType, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, tensor, other);
  return result;
}
ScalarType result_type_Scalar(const Tensor & tensor, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Scalar")
      .typed<ScalarType (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<ScalarType, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, tensor, other);
  return result;
}
ScalarType result_type_Scalar_Tensor(Scalar scalar, const Tensor & tensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Scalar_Tensor")
      .typed<ScalarType (Scalar, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<ScalarType, Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, scalar, tensor);
  return result;
}
ScalarType result_type_Scalar_Scalar(Scalar scalar1, Scalar scalar2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Scalar_Scalar")
      .typed<ScalarType (Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<ScalarType, Scalar, Scalar>(op, c10::DispatchKey::Tracer, scalar1, scalar2);
  return result;
}
std::tuple<Tensor,Tensor> rnn_tanh_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_tanh", "input")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool>(op, c10::DispatchKey::Tracer, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> rnn_tanh_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_tanh", "data")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool>(op, c10::DispatchKey::Tracer, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::roll");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shifts", shifts);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::roll", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, self, shifts, dims);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rot90(const Tensor & self, int64_t k, IntArrayRef dims) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rot90");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rot90", "")
      .typed<Tensor (const Tensor &, int64_t, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, IntArrayRef>(op, c10::DispatchKey::Tracer, self, k, dims);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & round_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::round");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::round", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, bool self_is_result) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, bool>(op, c10::DispatchKey::Tracer, grad_output, self, noise, lower, upper, training, self_is_result);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & rsqrt_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rsqrt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsqrt", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & searchsorted_out_Tensor_out(Tensor & out, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::searchsorted");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sorted_sequence", sorted_sequence);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("searchsorted_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::searchsorted", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, out, sorted_sequence, self, out_int32, right);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor selu(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::selu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::selu", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & selu_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::selu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::selu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("selu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::selu_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & slow_conv3d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, weight, kernel_size, bias, stride, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & slow_conv_transpose2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_transpose2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv_transpose2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & smooth_l1_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("smooth_l1_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, out, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & soft_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::soft_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, beta, threshold);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & softplus_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softplus_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "output", output);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, beta, threshold, output);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
std::tuple<Tensor,Tensor> sort_dimname(const Tensor & self, Dimname dim, bool descending) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
std::vector<Tensor> split_Tensor(const Tensor & self, int64_t split_size, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::split");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_size", split_size);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::split", "Tensor")
      .typed<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, split_size, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sspaddmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sspaddmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sspaddmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sspaddmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, mat1, mat2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor std(const Tensor & self, bool unbiased) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, unbiased);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor std_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Tracer, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor std_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::std");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "names_dim")
      .typed<Tensor (const Tensor &, DimnameList, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Tracer, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
int64_t stride_int(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stride", "int")
      .typed<int64_t (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  return result;
}
int64_t stride_Dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stride", "Dimname")
      .typed<int64_t (const Tensor &, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &, Dimname>(op, c10::DispatchKey::Tracer, self, dim);
  return result;
}
Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "")
      .typed<Tensor (const Tensor &, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor sum_dim_IntList(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "dim_IntList")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor sum_dim_DimnameList(const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "dim_DimnameList")
      .typed<Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Tracer, self, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor sum_to_size(const Tensor & self, IntArrayRef size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sum_to_size");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum_to_size", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor t(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::t");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::t", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & t_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::t");
    } else {
      op_name = jit::Symbol::fromQualString("aten::t_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("t_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::t_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor take(const Tensor & self, const Tensor & index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::take");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::take", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & tanh_out_out(Tensor & out, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, self, weight, kernel_size, bias, stride, padding, dilation);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_weight", grad_weight);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
  }
  #endif
  return std::forward_as_tuple(grad_input, grad_weight);
}
Tensor to_mkldnn(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_mkldnn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_mkldnn", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor to_sparse_sparse_dim(const Tensor & self, int64_t sparse_dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_sparse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_sparse", "sparse_dim")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, sparse_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor to_sparse(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_sparse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_sparse", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::topk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor values;
  Tensor indices;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::topk", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool)>();
  std::tie(values, indices) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, self, k, dim, largest, sorted);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor transpose_int(const Tensor & self, int64_t dim0, int64_t dim1) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::transpose", "int")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, dim0, dim1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor transpose_Dimname(const Tensor & self, Dimname dim0, Dimname dim1) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::transpose", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, Dimname>(op, c10::DispatchKey::Tracer, self, dim0, dim1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::transpose");
    } else {
      op_name = jit::Symbol::fromQualString("aten::transpose_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("transpose_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::transpose_", "")
      .typed<Tensor & (Tensor &, int64_t, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, dim0, dim1);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor trapz_x(const Tensor & y, const Tensor & x, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trapz");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trapz", "x")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, y, x, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor trapz_dx(const Tensor & y, double dx, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trapz");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trapz", "dx")
      .typed<Tensor (const Tensor &, double, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t>(op, c10::DispatchKey::Tracer, y, dx, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & triu_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("triu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, out, self, diagonal);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & true_divide_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::true_divide");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("true_divide_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::true_divide", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor trunc(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trunc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & trunc_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::trunc");
    } else {
      op_name = jit::Symbol::fromQualString("aten::trunc_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc_", "")
      .typed<Tensor & (Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & upsample_bicubic2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bicubic2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bicubic2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, out, self, output_size, align_corners, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_output, output_size, input_size, align_corners, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_output, output_size, input_size, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, self, output_size, scales_d, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_nearest3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & upsample_trilinear3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_trilinear3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, out, self, output_size, align_corners, scales_d, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor values(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::values", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> var_mean(const Tensor & self, bool unbiased) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var_mean", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, unbiased);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> var_mean_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var_mean", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Tracer, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> var_mean_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var_mean", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, DimnameList, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Tracer, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl_UNBOXED("__irshift__.Scalar", &TraceType::__irshift___Scalar);
  m.impl_UNBOXED("__irshift__.Tensor", &TraceType::__irshift___Tensor);
  m.impl("__rshift__.Scalar", TORCH_FN(TraceType::__rshift___Scalar));
  m.impl("__rshift__.Tensor", TORCH_FN(TraceType::__rshift___Tensor));
  m.impl("_adaptive_avg_pool2d", TORCH_FN(TraceType::_adaptive_avg_pool2d));
  m.impl("_addr", TORCH_FN(TraceType::_addr));
  m.impl_UNBOXED("_addr_", &TraceType::_addr_);
  m.impl_UNBOXED("_amp_update_scale", &TraceType::_amp_update_scale);
  m.impl("_bmm", TORCH_FN(TraceType::_bmm));
  m.impl("_cast_Byte", TORCH_FN(TraceType::_cast_Byte));
  m.impl("_cast_Half", TORCH_FN(TraceType::_cast_Half));
  m.impl("_cast_Int", TORCH_FN(TraceType::_cast_Int));
  m.impl_UNBOXED("_cat.out", &TraceType::_cat_out_out);
  m.impl("_cdist_backward", TORCH_FN(TraceType::_cdist_backward));
  m.impl("_cholesky_helper", TORCH_FN(TraceType::_cholesky_helper));
  m.impl_UNBOXED("_coalesced_", &TraceType::_coalesced_);
  m.impl_UNBOXED("_convolution", &TraceType::_convolution);
  m.impl_UNBOXED("_cudnn_rnn_backward", &TraceType::_cudnn_rnn_backward);
  m.impl("_cufft_get_plan_cache_size", TORCH_FN(TraceType::_cufft_get_plan_cache_size));
  m.impl_UNBOXED("_cummax_helper", &TraceType::_cummax_helper);
  m.impl_UNBOXED("_cumprod.out", &TraceType::_cumprod_out_out);
  m.impl("_cumsum", TORCH_FN(TraceType::_cumsum));
  m.impl("_dim_arange", TORCH_FN(TraceType::_dim_arange));
  m.impl("_dirichlet_grad", TORCH_FN(TraceType::_dirichlet_grad));
  m.impl_UNBOXED("_embedding_bag_backward", &TraceType::_embedding_bag_backward);
  m.impl_UNBOXED("_fused_dropout", &TraceType::_fused_dropout);
  m.impl("_logcumsumexp", TORCH_FN(TraceType::_logcumsumexp));
  m.impl("_make_per_tensor_quantized_tensor", TORCH_FN(TraceType::_make_per_tensor_quantized_tensor));
  m.impl("_mode", TORCH_FN(TraceType::_mode));
  m.impl_UNBOXED("_nnpack_spatial_convolution", &TraceType::_nnpack_spatial_convolution);
  m.impl("_softmax_backward_data", TORCH_FN(TraceType::_softmax_backward_data));
  m.impl_UNBOXED("_sparse_log_softmax.int", &TraceType::_sparse_log_softmax_int);
  m.impl_UNBOXED("_sparse_log_softmax.Dimname", &TraceType::_sparse_log_softmax_Dimname);
  m.impl("_sparse_log_softmax", TORCH_FN(TraceType::_sparse_log_softmax));
  m.impl("_sparse_sum_backward", TORCH_FN(TraceType::_sparse_sum_backward));
  m.impl_UNBOXED("_thnn_differentiable_gru_cell_backward", &TraceType::_thnn_differentiable_gru_cell_backward);
  m.impl("_thnn_fused_gru_cell_backward", TORCH_FN(TraceType::_thnn_fused_gru_cell_backward));
  m.impl("_use_cudnn_rnn_flatten_weight", TORCH_FN(TraceType::_use_cudnn_rnn_flatten_weight));
  m.impl("_values", TORCH_FN(TraceType::_values));
  m.impl("_weight_norm_cuda_interface_backward", TORCH_FN(TraceType::_weight_norm_cuda_interface_backward));
  m.impl_UNBOXED("acos.out", &TraceType::acos_out_out);
  m.impl("acosh", TORCH_FN(TraceType::acosh));
  m.impl_UNBOXED("acosh_", &TraceType::acosh_);
  m.impl("adaptive_avg_pool2d", TORCH_FN(TraceType::adaptive_avg_pool2d));
  m.impl_UNBOXED("adaptive_avg_pool3d.out", &TraceType::adaptive_avg_pool3d_out_out);
  m.impl("adaptive_max_pool3d_backward", TORCH_FN(TraceType::adaptive_max_pool3d_backward));
  m.impl_UNBOXED("add.out", &TraceType::add_out_out);
  m.impl("addr", TORCH_FN(TraceType::addr));
  m.impl_UNBOXED("addr_", &TraceType::addr_);
  m.impl("affine_grid_generator", TORCH_FN(TraceType::affine_grid_generator));
  m.impl_UNBOXED("arange.out", &TraceType::arange_out_out);
  m.impl_UNBOXED("arange.start_out", &TraceType::arange_out_start_out);
  m.impl_UNBOXED("asin.out", &TraceType::asin_out_out);
  m.impl("asinh", TORCH_FN(TraceType::asinh));
  m.impl_UNBOXED("asinh_", &TraceType::asinh_);
  m.impl("avg_pool2d_backward", TORCH_FN(TraceType::avg_pool2d_backward));
  m.impl("avg_pool3d", TORCH_FN(TraceType::avg_pool3d));
  m.impl_UNBOXED("avg_pool3d_backward.grad_input", &TraceType::avg_pool3d_backward_out_grad_input);
  m.impl("baddbmm", TORCH_FN(TraceType::baddbmm));
  m.impl_UNBOXED("baddbmm_", &TraceType::baddbmm_);
  m.impl_UNBOXED("batch_norm_backward_reduce", &TraceType::batch_norm_backward_reduce);
  m.impl_UNBOXED("bernoulli.out", &TraceType::bernoulli_out_out);
  m.impl_UNBOXED("binary_cross_entropy_with_logits", &TraceType::binary_cross_entropy_with_logits);
  m.impl_UNBOXED("bincount", &TraceType::bincount);
  m.impl("bitwise_and.Scalar", TORCH_FN(TraceType::bitwise_and_Scalar));
  m.impl("bitwise_and.Tensor", TORCH_FN(TraceType::bitwise_and_Tensor));
  m.impl_UNBOXED("bitwise_and_.Scalar", &TraceType::bitwise_and__Scalar);
  m.impl_UNBOXED("bitwise_and_.Tensor", &TraceType::bitwise_and__Tensor);
  m.impl("bitwise_not", TORCH_FN(TraceType::bitwise_not));
  m.impl_UNBOXED("bitwise_not_", &TraceType::bitwise_not_);
  m.impl("bmm", TORCH_FN(TraceType::bmm));
  m.impl("bucketize.Tensor", TORCH_FN(TraceType::bucketize_Tensor));
  m.impl("bucketize.Scalar", TORCH_FN(TraceType::bucketize_Scalar));
  m.impl("cartesian_prod", TORCH_FN(TraceType::cartesian_prod));
  m.impl_UNBOXED("cat.out", &TraceType::cat_out_out);
  m.impl_UNBOXED("cat.names_out", &TraceType::cat_out_names_out);
  m.impl("chain_matmul", TORCH_FN(TraceType::chain_matmul));
  m.impl("cholesky", TORCH_FN(TraceType::cholesky));
  m.impl("clamp_max", TORCH_FN(TraceType::clamp_max));
  m.impl_UNBOXED("clamp_max_", &TraceType::clamp_max_);
  m.impl("coalesce", TORCH_FN(TraceType::coalesce));
  m.impl_UNBOXED("col2im.out", &TraceType::col2im_out_out);
  m.impl("combinations", TORCH_FN(TraceType::combinations));
  m.impl("conj", TORCH_FN(TraceType::conj));
  m.impl("conv_tbc", TORCH_FN(TraceType::conv_tbc));
  m.impl_UNBOXED("convolution", &TraceType::convolution);
  m.impl("cosine_similarity", TORCH_FN(TraceType::cosine_similarity));
  m.impl("cudnn_convolution_backward_weight", TORCH_FN(TraceType::cudnn_convolution_backward_weight));
  m.impl("cummax", TORCH_FN(TraceType::cummax));
  m.impl_UNBOXED("cummax.dimname", &TraceType::cummax_dimname);
  m.impl_UNBOXED("cumprod.out", &TraceType::cumprod_out_out);
  m.impl_UNBOXED("cumprod.dimname_out", &TraceType::cumprod_out_dimname_out);
  m.impl_UNBOXED("cumsum", &TraceType::cumsum);
  m.impl_UNBOXED("cumsum.dimname", &TraceType::cumsum_dimname);
  m.impl("dense_dim", TORCH_FN(TraceType::dense_dim));
  m.impl("diagonal", TORCH_FN(TraceType::diagonal));
  m.impl_UNBOXED("diagonal.Dimname", &TraceType::diagonal_Dimname);
  m.impl("dist", TORCH_FN(TraceType::dist));
  m.impl_UNBOXED("dot.out", &TraceType::dot_out_out);
  m.impl("dropout", TORCH_FN(TraceType::dropout));
  m.impl_UNBOXED("dropout_", &TraceType::dropout_);
  m.impl("elu", TORCH_FN(TraceType::elu));
  m.impl_UNBOXED("elu_", &TraceType::elu_);
  m.impl_UNBOXED("elu_backward.grad_input", &TraceType::elu_backward_out_grad_input);
  m.impl_UNBOXED("embedding_renorm_", &TraceType::embedding_renorm_);
  m.impl("equal", TORCH_FN(TraceType::equal));
  m.impl_UNBOXED("erf.out", &TraceType::erf_out_out);
  m.impl("erfc", TORCH_FN(TraceType::erfc));
  m.impl_UNBOXED("erfc_", &TraceType::erfc_);
  m.impl("expm1", TORCH_FN(TraceType::expm1));
  m.impl_UNBOXED("expm1_", &TraceType::expm1_);
  m.impl_UNBOXED("exponential_", &TraceType::exponential_);
  m.impl("fake_quantize_per_channel_affine", TORCH_FN(TraceType::fake_quantize_per_channel_affine));
  m.impl("fbgemm_linear_fp16_weight_fp32_activation", TORCH_FN(TraceType::fbgemm_linear_fp16_weight_fp32_activation));
  m.impl("fbgemm_linear_int8_weight_fp32_activation", TORCH_FN(TraceType::fbgemm_linear_int8_weight_fp32_activation));
  m.impl("fbgemm_linear_quantize_weight", TORCH_FN(TraceType::fbgemm_linear_quantize_weight));
  m.impl("floor", TORCH_FN(TraceType::floor));
  m.impl_UNBOXED("floor_", &TraceType::floor_);
  m.impl_UNBOXED("fmod.Scalar_out", &TraceType::fmod_out_Scalar_out);
  m.impl_UNBOXED("fmod.Tensor_out", &TraceType::fmod_out_Tensor_out);
  m.impl_UNBOXED("frac.out", &TraceType::frac_out_out);
  m.impl_UNBOXED("fractional_max_pool2d.output", &TraceType::fractional_max_pool2d_out_output);
  m.impl_UNBOXED("frobenius_norm.out", &TraceType::frobenius_norm_out_out);
  m.impl_UNBOXED("full_like", &TraceType::full_like);
  m.impl_UNBOXED("group_norm", &TraceType::group_norm);
  m.impl_UNBOXED("hamming_window", &TraceType::hamming_window);
  m.impl_UNBOXED("hamming_window.periodic", &TraceType::hamming_window_periodic);
  m.impl_UNBOXED("hamming_window.periodic_alpha", &TraceType::hamming_window_periodic_alpha);
  m.impl_UNBOXED("hamming_window.periodic_alpha_beta", &TraceType::hamming_window_periodic_alpha_beta);
  m.impl("hardshrink_backward", TORCH_FN(TraceType::hardshrink_backward));
  m.impl_UNBOXED("hardtanh.out", &TraceType::hardtanh_out_out);
  m.impl_UNBOXED("im2col.out", &TraceType::im2col_out_out);
  m.impl_UNBOXED("index.Tensor", &TraceType::index_Tensor);
  m.impl_UNBOXED("index_put", &TraceType::index_put);
  m.impl_UNBOXED("index_put_", &TraceType::index_put_);
  m.impl("index_select", TORCH_FN(TraceType::index_select));
  m.impl_UNBOXED("index_select.dimname", &TraceType::index_select_dimname);
  m.impl("is_coalesced", TORCH_FN(TraceType::is_coalesced));
  m.impl("is_floating_point", TORCH_FN(TraceType::is_floating_point));
  m.impl("is_vulkan_available", TORCH_FN(TraceType::is_vulkan_available));
  m.impl("item", TORCH_FN(TraceType::item));
  m.impl("l1_loss", TORCH_FN(TraceType::l1_loss));
  m.impl_UNBOXED("l1_loss_backward.grad_input", &TraceType::l1_loss_backward_out_grad_input);
  m.impl_UNBOXED("linspace.out", &TraceType::linspace_out_out);
  m.impl_UNBOXED("log2.out", &TraceType::log2_out_out);
  m.impl_UNBOXED("log_normal_", &TraceType::log_normal_);
  m.impl_UNBOXED("log.out", &TraceType::log_out_out);
  m.impl("log_sigmoid_backward", TORCH_FN(TraceType::log_sigmoid_backward));
  m.impl_UNBOXED("log_sigmoid_forward.output", &TraceType::log_sigmoid_forward_out_output);
  m.impl_UNBOXED("logaddexp2.out", &TraceType::logaddexp2_out_out);
  m.impl_UNBOXED("logaddexp.out", &TraceType::logaddexp_out_out);
  m.impl_UNBOXED("logcumsumexp", &TraceType::logcumsumexp);
  m.impl_UNBOXED("logcumsumexp.dimname", &TraceType::logcumsumexp_dimname);
  m.impl("logical_or", TORCH_FN(TraceType::logical_or));
  m.impl_UNBOXED("logical_or_", &TraceType::logical_or_);
  m.impl("logical_xor", TORCH_FN(TraceType::logical_xor));
  m.impl_UNBOXED("logical_xor_", &TraceType::logical_xor_);
  m.impl_UNBOXED("logspace", &TraceType::logspace);
  m.impl("logsumexp", TORCH_FN(TraceType::logsumexp));
  m.impl_UNBOXED("logsumexp.names", &TraceType::logsumexp_names);
  m.impl_UNBOXED("lstsq.X", &TraceType::lstsq_out_X);
  m.impl("matmul", TORCH_FN(TraceType::matmul));
  m.impl("max.dim", TORCH_FN(TraceType::max_dim));
  m.impl_UNBOXED("max.names_dim", &TraceType::max_names_dim);
  m.impl("max.other", TORCH_FN(TraceType::max_other));
  m.impl("max", TORCH_FN(TraceType::max));
  m.impl("max_pool1d_with_indices", TORCH_FN(TraceType::max_pool1d_with_indices));
  m.impl_UNBOXED("max_pool2d_with_indices.out", &TraceType::max_pool2d_with_indices_out_out);
  m.impl("max_unpool2d_backward", TORCH_FN(TraceType::max_unpool2d_backward));
  m.impl("max_unpool3d", TORCH_FN(TraceType::max_unpool3d));
  m.impl_UNBOXED("max_unpool3d_backward.grad_input", &TraceType::max_unpool3d_backward_out_grad_input);
  m.impl_UNBOXED("mean.out", &TraceType::mean_out_out);
  m.impl_UNBOXED("mean.names_out", &TraceType::mean_out_names_out);
  m.impl_UNBOXED("median.dim_values", &TraceType::median_out_dim_values);
  m.impl_UNBOXED("median.names_dim_values", &TraceType::median_out_names_dim_values);
  m.impl("meshgrid", TORCH_FN(TraceType::meshgrid));
  m.impl_UNBOXED("miopen_batch_norm", &TraceType::miopen_batch_norm);
  m.impl("miopen_convolution_transpose_backward", TORCH_FN(TraceType::miopen_convolution_transpose_backward));
  m.impl("miopen_convolution_transpose_backward_input", TORCH_FN(TraceType::miopen_convolution_transpose_backward_input));
  m.impl("miopen_depthwise_convolution_backward_weight", TORCH_FN(TraceType::miopen_depthwise_convolution_backward_weight));
  m.impl("mkldnn_convolution_backward", TORCH_FN(TraceType::mkldnn_convolution_backward));
  m.impl("mkldnn_convolution_backward_input", TORCH_FN(TraceType::mkldnn_convolution_backward_input));
  m.impl("mode", TORCH_FN(TraceType::mode));
  m.impl_UNBOXED("mode.dimname", &TraceType::mode_dimname);
  m.impl_UNBOXED("multi_margin_loss.out", &TraceType::multi_margin_loss_out_out);
  m.impl("multilabel_margin_loss_forward", TORCH_FN(TraceType::multilabel_margin_loss_forward));
  m.impl_UNBOXED("mv.out", &TraceType::mv_out_out);
  m.impl_UNBOXED("native_batch_norm_backward", &TraceType::native_batch_norm_backward);
  m.impl("native_norm", TORCH_FN(TraceType::native_norm));
  m.impl("ne.Scalar", TORCH_FN(TraceType::ne_Scalar));
  m.impl("ne.Tensor", TORCH_FN(TraceType::ne_Tensor));
  m.impl_UNBOXED("ne_.Scalar", &TraceType::ne__Scalar);
  m.impl_UNBOXED("ne_.Tensor", &TraceType::ne__Tensor);
  m.impl_UNBOXED("nll_loss2d_backward", &TraceType::nll_loss2d_backward);
  m.impl_UNBOXED("nll_loss2d_forward.output", &TraceType::nll_loss2d_forward_out_output);
  m.impl_UNBOXED("nll_loss_backward", &TraceType::nll_loss_backward);
  m.impl_UNBOXED("nll_loss_forward.output", &TraceType::nll_loss_forward_out_output);
  m.impl_UNBOXED("ones.out", &TraceType::ones_out_out);
  m.impl("ormqr", TORCH_FN(TraceType::ormqr));
  m.impl("pairwise_distance", TORCH_FN(TraceType::pairwise_distance));
  m.impl("pinverse", TORCH_FN(TraceType::pinverse));
  m.impl_UNBOXED("polygamma.out", &TraceType::polygamma_out_out);
  m.impl_UNBOXED("pow.Tensor_Scalar_out", &TraceType::pow_out_Tensor_Scalar_out);
  m.impl_UNBOXED("pow.Tensor_Tensor_out", &TraceType::pow_out_Tensor_Tensor_out);
  m.impl_UNBOXED("pow.Scalar_out", &TraceType::pow_out_Scalar_out);
  m.impl_UNBOXED("prod.int_out", &TraceType::prod_out_int_out);
  m.impl_UNBOXED("prod.Dimname_out", &TraceType::prod_out_Dimname_out);
  m.impl("q_per_channel_axis", TORCH_FN(TraceType::q_per_channel_axis));
  m.impl("q_per_channel_zero_points", TORCH_FN(TraceType::q_per_channel_zero_points));
  m.impl_UNBOXED("qr.Q", &TraceType::qr_out_Q);
  m.impl("quantized_gru_cell", TORCH_FN(TraceType::quantized_gru_cell));
  m.impl("quantized_rnn_relu_cell", TORCH_FN(TraceType::quantized_rnn_relu_cell));
  m.impl_UNBOXED("rand.names", &TraceType::rand_names);
  m.impl_UNBOXED("rand.generator_with_names", &TraceType::rand_generator_with_names);
  m.impl_UNBOXED("rand", &TraceType::rand);
  m.impl_UNBOXED("rand.generator", &TraceType::rand_generator);
  m.impl_UNBOXED("randint.out", &TraceType::randint_out_out);
  m.impl_UNBOXED("randint.generator_out", &TraceType::randint_out_generator_out);
  m.impl_UNBOXED("randint.low_out", &TraceType::randint_out_low_out);
  m.impl_UNBOXED("randint.low_generator_out", &TraceType::randint_out_low_generator_out);
  m.impl_UNBOXED("randn", &TraceType::randn);
  m.impl_UNBOXED("randn.generator", &TraceType::randn_generator);
  m.impl_UNBOXED("randn.names", &TraceType::randn_names);
  m.impl_UNBOXED("randn.generator_with_names", &TraceType::randn_generator_with_names);
  m.impl_UNBOXED("random_.from", &TraceType::random__from);
  m.impl_UNBOXED("random_.to", &TraceType::random__to);
  m.impl_UNBOXED("random_", &TraceType::random_);
  m.impl_UNBOXED("randperm.out", &TraceType::randperm_out_out);
  m.impl_UNBOXED("randperm.generator_out", &TraceType::randperm_out_generator_out);
  m.impl("reflection_pad1d_backward", TORCH_FN(TraceType::reflection_pad1d_backward));
  m.impl("reflection_pad2d", TORCH_FN(TraceType::reflection_pad2d));
  m.impl_UNBOXED("reflection_pad2d_backward.grad_input", &TraceType::reflection_pad2d_backward_out_grad_input);
  m.impl("remainder.Scalar", TORCH_FN(TraceType::remainder_Scalar));
  m.impl("remainder.Tensor", TORCH_FN(TraceType::remainder_Tensor));
  m.impl_UNBOXED("remainder_.Scalar", &TraceType::remainder__Scalar);
  m.impl_UNBOXED("remainder_.Tensor", &TraceType::remainder__Tensor);
  m.impl("repeat", TORCH_FN(TraceType::repeat));
  m.impl("replication_pad1d", TORCH_FN(TraceType::replication_pad1d));
  m.impl_UNBOXED("replication_pad1d_backward.grad_input", &TraceType::replication_pad1d_backward_out_grad_input);
  m.impl_UNBOXED("replication_pad2d.out", &TraceType::replication_pad2d_out_out);
  m.impl("reshape_as", TORCH_FN(TraceType::reshape_as));
  m.impl_UNBOXED("result_type.Tensor", &TraceType::result_type_Tensor);
  m.impl_UNBOXED("result_type.Scalar", &TraceType::result_type_Scalar);
  m.impl_UNBOXED("result_type.Scalar_Tensor", &TraceType::result_type_Scalar_Tensor);
  m.impl_UNBOXED("result_type.Scalar_Scalar", &TraceType::result_type_Scalar_Scalar);
  m.impl("rnn_tanh.input", TORCH_FN(TraceType::rnn_tanh_input));
  m.impl("rnn_tanh.data", TORCH_FN(TraceType::rnn_tanh_data));
  m.impl("roll", TORCH_FN(TraceType::roll));
  m.impl("rot90", TORCH_FN(TraceType::rot90));
  m.impl_UNBOXED("round.out", &TraceType::round_out_out);
  m.impl("rrelu_with_noise_backward", TORCH_FN(TraceType::rrelu_with_noise_backward));
  m.impl_UNBOXED("rsqrt.out", &TraceType::rsqrt_out_out);
  m.impl_UNBOXED("searchsorted.Tensor_out", &TraceType::searchsorted_out_Tensor_out);
  m.impl("selu", TORCH_FN(TraceType::selu));
  m.impl_UNBOXED("selu_", &TraceType::selu_);
  m.impl_UNBOXED("slow_conv3d.out", &TraceType::slow_conv3d_out_out);
  m.impl_UNBOXED("slow_conv_transpose2d.out", &TraceType::slow_conv_transpose2d_out_out);
  m.impl_UNBOXED("smooth_l1_loss.out", &TraceType::smooth_l1_loss_out_out);
  m.impl("soft_margin_loss", TORCH_FN(TraceType::soft_margin_loss));
  m.impl_UNBOXED("soft_margin_loss_backward.grad_input", &TraceType::soft_margin_loss_backward_out_grad_input);
  m.impl("softplus", TORCH_FN(TraceType::softplus));
  m.impl_UNBOXED("softplus_backward.grad_input", &TraceType::softplus_backward_out_grad_input);
  m.impl("sort", TORCH_FN(TraceType::sort));
  m.impl_UNBOXED("sort.dimname", &TraceType::sort_dimname);
  m.impl("split.Tensor", TORCH_FN(TraceType::split_Tensor));
  m.impl_UNBOXED("sspaddmm.out", &TraceType::sspaddmm_out_out);
  m.impl("std", TORCH_FN(TraceType::std));
  m.impl("std.dim", TORCH_FN(TraceType::std_dim));
  m.impl_UNBOXED("std.names_dim", &TraceType::std_names_dim);
  m.impl("stride.int", TORCH_FN(TraceType::stride_int));
  m.impl_UNBOXED("stride.Dimname", &TraceType::stride_Dimname);
  m.impl_UNBOXED("sum", &TraceType::sum);
  m.impl_UNBOXED("sum.dim_IntList", &TraceType::sum_dim_IntList);
  m.impl_UNBOXED("sum.dim_DimnameList", &TraceType::sum_dim_DimnameList);
  m.impl("sum_to_size", TORCH_FN(TraceType::sum_to_size));
  m.impl("t", TORCH_FN(TraceType::t));
  m.impl_UNBOXED("t_", &TraceType::t_);
  m.impl("take", TORCH_FN(TraceType::take));
  m.impl_UNBOXED("tanh.out", &TraceType::tanh_out_out);
  m.impl_UNBOXED("thnn_conv_depthwise2d", &TraceType::thnn_conv_depthwise2d);
  m.impl_UNBOXED("thnn_conv_depthwise2d_backward.grad_input", &TraceType::thnn_conv_depthwise2d_backward_out_grad_input);
  m.impl("to_mkldnn", TORCH_FN(TraceType::to_mkldnn));
  m.impl("to_sparse.sparse_dim", TORCH_FN(TraceType::to_sparse_sparse_dim));
  m.impl("to_sparse", TORCH_FN(TraceType::to_sparse));
  m.impl("topk", TORCH_FN(TraceType::topk));
  m.impl("transpose.int", TORCH_FN(TraceType::transpose_int));
  m.impl_UNBOXED("transpose.Dimname", &TraceType::transpose_Dimname);
  m.impl_UNBOXED("transpose_", &TraceType::transpose_);
  m.impl("trapz.x", TORCH_FN(TraceType::trapz_x));
  m.impl("trapz.dx", TORCH_FN(TraceType::trapz_dx));
  m.impl_UNBOXED("triu.out", &TraceType::triu_out_out);
  m.impl_UNBOXED("true_divide.out", &TraceType::true_divide_out_out);
  m.impl("trunc", TORCH_FN(TraceType::trunc));
  m.impl_UNBOXED("trunc_", &TraceType::trunc_);
  m.impl_UNBOXED("upsample_bicubic2d.out", &TraceType::upsample_bicubic2d_out_out);
  m.impl("upsample_linear1d_backward", TORCH_FN(TraceType::upsample_linear1d_backward));
  m.impl("upsample_nearest2d_backward", TORCH_FN(TraceType::upsample_nearest2d_backward));
  m.impl("upsample_nearest3d", TORCH_FN(TraceType::upsample_nearest3d));
  m.impl_UNBOXED("upsample_nearest3d_backward.grad_input", &TraceType::upsample_nearest3d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_trilinear3d.out", &TraceType::upsample_trilinear3d_out_out);
  m.impl("values", TORCH_FN(TraceType::values));
  m.impl("var_mean", TORCH_FN(TraceType::var_mean));
  m.impl("var_mean.dim", TORCH_FN(TraceType::var_mean_dim));
  m.impl_UNBOXED("var_mean.names_dim", &TraceType::var_mean_names_dim);;
}

}  // namespace

} // namespace torch
