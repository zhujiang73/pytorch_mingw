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
Tensor & __ilshift___Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ilshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ilshift__", "Scalar")
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
Tensor & __ilshift___Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ilshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ilshift__", "Tensor")
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
Tensor & __ior___Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ior__", "Scalar")
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
Tensor & __ior___Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ior__", "Tensor")
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
Tensor & __ixor___Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ixor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ixor__", "Scalar")
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
Tensor & __ixor___Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__ixor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ixor__", "Tensor")
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
Tensor __lshift___Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__lshift__", "Scalar")
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
Tensor __lshift___Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__lshift__", "Tensor")
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
Tensor __or___Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__or__", "Scalar")
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
Tensor __or___Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__or__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__or__", "Tensor")
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
Tensor __xor___Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__xor__", "Scalar")
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
Tensor __xor___Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__xor__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__xor__", "Tensor")
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
Tensor & _addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_addr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, vec1, vec2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & _baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_baddbmm_mkl");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_baddbmm_mkl_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_baddbmm_mkl_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_baddbmm_mkl_", "")
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
Tensor & _bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2, bool deterministic) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_bmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_bmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, out, self, mat2, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor _cast_Double(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Double");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Double", "")
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
Tensor _cast_Short(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Short");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Short", "")
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
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_rnn");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "weight_buf", weight_buf);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
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
      .findSchemaOrThrow("aten::_cudnn_rnn", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &)>();
  std::tie(result0, result1, result2, result3, result4) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Tracer, input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
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
Tensor & _cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumsum", "out")
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
int64_t _dimV(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dimV", "")
      .typed<int64_t (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool)>();
  std::tie(result0, result1, result2, result3) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool>(op, c10::DispatchKey::Tracer, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
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
Tensor _embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_embedding_bag_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_sparse_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_gather_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_gather_sparse_backward", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_index_put_impl");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_index_put_impl_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices, true);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    jit::tracer::addInputs(node, "unsafe", unsafe);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_index_put_impl_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_index_put_impl_", "")
      .typed<Tensor & (Tensor &, TensorList, const Tensor &, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, indices, values, accumulate, unsafe);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _indices(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_indices", "")
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
Scalar _local_scalar_dense(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_local_scalar_dense", "")
      .typed<Scalar (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Scalar, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
Tensor & _logcumsumexp_out_out(Tensor & out, const Tensor & self, int64_t dim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_logcumsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_logcumsumexp", "out")
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
Tensor _mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose");
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
      .findSchemaOrThrow("aten::_mkldnn_transpose", "")
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
Tensor & _mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_mkldnn_transpose_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mkldnn_transpose_", "")
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
std::tuple<Tensor &,Tensor &> _mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_mode");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mode", "values")
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
Tensor _nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution_backward_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weightsize", weightsize);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution_backward_weight", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, input, weightsize, grad_output, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _pdist_forward(const Tensor & self, double p) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pdist_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pdist_forward", "")
      .typed<Tensor (const Tensor &, double)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Tracer, self, p);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_softmax");
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
      .findSchemaOrThrow("aten::_softmax", "")
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
Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims", "")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Tracer, sparse_dim, dense_dim, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_softmax_backward_data");
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
      .findSchemaOrThrow("aten::_sparse_softmax_backward_data", "")
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
Tensor _sparse_sum(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "")
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
Tensor _sparse_sum_dtype(const Tensor & self, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "dtype")
      .typed<Tensor (const Tensor &, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, ScalarType>(op, c10::DispatchKey::Tracer, self, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_sum_dim(const Tensor & self, IntArrayRef dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_sum_dim_dtype(const Tensor & self, IntArrayRef dim, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_sum");
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
      .findSchemaOrThrow("aten::_sparse_sum", "dim_dtype")
      .typed<Tensor (const Tensor &, IntArrayRef, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, ScalarType>(op, c10::DispatchKey::Tracer, self, dim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & input_bias, const Tensor & hidden_bias, const Tensor & cx, const Tensor & cy) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_differentiable_lstm_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "cy", cy);
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
      .findSchemaOrThrow("aten::_thnn_differentiable_lstm_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  std::tie(result0, result1, result2, result3, result4) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
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
std::tuple<Tensor,Tensor> _thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_gru_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
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
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_gru_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input_gates, hidden_gates, hx, input_bias, hidden_bias);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_lstm_cell_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "cy", cy);
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
      .findSchemaOrThrow("aten::_thnn_fused_lstm_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool)>();
  std::tie(result0, result1, result2, result3, result4) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, grad_hy, grad_cy, cx, cy, workspace, has_bias);
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
Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_trilinear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "i1", i1);
    jit::tracer::addInputs(node, "i2", i2);
    jit::tracer::addInputs(node, "i3", i3);
    jit::tracer::addInputs(node, "expand1", expand1);
    jit::tracer::addInputs(node, "expand2", expand2);
    jit::tracer::addInputs(node, "expand3", expand3);
    jit::tracer::addInputs(node, "sumdim", sumdim);
    jit::tracer::addInputs(node, "unroll_dim", unroll_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_trilinear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Tracer, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _unsafe_view(const Tensor & self, IntArrayRef size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_unsafe_view");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_unsafe_view", "")
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
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm_cuda_interface");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_cuda_interface", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, v, g, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor abs(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::abs");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs", "")
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
Tensor & abs_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::abs");
    } else {
      op_name = jit::Symbol::fromQualString("aten::abs_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs_", "")
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
Tensor & acosh_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acosh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acosh", "out")
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
Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool1d", "")
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
Tensor & adaptive_avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool2d", "out")
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
Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
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
      .findSchemaOrThrow("aten::adaptive_max_pool2d_backward", "")
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
std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & adaptive_max_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, indices);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcmul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, tensor1, tensor2, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addcmul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addcmul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcmul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, tensor1, tensor2, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, mat1, mat2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, mat1, mat2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, vec1, vec2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor align_as(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_as", "")
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
std::vector<Tensor> align_tensors(TensorList tensors) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_tensors", "")
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
Tensor align_to(const Tensor & self, DimnameList names) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_to", "")
      .typed<Tensor (const Tensor &, DimnameList)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList>(op, c10::DispatchKey::Tracer, self, names);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor align_to_ellipsis_idx(const Tensor & self, DimnameList order, int64_t ellipsis_idx) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::align_to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "order", order);
    jit::tracer::addInputs(node, "ellipsis_idx", ellipsis_idx);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_to", "ellipsis_idx")
      .typed<Tensor (const Tensor &, DimnameList, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, int64_t>(op, c10::DispatchKey::Tracer, self, order, ellipsis_idx);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argmax");
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
      .findSchemaOrThrow("aten::argmax", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Tracer, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argsort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argsort", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor argsort_dimname(const Tensor & self, Dimname dim, bool descending) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::argsort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argsort", "dimname")
      .typed<Tensor (const Tensor &, Dimname, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & asinh_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asinh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asinh", "out")
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
Tensor atan(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan", "")
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
Tensor atan2(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atan2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2", "")
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
Tensor & atan2_(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::atan2");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atan2_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2_", "")
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
Tensor & atan_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::atan");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atan_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan_", "")
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
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool2d");
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
      .findSchemaOrThrow("aten::avg_pool2d", "")
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
Tensor & avg_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d_backward", "grad_input")
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
Tensor & avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & baddbmm_out_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::baddbmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, batch1, batch2, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor bartlett_window(int64_t window_length, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bartlett_window", "")
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
Tensor bartlett_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bartlett_window");
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
      .findSchemaOrThrow("aten::bartlett_window", "periodic")
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
std::tuple<Tensor,Tensor> batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_update_stats");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_update_stats", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, double)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Tracer, input, running_mean, running_var, momentum);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_output, self, target, weight, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & bitwise_and_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Tensor_out")
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
Tensor & bitwise_and_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Scalar_out")
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
Tensor & bitwise_not_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not", "out")
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
Tensor & bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, mat2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & bucketize_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bucketize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, out, self, boundaries, out_int32, right);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor cdist(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cdist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cdist", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, x1, x2, p, compute_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor celu(const Tensor & self, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::celu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::celu", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & celu_(Tensor & self, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::celu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::celu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("celu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::celu_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor cholesky_inverse(const Tensor & self, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky_inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_inverse", "")
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
Tensor & cholesky_out_out(Tensor & out, const Tensor & self, bool upper) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, out, self, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp", "")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, c10::DispatchKey::Tracer, self, min, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_", "")
      .typed<Tensor & (Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, c10::DispatchKey::Tracer, self, min, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & clamp_max_out_out(Tensor & out, const Tensor & self, Scalar max) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, self, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & conj_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("conj_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conj", "out")
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
Tensor conv_transpose3d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose3d", "input")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::convolution_backward_overrideable");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution_backward_overrideable", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>)>();
  std::tie(grad_input, grad_weight, grad_bias) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Tracer, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor cos(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cos", "")
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
Tensor & cos_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cos");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cos_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cos_", "")
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
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon, const Tensor & reserveSpace) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_batch_norm_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    jit::tracer::addInputs(node, "reserveSpace", reserveSpace);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Tracer, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
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
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_grid_sampler_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor grad_self;
  Tensor grad_grid;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_grid_sampler_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>();
  std::tie(grad_self, grad_grid) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, grid, grad_output);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_self);
    jit::tracer::addOutput(node, grad_grid);
  }
  #endif
  return std::make_tuple(std::move(grad_self), std::move(grad_grid));
}
std::tuple<Tensor &,Tensor &> cummax_out_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummax_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, values, indices, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> cummax_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummax_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "dimname_out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname>(op, c10::DispatchKey::Tracer, values, indices, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor & cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "out")
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
Tensor & cumsum_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "dimname_out")
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
Tensor deg2rad(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::deg2rad");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad", "")
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
Tensor & deg2rad_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::deg2rad");
    } else {
      op_name = jit::Symbol::fromQualString("aten::deg2rad_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("deg2rad_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad_", "")
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
Tensor diag(const Tensor & self, int64_t diagonal) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diag", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, diagonal);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor digamma(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::digamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma", "")
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
Tensor & digamma_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::digamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::digamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma_", "")
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
Tensor & elu_out_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, alpha, scale, input_scale);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  Tensor result3;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_bag", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool)>();
  std::tie(result0, result1, result2, result3) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool>(op, c10::DispatchKey::Tracer, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
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
Tensor embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_dense_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Tracer, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Tracer, self, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor empty_quantized(IntArrayRef size, const Tensor & qtensor) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_quantized");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "qtensor", qtensor);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_quantized", "")
      .typed<Tensor (IntArrayRef, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Tracer, size, qtensor);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty_strided");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_strided", "")
      .typed<Tensor (IntArrayRef, IntArrayRef, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, stride, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & erfc_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfc", "out")
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
Tensor erfinv(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erfinv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv", "")
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
Tensor & erfinv_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::erfinv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erfinv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv_", "")
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
Tensor expand(const Tensor & self, IntArrayRef size, bool implicit) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expand");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "implicit", implicit);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expand", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, self, size, implicit);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & expm1_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expm1_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1", "out")
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
Tensor fake_quantize_per_tensor_affine_backward(const Tensor & grad, const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_tensor_affine_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Tracer, grad, self, scale, zero_point, quant_min, quant_max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fft", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, signal_ndim, normalized);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_using_ints(const Tensor & self, int64_t start_dim, int64_t end_dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "using_ints")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, start_dim, end_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_named_out_dim(const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "named_out_dim")
      .typed<Tensor (const Tensor &, int64_t, int64_t, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, Dimname>(op, c10::DispatchKey::Tracer, self, start_dim, end_dim, out_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_using_names(const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "start_dim", start_dim);
    jit::tracer::addInputs(node, "end_dim", end_dim);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "using_names")
      .typed<Tensor (const Tensor &, Dimname, Dimname, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, Dimname, Dimname>(op, c10::DispatchKey::Tracer, self, start_dim, end_dim, out_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor flatten_DimnameList(const Tensor & self, DimnameList dims, Dimname out_dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flatten");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "DimnameList")
      .typed<Tensor (const Tensor &, DimnameList, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, Dimname>(op, c10::DispatchKey::Tracer, self, dims, out_dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor floor_divide(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "")
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
Tensor floor_divide_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "Scalar")
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
Tensor & floor_divide__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::floor_divide");
    } else {
      op_name = jit::Symbol::fromQualString("aten::floor_divide_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide_", "Tensor")
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
Tensor & floor_divide__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::floor_divide");
    } else {
      op_name = jit::Symbol::fromQualString("aten::floor_divide_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide_", "Scalar")
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
Tensor & floor_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor", "out")
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
Tensor full_names(IntArrayRef size, Scalar fill_value, c10::optional<DimnameList> names, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "names")
      .typed<Tensor (IntArrayRef, Scalar, c10::optional<DimnameList>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, Scalar, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, fill_value, names, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "")
      .typed<Tensor (IntArrayRef, Scalar, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, Scalar, const TensorOptions &>(op, c10::DispatchKey::Tracer, size, fill_value, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, dim, index, sparse_grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor gather_dimname(const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gather");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "sparse_grad", sparse_grad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, dim, index, sparse_grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor gelu_backward(const Tensor & grad, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gelu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gelu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_3d_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Tracer, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gru_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor hann_window(int64_t window_length, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hann_window", "")
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
Tensor hann_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hann_window");
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
      .findSchemaOrThrow("aten::hann_window", "periodic")
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
Tensor hardshrink(const Tensor & self, Scalar lambd) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardshrink", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, lambd);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ifft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ifft", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, self, signal_ndim, normalized);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_select_out_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & index_select_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor indices(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::indices", "")
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
bool is_complex(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_complex", "")
      .typed<bool (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
bool is_same_size(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_same_size", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, other);
  return result;
}
Tensor & l1_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("l1_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss", "out")
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
Tensor layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::layer_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enable", cudnn_enable);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::layer_norm", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool>(op, c10::DispatchKey::Tracer, input, normalized_shape, weight, bias, eps, cudnn_enable);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope, bool self_is_result) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, bool>(op, c10::DispatchKey::Tracer, grad_output, self, negative_slope, self_is_result);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor lerp_Scalar(const Tensor & self, const Tensor & end, Scalar weight) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Scalar")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, end, weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor lerp_Tensor(const Tensor & self, const Tensor & end, const Tensor & weight) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lerp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, end, weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & lerp__Scalar(Tensor & self, const Tensor & end, Scalar weight) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp_", "Scalar")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, end, weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & lerp__Tensor(Tensor & self, const Tensor & end, const Tensor & weight) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lerp");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lerp_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, end, weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input, weight, bias);
  return result;
}
Tensor log_sigmoid(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid", "")
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
Tensor & log_sigmoid_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, buffer);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & logcumsumexp_out_out(Tensor & out, const Tensor & self, int64_t dim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logcumsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "out")
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
Tensor & logcumsumexp_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logcumsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname>(op, c10::DispatchKey::Tracer, out, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & logical_or_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_or_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or", "out")
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
Tensor & logical_xor_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor", "out")
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
Tensor & logspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logspace", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, int64_t, double)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, int64_t, double>(op, c10::DispatchKey::Tracer, out, start, end, steps, base);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & logsumexp_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "out")
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
Tensor & logsumexp_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool>(op, c10::DispatchKey::Tracer, out, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & matmul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matmul", "out")
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
std::tuple<Tensor &,Tensor &> max_out_dim_max(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_values", max_values);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "dim_max")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, max, max_values, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  #endif
  return std::forward_as_tuple(max, max_values);
}
std::tuple<Tensor &,Tensor &> max_out_names_dim_max(Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "max_values", max_values);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "max", max);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "names_dim_max")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, max, max_values, self, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  #endif
  return std::forward_as_tuple(max, max_values);
}
Tensor & max_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "out")
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
Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_unpool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, indices, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & max_unpool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, indices, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & max_unpool3d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, indices, output_size, stride, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor min_values(const Tensor & self, IntArrayRef dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min_values");
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
      .findSchemaOrThrow("aten::min_values", "")
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
Tensor min_values_names(const Tensor & self, DimnameList dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min_values");
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
      .findSchemaOrThrow("aten::min_values", "names")
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
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
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
      .findSchemaOrThrow("aten::miopen_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Tracer, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
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
Tensor miopen_convolution_backward_bias(const Tensor & grad_output) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_bias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_backward_bias", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor miopen_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self_size", self_size);
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
      .findSchemaOrThrow("aten::miopen_convolution_backward_input", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
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
      .findSchemaOrThrow("aten::miopen_convolution_transpose", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_adaptive_avg_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_adaptive_avg_pool2d", "")
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
Tensor mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Tracer, self, weight, bias, padding, stride, dilation, groups);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_reorder_conv2d_weight");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_reorder_conv2d_weight", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Tracer, self, padding, stride, dilation, groups);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "values")
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
std::tuple<Tensor &,Tensor &> mode_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mode");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "dimname_out")
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
Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_output, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, target, reduction, is_target);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out_output(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "is_target", is_target);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, output, is_target, self, target, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  #endif
  return std::forward_as_tuple(output, is_target);
}
Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multinomial");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "replacement", replacement);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multinomial", "")
      .typed<Tensor (const Tensor &, int64_t, bool, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, num_samples, replacement, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor mvlgamma(const Tensor & self, int64_t p) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mvlgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mvlgamma", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, p);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & mvlgamma_(Tensor & self, int64_t p) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mvlgamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mvlgamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mvlgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mvlgamma_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, p);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::narrow", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, dim, start, length);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor narrow_Tensor(const Tensor & self, int64_t dim, const Tensor & start, int64_t length) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::narrow");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::narrow", "Tensor")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim, start, length);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Tracer, input, weight, bias, running_mean, running_var, training, momentum, eps);
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
Tensor & ne_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Scalar_out")
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
Tensor & ne_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Tensor_out")
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
Tensor new_full(const Tensor & self, IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::new_full");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_full", "")
      .typed<Tensor (const Tensor &, IntArrayRef, Scalar, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, Scalar, const TensorOptions &>(op, c10::DispatchKey::Tracer, self, size, fill_value, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & nll_loss2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & nll_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor nuclear_norm(const Tensor & self, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor nuclear_norm_dim(const Tensor & self, IntArrayRef dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nuclear_norm");
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
      .findSchemaOrThrow("aten::nuclear_norm", "dim")
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
Tensor orgqr(const Tensor & self, const Tensor & input2) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::orgqr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::orgqr", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, input2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & ormqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ormqr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ormqr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, out, self, input2, input3, left, transpose);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor permute(const Tensor & self, IntArrayRef dims) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::permute");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::permute", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, dims);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor pixel_shuffle(const Tensor & self, int64_t upscale_factor) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pixel_shuffle");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upscale_factor", upscale_factor);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pixel_shuffle", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, upscale_factor);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::put");
    } else {
      op_name = jit::Symbol::fromQualString("aten::put_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::put_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, index, source, accumulate);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
int64_t q_zero_point(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_zero_point", "")
      .typed<int64_t (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
Tensor quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantize_per_tensor", "")
      .typed<Tensor (const Tensor &, double, int64_t, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t, ScalarType>(op, c10::DispatchKey::Tracer, self, scale, zero_point, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::vector<Tensor> quantize_per_tensor_tensors(TensorList tensors, const Tensor & scales, const Tensor & zero_points, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantize_per_tensor", "tensors")
      .typed<std::vector<Tensor> (TensorList, const Tensor &, const Tensor &, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList, const Tensor &, const Tensor &, ScalarType>(op, c10::DispatchKey::Tracer, tensors, scales, zero_points, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_lstm_cell");
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
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Tracer, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor rad2deg(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rad2deg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg", "")
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
Tensor & rad2deg_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rad2deg");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rad2deg_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rad2deg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg_", "")
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
Tensor & rand_out_out(Tensor & out, IntArrayRef size) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "out")
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
Tensor & rand_out_generator_out(Tensor & out, IntArrayRef size, c10::optional<Generator> generator) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "generator_out")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, size, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & randn_out_out(Tensor & out, IntArrayRef size) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "out")
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
Tensor & randn_out_generator_out(Tensor & out, IntArrayRef size, c10::optional<Generator> generator) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "generator_out")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, size, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor range_step(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "step")
      .typed<Tensor (Scalar, Scalar, Scalar, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, Scalar, const TensorOptions &>(op, c10::DispatchKey::Tracer, start, end, step, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor range(Scalar start, Scalar end, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::range");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "")
      .typed<Tensor (Scalar, Scalar, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, const TensorOptions &>(op, c10::DispatchKey::Tracer, start, end, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor real(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::real");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::real", "")
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
Tensor reciprocal(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reciprocal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal", "")
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
Tensor & reciprocal_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::reciprocal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::reciprocal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal_", "")
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
Tensor refine_names(const Tensor & self, DimnameList names) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::refine_names");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::refine_names", "")
      .typed<Tensor (const Tensor &, DimnameList)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList>(op, c10::DispatchKey::Tracer, self, names);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor reflection_pad1d(const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reflection_pad1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d", "")
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
Tensor & reflection_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d_backward", "grad_input")
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
Tensor & reflection_pad2d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d", "out")
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
Tensor relu(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::relu", "")
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
Tensor & relu_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::relu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::relu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::relu_", "")
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
Tensor & remainder_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Scalar_out")
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
Tensor & remainder_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Tensor_out")
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
Tensor & replication_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d", "out")
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
Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_relu_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, noise, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rrelu_with_noise_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise_", "")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, noise, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_add", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, src);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor scatter_add_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_add", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, src);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::scatter_add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::scatter_add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_add_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, src);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor select_Dimname(const Tensor & self, Dimname dim, int64_t index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select");
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
      .findSchemaOrThrow("aten::select", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, int64_t>(op, c10::DispatchKey::Tracer, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor select_int(const Tensor & self, int64_t dim, int64_t index) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::select");
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
      .findSchemaOrThrow("aten::select", "int")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, dim, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor sin(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sin", "")
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
Tensor & sin_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sin");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sin_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sin_", "")
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
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated3d_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>)>();
  std::tie(grad_input, grad_weight, grad_bias) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>>(op, c10::DispatchKey::Tracer, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
Tensor & soft_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss", "out")
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
Tensor softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softmax");
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
      .findSchemaOrThrow("aten::softmax", "int")
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
Tensor softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softmax");
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
      .findSchemaOrThrow("aten::softmax", "Dimname")
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
Tensor & softplus_out_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, out, self, beta, threshold);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, grad_output, self, lambd);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> sort_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, values, indices, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
std::tuple<Tensor &,Tensor &> sort_out_dimname_values(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool descending) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sort");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "dimname_values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Tracer, values, indices, self, dim, descending);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor squeeze(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze", "")
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
Tensor squeeze_dim(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze", "dim")
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
Tensor squeeze_dimname(const Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze", "dimname")
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
Tensor & squeeze_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze_", "")
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
Tensor & squeeze__dim(Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze_", "dim")
      .typed<Tensor & (Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & squeeze__dimname(Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze_", "dimname")
      .typed<Tensor & (Tensor &, Dimname)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & std_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Tracer, out, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & std_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("std_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Tracer, out, self, dim, unbiased, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor sub_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor sub_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sub");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "Scalar")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, other, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sub__Tensor(Tensor & self, const Tensor & other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sub");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, other, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & sub__Scalar(Tensor & self, Scalar other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sub");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, other, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & sum_out_IntList_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "IntList_out")
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
Tensor & sum_out_DimnameList_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "DimnameList_out")
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
Tensor & take_out_out(Tensor & out, const Tensor & self, const Tensor & index) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("take_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::take", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor output;
  Tensor finput;
  Tensor fgrad_input;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_forward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  std::tie(output, finput, fgrad_input) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, self, weight, kernel_size, bias, stride, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  #endif
  return std::make_tuple(std::move(output), std::move(finput), std::move(fgrad_input));
}
Tensor & thnn_conv_depthwise2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, weight, kernel_size, bias, stride, padding, dilation);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor to_dense_backward(const Tensor & grad, const Tensor & input) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_dense_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_dense_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad, input);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor &,Tensor &> topk_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::topk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("topk_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::topk", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, values, indices, self, k, dim, largest, sorted);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(values, indices);
}
Tensor & trunc_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc", "out")
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
std::vector<Tensor> unbind_int(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unbind");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unbind", "int")
      .typed<std::vector<Tensor> (const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::vector<Tensor> unbind_Dimname(const Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unbind");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unbind", "Dimname")
      .typed<std::vector<Tensor> (const Tensor &, Dimname)>();
  auto result =c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, Dimname>(op, c10::DispatchKey::Tracer, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unique_consecutive");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_consecutive", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool, c10::optional<int64_t>)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, self, return_inverse, return_counts, dim);
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
Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_linear1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Tracer, self, output_size, align_corners, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_linear1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_input, grad_output, output_size, input_size, align_corners, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_output, output_size, input_size, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, self, output_size, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_nearest2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_input, grad_output, output_size, input_size, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & upsample_nearest3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, out, self, output_size, scales_d, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor vander(const Tensor & x, c10::optional<int64_t> N, bool increasing) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::vander");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "increasing", increasing);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::vander", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Tracer, x, N, increasing);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor view_as(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view_as", "")
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
Tensor view_as_complex(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view_as_complex");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view_as_complex", "")
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
Tensor view_as_real(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::view_as_real");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view_as_real", "")
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
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl_UNBOXED("__ilshift__.Scalar", &TraceType::__ilshift___Scalar);
  m.impl_UNBOXED("__ilshift__.Tensor", &TraceType::__ilshift___Tensor);
  m.impl_UNBOXED("__ior__.Scalar", &TraceType::__ior___Scalar);
  m.impl_UNBOXED("__ior__.Tensor", &TraceType::__ior___Tensor);
  m.impl_UNBOXED("__ixor__.Scalar", &TraceType::__ixor___Scalar);
  m.impl_UNBOXED("__ixor__.Tensor", &TraceType::__ixor___Tensor);
  m.impl("__lshift__.Scalar", TORCH_FN(TraceType::__lshift___Scalar));
  m.impl("__lshift__.Tensor", TORCH_FN(TraceType::__lshift___Tensor));
  m.impl("__or__.Scalar", TORCH_FN(TraceType::__or___Scalar));
  m.impl("__or__.Tensor", TORCH_FN(TraceType::__or___Tensor));
  m.impl("__xor__.Scalar", TORCH_FN(TraceType::__xor___Scalar));
  m.impl("__xor__.Tensor", TORCH_FN(TraceType::__xor___Tensor));
  m.impl_UNBOXED("_addr.out", &TraceType::_addr_out_out);
  m.impl_UNBOXED("_baddbmm_mkl_", &TraceType::_baddbmm_mkl_);
  m.impl_UNBOXED("_bmm.out", &TraceType::_bmm_out_out);
  m.impl("_cast_Double", TORCH_FN(TraceType::_cast_Double));
  m.impl("_cast_Short", TORCH_FN(TraceType::_cast_Short));
  m.impl_UNBOXED("_cudnn_rnn", &TraceType::_cudnn_rnn);
  m.impl_UNBOXED("_cumsum.out", &TraceType::_cumsum_out_out);
  m.impl("_dimV", TORCH_FN(TraceType::_dimV));
  m.impl_UNBOXED("_embedding_bag", &TraceType::_embedding_bag);
  m.impl_UNBOXED("_embedding_bag_sparse_backward", &TraceType::_embedding_bag_sparse_backward);
  m.impl("_gather_sparse_backward", TORCH_FN(TraceType::_gather_sparse_backward));
  m.impl_UNBOXED("_index_put_impl_", &TraceType::_index_put_impl_);
  m.impl("_indices", TORCH_FN(TraceType::_indices));
  m.impl("_local_scalar_dense", TORCH_FN(TraceType::_local_scalar_dense));
  m.impl_UNBOXED("_logcumsumexp.out", &TraceType::_logcumsumexp_out_out);
  m.impl("_mkldnn_transpose", TORCH_FN(TraceType::_mkldnn_transpose));
  m.impl_UNBOXED("_mkldnn_transpose_", &TraceType::_mkldnn_transpose_);
  m.impl_UNBOXED("_mode.values", &TraceType::_mode_out_values);
  m.impl("_nnpack_spatial_convolution_backward_weight", TORCH_FN(TraceType::_nnpack_spatial_convolution_backward_weight));
  m.impl("_pdist_forward", TORCH_FN(TraceType::_pdist_forward));
  m.impl("_softmax", TORCH_FN(TraceType::_softmax));
  m.impl_UNBOXED("_sparse_coo_tensor_with_dims", &TraceType::_sparse_coo_tensor_with_dims);
  m.impl_UNBOXED("_sparse_softmax_backward_data", &TraceType::_sparse_softmax_backward_data);
  m.impl("_sparse_sum", TORCH_FN(TraceType::_sparse_sum));
  m.impl_UNBOXED("_sparse_sum.dtype", &TraceType::_sparse_sum_dtype);
  m.impl("_sparse_sum.dim", TORCH_FN(TraceType::_sparse_sum_dim));
  m.impl_UNBOXED("_sparse_sum.dim_dtype", &TraceType::_sparse_sum_dim_dtype);
  m.impl_UNBOXED("_thnn_differentiable_lstm_cell_backward", &TraceType::_thnn_differentiable_lstm_cell_backward);
  m.impl_UNBOXED("_thnn_fused_gru_cell", &TraceType::_thnn_fused_gru_cell);
  m.impl_UNBOXED("_thnn_fused_lstm_cell_backward", &TraceType::_thnn_fused_lstm_cell_backward);
  m.impl("_trilinear", TORCH_FN(TraceType::_trilinear));
  m.impl("_unsafe_view", TORCH_FN(TraceType::_unsafe_view));
  m.impl("_weight_norm_cuda_interface", TORCH_FN(TraceType::_weight_norm_cuda_interface));
  m.impl("abs", TORCH_FN(TraceType::abs));
  m.impl_UNBOXED("abs_", &TraceType::abs_);
  m.impl_UNBOXED("acosh.out", &TraceType::acosh_out_out);
  m.impl("adaptive_avg_pool1d", TORCH_FN(TraceType::adaptive_avg_pool1d));
  m.impl_UNBOXED("adaptive_avg_pool2d.out", &TraceType::adaptive_avg_pool2d_out_out);
  m.impl("adaptive_max_pool2d_backward", TORCH_FN(TraceType::adaptive_max_pool2d_backward));
  m.impl("adaptive_max_pool3d", TORCH_FN(TraceType::adaptive_max_pool3d));
  m.impl_UNBOXED("adaptive_max_pool3d_backward.grad_input", &TraceType::adaptive_max_pool3d_backward_out_grad_input);
  m.impl("addcmul", TORCH_FN(TraceType::addcmul));
  m.impl_UNBOXED("addcmul_", &TraceType::addcmul_);
  m.impl("addmm", TORCH_FN(TraceType::addmm));
  m.impl_UNBOXED("addmm_", &TraceType::addmm_);
  m.impl_UNBOXED("addr.out", &TraceType::addr_out_out);
  m.impl("align_as", TORCH_FN(TraceType::align_as));
  m.impl("align_tensors", TORCH_FN(TraceType::align_tensors));
  m.impl_UNBOXED("align_to", &TraceType::align_to);
  m.impl_UNBOXED("align_to.ellipsis_idx", &TraceType::align_to_ellipsis_idx);
  m.impl("argmax", TORCH_FN(TraceType::argmax));
  m.impl("argsort", TORCH_FN(TraceType::argsort));
  m.impl_UNBOXED("argsort.dimname", &TraceType::argsort_dimname);
  m.impl_UNBOXED("asinh.out", &TraceType::asinh_out_out);
  m.impl("atan", TORCH_FN(TraceType::atan));
  m.impl("atan2", TORCH_FN(TraceType::atan2));
  m.impl_UNBOXED("atan2_", &TraceType::atan2_);
  m.impl_UNBOXED("atan_", &TraceType::atan_);
  m.impl("avg_pool2d", TORCH_FN(TraceType::avg_pool2d));
  m.impl_UNBOXED("avg_pool2d_backward.grad_input", &TraceType::avg_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("avg_pool3d.out", &TraceType::avg_pool3d_out_out);
  m.impl_UNBOXED("baddbmm.out", &TraceType::baddbmm_out_out);
  m.impl_UNBOXED("bartlett_window", &TraceType::bartlett_window);
  m.impl_UNBOXED("bartlett_window.periodic", &TraceType::bartlett_window_periodic);
  m.impl_UNBOXED("batch_norm_update_stats", &TraceType::batch_norm_update_stats);
  m.impl_UNBOXED("binary_cross_entropy_backward", &TraceType::binary_cross_entropy_backward);
  m.impl_UNBOXED("bitwise_and.Tensor_out", &TraceType::bitwise_and_out_Tensor_out);
  m.impl_UNBOXED("bitwise_and.Scalar_out", &TraceType::bitwise_and_out_Scalar_out);
  m.impl_UNBOXED("bitwise_not.out", &TraceType::bitwise_not_out_out);
  m.impl_UNBOXED("bmm.out", &TraceType::bmm_out_out);
  m.impl_UNBOXED("bucketize.Tensor_out", &TraceType::bucketize_out_Tensor_out);
  m.impl("cdist", TORCH_FN(TraceType::cdist));
  m.impl("celu", TORCH_FN(TraceType::celu));
  m.impl_UNBOXED("celu_", &TraceType::celu_);
  m.impl("cholesky_inverse", TORCH_FN(TraceType::cholesky_inverse));
  m.impl_UNBOXED("cholesky.out", &TraceType::cholesky_out_out);
  m.impl("clamp", TORCH_FN(TraceType::clamp));
  m.impl_UNBOXED("clamp_", &TraceType::clamp_);
  m.impl_UNBOXED("clamp_max.out", &TraceType::clamp_max_out_out);
  m.impl_UNBOXED("conj.out", &TraceType::conj_out_out);
  m.impl_UNBOXED("conv_transpose3d.input", &TraceType::conv_transpose3d_input);
  m.impl("convolution_backward_overrideable", TORCH_FN(TraceType::convolution_backward_overrideable));
  m.impl("cos", TORCH_FN(TraceType::cos));
  m.impl_UNBOXED("cos_", &TraceType::cos_);
  m.impl_UNBOXED("cudnn_batch_norm_backward", &TraceType::cudnn_batch_norm_backward);
  m.impl("cudnn_grid_sampler_backward", TORCH_FN(TraceType::cudnn_grid_sampler_backward));
  m.impl_UNBOXED("cummax.out", &TraceType::cummax_out_out);
  m.impl_UNBOXED("cummax.dimname_out", &TraceType::cummax_out_dimname_out);
  m.impl_UNBOXED("cumsum.out", &TraceType::cumsum_out_out);
  m.impl_UNBOXED("cumsum.dimname_out", &TraceType::cumsum_out_dimname_out);
  m.impl("deg2rad", TORCH_FN(TraceType::deg2rad));
  m.impl_UNBOXED("deg2rad_", &TraceType::deg2rad_);
  m.impl("diag", TORCH_FN(TraceType::diag));
  m.impl("digamma", TORCH_FN(TraceType::digamma));
  m.impl_UNBOXED("digamma_", &TraceType::digamma_);
  m.impl_UNBOXED("elu.out", &TraceType::elu_out_out);
  m.impl_UNBOXED("embedding_bag", &TraceType::embedding_bag);
  m.impl("embedding_dense_backward", TORCH_FN(TraceType::embedding_dense_backward));
  m.impl_UNBOXED("empty_like", &TraceType::empty_like);
  m.impl_UNBOXED("empty_quantized", &TraceType::empty_quantized);
  m.impl_UNBOXED("empty_strided", &TraceType::empty_strided);
  m.impl_UNBOXED("erfc.out", &TraceType::erfc_out_out);
  m.impl("erfinv", TORCH_FN(TraceType::erfinv));
  m.impl_UNBOXED("erfinv_", &TraceType::erfinv_);
  m.impl("expand", TORCH_FN(TraceType::expand));
  m.impl_UNBOXED("expm1.out", &TraceType::expm1_out_out);
  m.impl("fake_quantize_per_tensor_affine_backward", TORCH_FN(TraceType::fake_quantize_per_tensor_affine_backward));
  m.impl("fft", TORCH_FN(TraceType::fft));
  m.impl("flatten.using_ints", TORCH_FN(TraceType::flatten_using_ints));
  m.impl_UNBOXED("flatten.named_out_dim", &TraceType::flatten_named_out_dim);
  m.impl_UNBOXED("flatten.using_names", &TraceType::flatten_using_names);
  m.impl_UNBOXED("flatten.DimnameList", &TraceType::flatten_DimnameList);
  m.impl("floor_divide", TORCH_FN(TraceType::floor_divide));
  m.impl("floor_divide.Scalar", TORCH_FN(TraceType::floor_divide_Scalar));
  m.impl_UNBOXED("floor_divide_.Tensor", &TraceType::floor_divide__Tensor);
  m.impl_UNBOXED("floor_divide_.Scalar", &TraceType::floor_divide__Scalar);
  m.impl_UNBOXED("floor.out", &TraceType::floor_out_out);
  m.impl_UNBOXED("full.names", &TraceType::full_names);
  m.impl_UNBOXED("full", &TraceType::full);
  m.impl("gather", TORCH_FN(TraceType::gather));
  m.impl_UNBOXED("gather.dimname", &TraceType::gather_dimname);
  m.impl("gelu_backward", TORCH_FN(TraceType::gelu_backward));
  m.impl("grid_sampler_3d_backward", TORCH_FN(TraceType::grid_sampler_3d_backward));
  m.impl_UNBOXED("gru_cell", &TraceType::gru_cell);
  m.impl_UNBOXED("hann_window", &TraceType::hann_window);
  m.impl_UNBOXED("hann_window.periodic", &TraceType::hann_window_periodic);
  m.impl("hardshrink", TORCH_FN(TraceType::hardshrink));
  m.impl("ifft", TORCH_FN(TraceType::ifft));
  m.impl_UNBOXED("index_select.out", &TraceType::index_select_out_out);
  m.impl_UNBOXED("index_select.dimname_out", &TraceType::index_select_out_dimname_out);
  m.impl("indices", TORCH_FN(TraceType::indices));
  m.impl("is_complex", TORCH_FN(TraceType::is_complex));
  m.impl("is_same_size", TORCH_FN(TraceType::is_same_size));
  m.impl_UNBOXED("l1_loss.out", &TraceType::l1_loss_out_out);
  m.impl_UNBOXED("layer_norm", &TraceType::layer_norm);
  m.impl("leaky_relu_backward", TORCH_FN(TraceType::leaky_relu_backward));
  m.impl("lerp.Scalar", TORCH_FN(TraceType::lerp_Scalar));
  m.impl("lerp.Tensor", TORCH_FN(TraceType::lerp_Tensor));
  m.impl_UNBOXED("lerp_.Scalar", &TraceType::lerp__Scalar);
  m.impl_UNBOXED("lerp_.Tensor", &TraceType::lerp__Tensor);
  m.impl_UNBOXED("linear", &TraceType::linear);
  m.impl("log_sigmoid", TORCH_FN(TraceType::log_sigmoid));
  m.impl_UNBOXED("log_sigmoid_backward.grad_input", &TraceType::log_sigmoid_backward_out_grad_input);
  m.impl_UNBOXED("logcumsumexp.out", &TraceType::logcumsumexp_out_out);
  m.impl_UNBOXED("logcumsumexp.dimname_out", &TraceType::logcumsumexp_out_dimname_out);
  m.impl_UNBOXED("logical_or.out", &TraceType::logical_or_out_out);
  m.impl_UNBOXED("logical_xor.out", &TraceType::logical_xor_out_out);
  m.impl_UNBOXED("logspace.out", &TraceType::logspace_out_out);
  m.impl_UNBOXED("logsumexp.out", &TraceType::logsumexp_out_out);
  m.impl_UNBOXED("logsumexp.names_out", &TraceType::logsumexp_out_names_out);
  m.impl_UNBOXED("matmul.out", &TraceType::matmul_out_out);
  m.impl_UNBOXED("max.dim_max", &TraceType::max_out_dim_max);
  m.impl_UNBOXED("max.names_dim_max", &TraceType::max_out_names_dim_max);
  m.impl_UNBOXED("max.out", &TraceType::max_out_out);
  m.impl("max_unpool2d", TORCH_FN(TraceType::max_unpool2d));
  m.impl_UNBOXED("max_unpool2d_backward.grad_input", &TraceType::max_unpool2d_backward_out_grad_input);
  m.impl_UNBOXED("max_unpool3d.out", &TraceType::max_unpool3d_out_out);
  m.impl("min_values", TORCH_FN(TraceType::min_values));
  m.impl_UNBOXED("min_values.names", &TraceType::min_values_names);
  m.impl("miopen_convolution_backward", TORCH_FN(TraceType::miopen_convolution_backward));
  m.impl("miopen_convolution_backward_bias", TORCH_FN(TraceType::miopen_convolution_backward_bias));
  m.impl("miopen_convolution_backward_input", TORCH_FN(TraceType::miopen_convolution_backward_input));
  m.impl_UNBOXED("miopen_convolution_transpose", &TraceType::miopen_convolution_transpose);
  m.impl("mkldnn_adaptive_avg_pool2d", TORCH_FN(TraceType::mkldnn_adaptive_avg_pool2d));
  m.impl_UNBOXED("mkldnn_convolution", &TraceType::mkldnn_convolution);
  m.impl("mkldnn_reorder_conv2d_weight", TORCH_FN(TraceType::mkldnn_reorder_conv2d_weight));
  m.impl_UNBOXED("mode.values", &TraceType::mode_out_values);
  m.impl_UNBOXED("mode.dimname_out", &TraceType::mode_out_dimname_out);
  m.impl("mse_loss_backward", TORCH_FN(TraceType::mse_loss_backward));
  m.impl("multilabel_margin_loss_backward", TORCH_FN(TraceType::multilabel_margin_loss_backward));
  m.impl_UNBOXED("multilabel_margin_loss_forward.output", &TraceType::multilabel_margin_loss_forward_out_output);
  m.impl_UNBOXED("multinomial", &TraceType::multinomial);
  m.impl("mvlgamma", TORCH_FN(TraceType::mvlgamma));
  m.impl_UNBOXED("mvlgamma_", &TraceType::mvlgamma_);
  m.impl("narrow", TORCH_FN(TraceType::narrow));
  m.impl("narrow.Tensor", TORCH_FN(TraceType::narrow_Tensor));
  m.impl_UNBOXED("native_batch_norm", &TraceType::native_batch_norm);
  m.impl_UNBOXED("ne.Scalar_out", &TraceType::ne_out_Scalar_out);
  m.impl_UNBOXED("ne.Tensor_out", &TraceType::ne_out_Tensor_out);
  m.impl_UNBOXED("new_full", &TraceType::new_full);
  m.impl_UNBOXED("nll_loss", &TraceType::nll_loss);
  m.impl_UNBOXED("nll_loss2d", &TraceType::nll_loss2d);
  m.impl_UNBOXED("nll_loss2d_backward.grad_input", &TraceType::nll_loss2d_backward_out_grad_input);
  m.impl_UNBOXED("nll_loss_backward.grad_input", &TraceType::nll_loss_backward_out_grad_input);
  m.impl("nuclear_norm", TORCH_FN(TraceType::nuclear_norm));
  m.impl("nuclear_norm.dim", TORCH_FN(TraceType::nuclear_norm_dim));
  m.impl("orgqr", TORCH_FN(TraceType::orgqr));
  m.impl_UNBOXED("ormqr.out", &TraceType::ormqr_out_out);
  m.impl("permute", TORCH_FN(TraceType::permute));
  m.impl("pixel_shuffle", TORCH_FN(TraceType::pixel_shuffle));
  m.impl_UNBOXED("put_", &TraceType::put_);
  m.impl("q_zero_point", TORCH_FN(TraceType::q_zero_point));
  m.impl_UNBOXED("quantize_per_tensor", &TraceType::quantize_per_tensor);
  m.impl_UNBOXED("quantize_per_tensor.tensors", &TraceType::quantize_per_tensor_tensors);
  m.impl("quantized_lstm_cell", TORCH_FN(TraceType::quantized_lstm_cell));
  m.impl("rad2deg", TORCH_FN(TraceType::rad2deg));
  m.impl_UNBOXED("rad2deg_", &TraceType::rad2deg_);
  m.impl_UNBOXED("rand.out", &TraceType::rand_out_out);
  m.impl_UNBOXED("rand.generator_out", &TraceType::rand_out_generator_out);
  m.impl_UNBOXED("randn.out", &TraceType::randn_out_out);
  m.impl_UNBOXED("randn.generator_out", &TraceType::randn_out_generator_out);
  m.impl_UNBOXED("range.step", &TraceType::range_step);
  m.impl_UNBOXED("range", &TraceType::range);
  m.impl("real", TORCH_FN(TraceType::real));
  m.impl("reciprocal", TORCH_FN(TraceType::reciprocal));
  m.impl_UNBOXED("reciprocal_", &TraceType::reciprocal_);
  m.impl_UNBOXED("refine_names", &TraceType::refine_names);
  m.impl("reflection_pad1d", TORCH_FN(TraceType::reflection_pad1d));
  m.impl_UNBOXED("reflection_pad1d_backward.grad_input", &TraceType::reflection_pad1d_backward_out_grad_input);
  m.impl_UNBOXED("reflection_pad2d.out", &TraceType::reflection_pad2d_out_out);
  m.impl("relu", TORCH_FN(TraceType::relu));
  m.impl_UNBOXED("relu_", &TraceType::relu_);
  m.impl_UNBOXED("remainder.Scalar_out", &TraceType::remainder_out_Scalar_out);
  m.impl_UNBOXED("remainder.Tensor_out", &TraceType::remainder_out_Tensor_out);
  m.impl_UNBOXED("replication_pad1d.out", &TraceType::replication_pad1d_out_out);
  m.impl_UNBOXED("rnn_relu_cell", &TraceType::rnn_relu_cell);
  m.impl_UNBOXED("rrelu_with_noise", &TraceType::rrelu_with_noise);
  m.impl_UNBOXED("rrelu_with_noise_", &TraceType::rrelu_with_noise_);
  m.impl("scatter_add", TORCH_FN(TraceType::scatter_add));
  m.impl_UNBOXED("scatter_add.dimname", &TraceType::scatter_add_dimname);
  m.impl_UNBOXED("scatter_add_", &TraceType::scatter_add_);
  m.impl_UNBOXED("select.Dimname", &TraceType::select_Dimname);
  m.impl("select.int", TORCH_FN(TraceType::select_int));
  m.impl("sin", TORCH_FN(TraceType::sin));
  m.impl_UNBOXED("sin_", &TraceType::sin_);
  m.impl("slow_conv_dilated3d_backward", TORCH_FN(TraceType::slow_conv_dilated3d_backward));
  m.impl_UNBOXED("soft_margin_loss.out", &TraceType::soft_margin_loss_out_out);
  m.impl_UNBOXED("softmax.int", &TraceType::softmax_int);
  m.impl_UNBOXED("softmax.Dimname", &TraceType::softmax_Dimname);
  m.impl_UNBOXED("softplus.out", &TraceType::softplus_out_out);
  m.impl("softshrink_backward", TORCH_FN(TraceType::softshrink_backward));
  m.impl_UNBOXED("sort.values", &TraceType::sort_out_values);
  m.impl_UNBOXED("sort.dimname_values", &TraceType::sort_out_dimname_values);
  m.impl("squeeze", TORCH_FN(TraceType::squeeze));
  m.impl("squeeze.dim", TORCH_FN(TraceType::squeeze_dim));
  m.impl_UNBOXED("squeeze.dimname", &TraceType::squeeze_dimname);
  m.impl_UNBOXED("squeeze_", &TraceType::squeeze_);
  m.impl_UNBOXED("squeeze_.dim", &TraceType::squeeze__dim);
  m.impl_UNBOXED("squeeze_.dimname", &TraceType::squeeze__dimname);
  m.impl_UNBOXED("std.out", &TraceType::std_out_out);
  m.impl_UNBOXED("std.names_out", &TraceType::std_out_names_out);
  m.impl("sub.Tensor", TORCH_FN(TraceType::sub_Tensor));
  m.impl("sub.Scalar", TORCH_FN(TraceType::sub_Scalar));
  m.impl_UNBOXED("sub_.Tensor", &TraceType::sub__Tensor);
  m.impl_UNBOXED("sub_.Scalar", &TraceType::sub__Scalar);
  m.impl_UNBOXED("sum.IntList_out", &TraceType::sum_out_IntList_out);
  m.impl_UNBOXED("sum.DimnameList_out", &TraceType::sum_out_DimnameList_out);
  m.impl_UNBOXED("take.out", &TraceType::take_out_out);
  m.impl_UNBOXED("thnn_conv2d_forward", &TraceType::thnn_conv2d_forward);
  m.impl_UNBOXED("thnn_conv_depthwise2d.out", &TraceType::thnn_conv_depthwise2d_out_out);
  m.impl("to_dense_backward", TORCH_FN(TraceType::to_dense_backward));
  m.impl_UNBOXED("topk.values", &TraceType::topk_out_values);
  m.impl_UNBOXED("trunc.out", &TraceType::trunc_out_out);
  m.impl("unbind.int", TORCH_FN(TraceType::unbind_int));
  m.impl_UNBOXED("unbind.Dimname", &TraceType::unbind_Dimname);
  m.impl("unique_consecutive", TORCH_FN(TraceType::unique_consecutive));
  m.impl("upsample_bilinear2d_backward", TORCH_FN(TraceType::upsample_bilinear2d_backward));
  m.impl("upsample_linear1d", TORCH_FN(TraceType::upsample_linear1d));
  m.impl_UNBOXED("upsample_linear1d_backward.grad_input", &TraceType::upsample_linear1d_backward_out_grad_input);
  m.impl("upsample_nearest1d_backward", TORCH_FN(TraceType::upsample_nearest1d_backward));
  m.impl("upsample_nearest2d", TORCH_FN(TraceType::upsample_nearest2d));
  m.impl_UNBOXED("upsample_nearest2d_backward.grad_input", &TraceType::upsample_nearest2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest3d.out", &TraceType::upsample_nearest3d_out_out);
  m.impl("vander", TORCH_FN(TraceType::vander));
  m.impl("view_as", TORCH_FN(TraceType::view_as));
  m.impl("view_as_complex", TORCH_FN(TraceType::view_as_complex));
  m.impl("view_as_real", TORCH_FN(TraceType::view_as_real));;
}

}  // namespace

} // namespace torch
