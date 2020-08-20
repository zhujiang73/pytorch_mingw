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
std::tuple<Tensor,Tensor,Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var_transform, bool train, double eps, std::array<bool,3> output_mask, const Tensor & reservedSpace) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_batch_norm_impl_index_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "impl_index", impl_index);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var_transform", save_var_transform);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    jit::tracer::addInputs(node, "reservedSpace", reservedSpace);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_batch_norm_impl_index_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &>(op, c10::DispatchKey::Tracer, impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
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
Tensor _cast_Char(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Char");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Char", "")
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
Tensor _cast_Float(const Tensor & self, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Float");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Float", "")
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
Tensor _cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cholesky_solve_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cholesky_solve_helper", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, A, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution_nogroup");
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_convolution_nogroup", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, dilation, transposed, output_padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _copy_from(const Tensor & self, const Tensor & dst, bool non_blocking) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_copy_from");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dst", dst);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_copy_from", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, dst, non_blocking);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_ctc_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "neg_log_likelihood", neg_log_likelihood);
    jit::tracer::addInputs(node, "log_alpha", log_alpha);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_ctc_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cudnn_ctc_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_ctc_loss", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_set_plan_cache_max_size", "")
      .typed<void (int64_t, int64_t)>();
  c10::Dispatcher::singleton().redispatch<void, int64_t, int64_t>(op, c10::DispatchKey::Tracer, device_index, max_size);
}
void _cummin_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cummin_helper", "")
      .typed<void (const Tensor &, Tensor &, Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<void, const Tensor &, Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, values, indices, dim);
}
Tensor _euclidean_dist(const Tensor & x1, const Tensor & x2) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_euclidean_dist");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_euclidean_dist", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, x1, x2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_index_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_index_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_index_copy_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _inverse_helper(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_inverse_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_inverse_helper", "")
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
Tensor _masked_scale(const Tensor & self, const Tensor & mask, double scale) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_masked_scale");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_masked_scale", "")
      .typed<Tensor (const Tensor &, const Tensor &, double)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Tracer, self, mask, scale);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
bool _nnpack_available() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_available", "")
      .typed<bool ()>();
  auto result =c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Tracer);
  return result;
}
Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pdist_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "pdist", pdist);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pdist_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Tracer, grad, self, p, pdist);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_s_where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_s_where", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, condition, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sample_dirichlet(const Tensor & self, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sample_dirichlet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sample_dirichlet", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & _sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_scramble");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_scramble_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ltm", ltm);
    jit::tracer::addInputs(node, "dimension", dimension);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sobol_engine_scramble_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sobol_engine_scramble_", "")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, ltm, dimension);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor _sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_addmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "dense", dense);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_addmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, sparse, dense, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_unsafe");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_unsafe", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Tracer, indices, values, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims_and_tensors", "")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &>(op, c10::DispatchKey::Tracer, sparse_dim, dense_dim, size, indices, values, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor _sparse_softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_softmax");
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
      .findSchemaOrThrow("aten::_sparse_softmax", "int")
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
Tensor _sparse_softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_softmax");
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
      .findSchemaOrThrow("aten::_sparse_softmax", "Dimname")
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
Tensor _sparse_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_softmax");
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
      .findSchemaOrThrow("aten::_sparse_softmax", "")
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
Tensor _standard_gamma(const Tensor & self, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_standard_gamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_standard_gamma", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> _symeig_helper(const Tensor & self, bool eigenvectors, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_symeig_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_symeig_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, eigenvectors, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input_gates, hidden_gates, cx, input_bias, hidden_bias);
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
std::tuple<Tensor,Tensor> _triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_triangular_solve_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_triangular_solve_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Tracer, self, A, upper, transpose, unitriangular);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & abs_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs", "out")
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
std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool2d");
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
      .findSchemaOrThrow("aten::adaptive_max_pool2d", "")
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
Tensor & adaptive_max_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d_backward", "grad_input")
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
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_max_pool3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, out, indices, self, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  #endif
  return std::forward_as_tuple(out, indices);
}
Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addcdiv");
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
      .findSchemaOrThrow("aten::addcdiv", "")
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
Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addcdiv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addcdiv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcdiv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcdiv_", "")
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
Tensor & addcmul_out_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, self, tensor1, tensor2, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & addmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmm", "out")
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
Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addmv");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmv", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, mat, vec, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::addmv");
    } else {
      op_name = jit::Symbol::fromQualString("aten::addmv_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmv_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, mat, vec, beta, alpha);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor alpha_dropout(const Tensor & input, double p, bool train) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::alpha_dropout");
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
      .findSchemaOrThrow("aten::alpha_dropout", "")
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
Tensor & alpha_dropout_(Tensor & self, double p, bool train) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::alpha_dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::alpha_dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::alpha_dropout_", "")
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
Tensor angle(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::angle");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::angle", "")
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
Tensor & atan2_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2", "out")
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
Tensor & atan_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan", "out")
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
Tensor atanh(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::atanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atanh", "")
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
Tensor & atanh_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::atanh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::atanh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atanh_", "")
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
Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::avg_pool1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool>(op, c10::DispatchKey::Tracer, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d", "out")
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
Tensor batch_norm_backward_elemt(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_backward_elemt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "mean_dy", mean_dy);
    jit::tracer::addInputs(node, "mean_dy_xmu", mean_dy_xmu);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_backward_elemt", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, const Tensor & counts) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_gather_stats_with_counts");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "counts", counts);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_gather_stats_with_counts", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &>(op, c10::DispatchKey::Tracer, input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, self, target, weight, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & binary_cross_entropy_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, target, weight, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor bitwise_or_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or", "Scalar")
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
Tensor bitwise_or_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or", "Tensor")
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
Tensor & bitwise_or__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or_", "Scalar")
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
Tensor & bitwise_or__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_or_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or_", "Tensor")
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
Tensor bitwise_xor_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor", "Scalar")
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
Tensor bitwise_xor_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor", "Tensor")
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
Tensor & bitwise_xor__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor_", "Scalar")
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
Tensor & bitwise_xor__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bitwise_xor_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor_", "Tensor")
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
Tensor blackman_window(int64_t window_length, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::blackman_window", "")
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
Tensor blackman_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::blackman_window");
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
      .findSchemaOrThrow("aten::blackman_window", "periodic")
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
std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::broadcast_tensors");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::broadcast_tensors", "")
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
Tensor & cholesky_inverse_out_out(Tensor & out, const Tensor & self, bool upper) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_inverse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_inverse", "out")
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
Tensor cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cholesky_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_solve", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, input2, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor clamp_min(const Tensor & self, Scalar min) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clamp_min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_min", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, min);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & clamp_min_(Tensor & self, Scalar min) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::clamp_min");
    } else {
      op_name = jit::Symbol::fromQualString("aten::clamp_min_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_min_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_min_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, min);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & clamp_out_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, c10::DispatchKey::Tracer, out, self, min, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::clone");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clone", "")
      .typed<Tensor (const Tensor &, c10::optional<MemoryFormat>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Tracer, self, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor contiguous(const Tensor & self, MemoryFormat memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::contiguous");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::contiguous", "")
      .typed<Tensor (const Tensor &, MemoryFormat)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, MemoryFormat>(op, c10::DispatchKey::Tracer, self, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, dilation, groups);
  return result;
}
Tensor conv_transpose2d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose2d", "input")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::convolution_overrideable");
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution_overrideable", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(op, c10::DispatchKey::Tracer, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & cos_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cos", "out")
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
Tensor cosh(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosh", "")
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
Tensor & cosh_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cosh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cosh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosh_", "")
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
Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cross");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cross", "")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, self, other, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_batch_norm");
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
  Tensor result3;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  std::tie(result0, result1, result2, result3) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Tracer, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
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
std::tuple<Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,2> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward");
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
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,2>)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,2>>(op, c10::DispatchKey::Tracer, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward_input");
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
      .findSchemaOrThrow("aten::cudnn_convolution_transpose_backward_input", "")
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
Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_grid_sampler", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto output =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, grid);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  #endif
  return output;
}
bool cudnn_is_acceptable(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_is_acceptable", "")
      .typed<bool (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
std::tuple<Tensor,Tensor> cummin(const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummin");
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
      .findSchemaOrThrow("aten::cummin", "")
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
std::tuple<Tensor,Tensor> cummin_dimname(const Tensor & self, Dimname dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cummin");
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
      .findSchemaOrThrow("aten::cummin", "dimname")
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
Tensor & deg2rad_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("deg2rad_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad", "out")
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
Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::diag_embed");
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
      .findSchemaOrThrow("aten::diag_embed", "")
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
Tensor & diag_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diag_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diag", "out")
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
Tensor & digamma_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma", "out")
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
std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor eigenvalues;
  Tensor eigenvectors_return;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eig", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  std::tie(eigenvalues, eigenvectors_return) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, eigenvectors);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors_return);
  }
  #endif
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors_return));
}
Tensor empty_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &, c10::optional<MemoryFormat>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Tracer, size, names, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor empty_memory_format(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty", "memory_format")
      .typed<Tensor (IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Tracer, size, options, memory_format);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor eq_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq", "Scalar")
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
Tensor eq_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eq");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq", "Tensor")
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
Tensor & eq__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::eq");
    } else {
      op_name = jit::Symbol::fromQualString("aten::eq_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq_", "Scalar")
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
Tensor & eq__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::eq");
    } else {
      op_name = jit::Symbol::fromQualString("aten::eq_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq_", "Tensor")
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
Tensor & erfinv_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv", "out")
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
Tensor fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
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
      .findSchemaOrThrow("aten::fake_quantize_per_tensor_affine", "")
      .typed<Tensor (const Tensor &, double, int64_t, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Tracer, self, scale, zero_point, quant_min, quant_max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fbgemm_pack_quantized_matrix(const Tensor & input) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_pack_quantized_matrix");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_pack_quantized_matrix", "")
      .typed<Tensor (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Tracer, input);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor fbgemm_pack_quantized_matrix_KN(const Tensor & input, int64_t K, int64_t N) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_pack_quantized_matrix");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "K", K);
    jit::tracer::addInputs(node, "N", N);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_pack_quantized_matrix", "KN")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, input, K, N);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor feature_dropout(const Tensor & input, double p, bool train) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::feature_dropout");
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
      .findSchemaOrThrow("aten::feature_dropout", "")
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
Tensor & feature_dropout_(Tensor & self, double p, bool train) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::feature_dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::feature_dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("feature_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::feature_dropout_", "")
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
Tensor & fill_diagonal_(Tensor & self, Scalar fill_value, bool wrap) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::fill_diagonal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fill_diagonal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "wrap", wrap);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_diagonal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fill_diagonal_", "")
      .typed<Tensor & (Tensor &, Scalar, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, bool>(op, c10::DispatchKey::Tracer, self, fill_value, wrap);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor fliplr(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fliplr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fliplr", "")
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
Tensor flipud(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flipud");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flipud", "")
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
Tensor & floor_divide_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "out")
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
Tensor fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, kernel_size, output_size, indices);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & full_out_out(Tensor & out, IntArrayRef size, Scalar fill_value) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("full_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "out")
      .typed<Tensor & (Tensor &, IntArrayRef, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, Scalar>(op, c10::DispatchKey::Tracer, out, size, fill_value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & gather_out_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gather_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, const Tensor &, bool>(op, c10::DispatchKey::Tracer, out, self, dim, index, sparse_grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & gather_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gather_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, const Tensor &, bool>(op, c10::DispatchKey::Tracer, out, self, dim, index, sparse_grad);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor ge_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge", "Scalar")
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
Tensor ge_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ge");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge", "Tensor")
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
Tensor & ge__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ge");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ge_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge_", "Scalar")
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
Tensor & ge__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::ge");
    } else {
      op_name = jit::Symbol::fromQualString("aten::ge_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge_", "Tensor")
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
Tensor gelu(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gelu", "")
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
Tensor & geometric_(Tensor & self, double p, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::geometric");
    } else {
      op_name = jit::Symbol::fromQualString("aten::geometric_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("geometric_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geometric_", "")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, p, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Tracer, grad_output, self, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> grid_sampler_2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_2d_backward");
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
      .findSchemaOrThrow("aten::grid_sampler_2d_backward", "")
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
Tensor grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::grid_sampler_3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Tracer, input, grid, interpolation_mode, padding_mode, align_corners);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> gru_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gru");
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
      .findSchemaOrThrow("aten::gru", "input")
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
std::tuple<Tensor,Tensor> gru_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gru");
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
      .findSchemaOrThrow("aten::gru", "data")
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
Tensor gt_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt", "Scalar")
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
Tensor gt_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt", "Tensor")
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
Tensor & gt__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::gt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::gt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt_", "Scalar")
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
Tensor & gt__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::gt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::gt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt_", "Tensor")
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
Tensor hardsigmoid_backward(const Tensor & grad_output, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardsigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hardswish_backward(const Tensor & grad_output, const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardswish_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardswish_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hinge_embedding_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hinge_embedding_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, int64_t>(op, c10::DispatchKey::Tracer, self, target, margin, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::histc");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::histc", "")
      .typed<Tensor (const Tensor &, int64_t, Scalar, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, Scalar, Scalar>(op, c10::DispatchKey::Tracer, self, bins, min, max);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor hspmm(const Tensor & mat1, const Tensor & mat2) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hspmm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, mat1, mat2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor imag(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::imag");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::imag", "")
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
Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_copy_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_copy__dimname(Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_copy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_copy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy_", "dimname")
      .typed<Tensor & (Tensor &, Dimname, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, source);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor index_fill_int_Scalar(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "int_Scalar")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_fill_int_Tensor(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "int_Tensor")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_fill_Dimname_Scalar(const Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "Dimname_Scalar")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor index_fill_Dimname_Tensor(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "Dimname_Tensor")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & index_fill__int_Scalar(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "int_Scalar")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_fill__int_Tensor(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "int_Tensor")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_fill__Dimname_Scalar(Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "Dimname_Scalar")
      .typed<Tensor & (Tensor &, Dimname, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & index_fill__Dimname_Tensor(Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "Dimname_Tensor")
      .typed<Tensor & (Tensor &, Dimname, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, dim, index, value);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor inverse(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::inverse");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::inverse", "")
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
Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::irfft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "signal_sizes", signal_sizes);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::irfft", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(op, c10::DispatchKey::Tracer, self, signal_ndim, normalized, onesided, signal_sizes);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
bool is_nonzero(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_nonzero", "")
      .typed<bool (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
bool is_set_to(const Tensor & self, const Tensor & tensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_set_to", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, tensor);
  return result;
}
bool is_signed(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_signed", "")
      .typed<bool (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  return result;
}
Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::isclose");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "equal_nan", equal_nan);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::isclose", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, double, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, double, bool>(op, c10::DispatchKey::Tracer, self, other, rtol, atol, equal_nan);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor isfinite(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::isfinite");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::isfinite", "")
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
Tensor istft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool center, bool normalized, bool onesided, c10::optional<int64_t> length) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::istft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n_fft", n_fft);
    jit::tracer::addInputs(node, "hop_length", hop_length);
    jit::tracer::addInputs(node, "win_length", win_length);
    jit::tracer::addInputs(node, "window", window);
    jit::tracer::addInputs(node, "center", center);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::istft", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Tracer, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kl_div_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "log_target", log_target);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kl_div_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Tracer, grad_output, self, target, reduction, log_target);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor le_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le", "Scalar")
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
Tensor le_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::le");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le", "Tensor")
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
Tensor & le__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::le");
    } else {
      op_name = jit::Symbol::fromQualString("aten::le_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le_", "Scalar")
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
Tensor & le__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::le");
    } else {
      op_name = jit::Symbol::fromQualString("aten::le_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le_", "Tensor")
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
Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, negative_slope);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::leaky_relu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::leaky_relu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Tracer, self, negative_slope);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & lerp_out_Scalar_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, out, self, end, weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & lerp_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lerp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, end, weight);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor log10(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log10");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log10", "")
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
Tensor & log10_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log10");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log10_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log10_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log10_", "")
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
Tensor & log_sigmoid_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid", "out")
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
Tensor logdet(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::logdet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logdet", "")
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
std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input, hx, w_ih, w_hh, b_ih, b_hh);
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor lt_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt", "Scalar")
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
Tensor lt_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt", "Tensor")
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
Tensor & lt__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt_", "Scalar")
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
Tensor & lt__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::lt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::lt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt_", "Tensor")
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
Tensor masked_select(const Tensor & self, const Tensor & mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_select", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, self, mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor matrix_rank_tol(const Tensor & self, double tol, bool symmetric) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matrix_rank");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tol", tol);
    jit::tracer::addInputs(node, "symmetric", symmetric);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matrix_rank", "tol")
      .typed<Tensor (const Tensor &, double, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Tracer, self, tol, symmetric);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor matrix_rank(const Tensor & self, bool symmetric) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::matrix_rank");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "symmetric", symmetric);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matrix_rank", "")
      .typed<Tensor (const Tensor &, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Tracer, self, symmetric);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d");
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
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, self, kernel_size, stride, padding, dilation, ceil_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & max_unpool2d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_unpool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, out, self, indices, output_size);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor,Tensor> min_dim(const Tensor & self, int64_t dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
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
      .findSchemaOrThrow("aten::min", "dim")
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
std::tuple<Tensor,Tensor> min_names_dim(const Tensor & self, Dimname dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
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
      .findSchemaOrThrow("aten::min", "names_dim")
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
Tensor min_other(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "other")
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
Tensor min(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::min");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "")
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
Tensor miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
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
      .findSchemaOrThrow("aten::miopen_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_convolution_transpose_backward_weight");
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
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward_weight", "")
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
std::tuple<Tensor,Tensor> mkldnn_convolution_backward_weights(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_convolution_backward_weights");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_size", weight_size);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution_backward_weights", "")
      .typed<std::tuple<Tensor,Tensor> (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool)>();
  std::tie(result0, result1) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool>(op, c10::DispatchKey::Tracer, weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  #endif
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_linear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_linear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, input, weight, bias);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mse_loss");
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
      .findSchemaOrThrow("aten::mse_loss", "")
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
Tensor & mse_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss_backward", "grad_input")
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
Tensor mul_Tensor(const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul", "Tensor")
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
Tensor mul_Scalar(const Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul", "Scalar")
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
Tensor & mul__Tensor(Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul_", "Tensor")
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
Tensor & mul__Scalar(Tensor & self, Scalar other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::mul");
    } else {
      op_name = jit::Symbol::fromQualString("aten::mul_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mul_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul_", "Scalar")
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
Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multilabel_margin_loss");
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
      .findSchemaOrThrow("aten::multilabel_margin_loss", "")
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
Tensor & multilabel_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, target, reduction, is_target);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & multinomial_out_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multinomial_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multinomial", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, self, num_samples, replacement, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out_out(Tensor & out, Tensor & save_mean, Tensor & save_invstd, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_invstd", save_invstd);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_batch_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm", "out")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Tracer, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, save_mean);
    jit::tracer::addOutput(node, save_invstd);
  }
  #endif
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_group_norm_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "rstd", rstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "HxW", HxW);
    jit::tracer::addInputs(node, "group", group);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_group_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Tracer, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
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
Tensor neg(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::neg");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::neg", "")
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
Tensor & neg_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::neg");
    } else {
      op_name = jit::Symbol::fromQualString("aten::neg_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("neg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::neg_", "")
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
Tensor new_empty(const Tensor & self, IntArrayRef size, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::new_empty");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_empty", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Tracer, self, size, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & nll_loss2d_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, out, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & nll_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Tracer, out, self, target, weight, reduction, ignore_index);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor nonzero(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nonzero", "")
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
Tensor norm_ScalarOpt_dtype(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "ScalarOpt_dtype")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, ScalarType>(op, c10::DispatchKey::Tracer, self, p, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor norm_Scalar(const Tensor & self, Scalar p) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "Scalar")
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
Tensor norm_ScalarOpt_dim_dtype(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "ScalarOpt_dim_dtype")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(op, c10::DispatchKey::Tracer, self, p, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor norm_ScalarOpt_dim(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "ScalarOpt_dim")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, self, p, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor norm_names_ScalarOpt_dim_dtype(const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "names_ScalarOpt_dim_dtype")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType>(op, c10::DispatchKey::Tracer, self, p, dim, keepdim, dtype);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor norm_names_ScalarOpt_dim(const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "names_ScalarOpt_dim")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, DimnameList, bool>(op, c10::DispatchKey::Tracer, self, p, dim, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_Tensor_float(const Tensor & mean, double std, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "Tensor_float")
      .typed<Tensor (const Tensor &, double, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, mean, std, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_float_Tensor(double mean, const Tensor & std, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "float_Tensor")
      .typed<Tensor (double, const Tensor &, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, double, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, mean, std, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_Tensor_Tensor(const Tensor & mean, const Tensor & std, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "Tensor_Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, mean, std, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor normal_float_float(double mean, double std, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::normal");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "float_float")
      .typed<Tensor (double, double, IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, double, double, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Tracer, mean, std, size, generator, options);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & normal_(Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::normal");
    } else {
      op_name = jit::Symbol::fromQualString("aten::normal_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal_", "")
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
Tensor & nuclear_norm_out_out(Tensor & out, const Tensor & self, bool keepdim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nuclear_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Tracer, out, self, keepdim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor & nuclear_norm_out_dim_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nuclear_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "dim_out")
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
Tensor & orgqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("orgqr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::orgqr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, out, self, input2);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor poisson(const Tensor & self, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::poisson");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::poisson", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::poisson_nll_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "log_input", log_input);
    jit::tracer::addInputs(node, "full", full);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::poisson_nll_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool, bool, double, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool, bool, double, int64_t>(op, c10::DispatchKey::Tracer, input, target, log_input, full, eps, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
ScalarType promote_types(ScalarType type1, ScalarType type2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::promote_types", "")
      .typed<ScalarType (ScalarType, ScalarType)>();
  auto result =c10::Dispatcher::singleton().redispatch<ScalarType, ScalarType, ScalarType>(op, c10::DispatchKey::Tracer, type1, type2);
  return result;
}
Tensor q_per_channel_scales(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::q_per_channel_scales");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_scales", "")
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
Tensor quantized_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_batch_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "var", var);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_scale", output_scale);
    jit::tracer::addInputs(node, "output_zero_point", output_zero_point);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_batch_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Tracer, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantized_max_pool2d");
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
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Tracer, self, kernel_size, stride, padding, dilation, ceil_mode);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & rad2deg_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rad2deg_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg", "out")
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
Tensor & range_out_out(Tensor & out, Scalar start, Scalar end, Scalar step) {
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
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("range_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "out")
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
Tensor & reciprocal_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal", "out")
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
Tensor & reflection_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d", "out")
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
Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::renorm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::renorm", "")
      .typed<Tensor (const Tensor &, Scalar, int64_t, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, int64_t, Scalar>(op, c10::DispatchKey::Tracer, self, p, dim, maxnorm);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::renorm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::renorm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::renorm_", "")
      .typed<Tensor & (Tensor &, Scalar, int64_t, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, int64_t, Scalar>(op, c10::DispatchKey::Tracer, self, p, dim, maxnorm);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d_backward");
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
      .findSchemaOrThrow("aten::replication_pad3d_backward", "")
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
Tensor reshape(const Tensor & self, IntArrayRef shape) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::reshape");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reshape", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Tracer, self, shape);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rfft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rfft", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, self, signal_ndim, normalized, onesided);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor> rnn_relu_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_relu");
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
      .findSchemaOrThrow("aten::rnn_relu", "input")
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
std::tuple<Tensor,Tensor> rnn_relu_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rnn_relu");
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
      .findSchemaOrThrow("aten::rnn_relu", "data")
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
Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rrelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rrelu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rrelu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_", "")
      .typed<Tensor & (Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, self, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  #endif
  return self;
}
Tensor & rrelu_with_noise_out_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Tracer, out, self, noise, lower, upper, training, generator);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor rsub_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsub");
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
      .findSchemaOrThrow("aten::rsub", "Tensor")
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
Tensor rsub_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsub");
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
      .findSchemaOrThrow("aten::rsub", "Scalar")
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
Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, grad_output, output);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sin_out_out(Tensor & out, const Tensor & self) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sin", "out")
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
Tensor sinh(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sinh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sinh", "")
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
Tensor & sinh_(Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sinh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sinh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sinh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sinh_", "")
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
std::tuple<Tensor,Tensor> slogdet(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slogdet");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor sign;
  Tensor logabsdet;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slogdet", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  std::tie(sign, logabsdet) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Tracer, self);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, sign);
    jit::tracer::addOutput(node, logabsdet);
  }
  #endif
  return std::make_tuple(std::move(sign), std::move(logabsdet));
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv3d_forward");
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
      .findSchemaOrThrow("aten::slow_conv3d_forward", "")
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
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated2d_backward");
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
      .findSchemaOrThrow("aten::slow_conv_dilated2d_backward", "")
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
Tensor slow_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_dilated3d");
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
      .findSchemaOrThrow("aten::slow_conv_dilated3d", "")
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
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose3d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::slow_conv_transpose3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor grad_input_return;
  Tensor grad_weight;
  Tensor grad_bias;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  std::tie(grad_input_return, grad_weight, grad_bias) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Tracer, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input_return);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input_return), std::move(grad_weight), std::move(grad_bias));
}
Tensor softshrink(const Tensor & self, Scalar lambd) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink", "")
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
Tensor & softshrink_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softshrink_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, grad_input, grad_output, self, lambd);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor stack(TensorList tensors, int64_t dim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::stack");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stack", "")
      .typed<Tensor (TensorList, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t>(op, c10::DispatchKey::Tracer, tensors, dim);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::stft");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n_fft", n_fft);
    jit::tracer::addInputs(node, "hop_length", hop_length);
    jit::tracer::addInputs(node, "win_length", win_length);
    jit::tracer::addInputs(node, "window", window);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stft", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, n_fft, hop_length, win_length, window, normalized, onesided);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & sub_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "out")
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
std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::symeig");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor eigenvalues;
  Tensor eigenvectors_return;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::symeig", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  std::tie(eigenvalues, eigenvectors_return) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Tracer, self, eigenvectors, upper);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors_return);
  }
  #endif
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors_return));
}
Tensor tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tensordot");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims_self", dims_self);
    jit::tracer::addInputs(node, "dims_other", dims_other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tensordot", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, self, other, dims_self, dims_other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor grad_input_return;
  Tensor grad_weight;
  Tensor grad_bias;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  std::tie(grad_input_return, grad_weight, grad_bias) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Tracer, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input_return);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  #endif
  return std::make_tuple(std::move(grad_input_return), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out_output(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "finput", finput);
    jit::tracer::addInputs(node, "fgrad_input", fgrad_input);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Tracer, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, finput);
    jit::tracer::addOutput(node, fgrad_input);
  }
  #endif
  return std::forward_as_tuple(output, finput, fgrad_input);
}
Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::threshold_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::threshold_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Tracer, grad_output, self, threshold);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor to_dense(const Tensor & self) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_dense");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_dense", "")
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
std::tuple<Tensor,Tensor> triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triangular_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "transpose", transpose);
    jit::tracer::addInputs(node, "unitriangular", unitriangular);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor solution;
  Tensor cloned_coefficient;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triangular_solve", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>();
  std::tie(solution, cloned_coefficient) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Tracer, self, A, upper, transpose, unitriangular);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, cloned_coefficient);
  }
  #endif
  return std::make_tuple(std::move(solution), std::move(cloned_coefficient));
}
Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triplet_margin_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "anchor", anchor);
    jit::tracer::addInputs(node, "positive", positive);
    jit::tracer::addInputs(node, "negative", negative);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "swap", swap);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triplet_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t>(op, c10::DispatchKey::Tracer, anchor, positive, negative, margin, p, eps, swap, reduction);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor unfold_backward(const Tensor & grad_in, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unfold_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_in", grad_in);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unfold_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, int64_t, int64_t, int64_t)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Tracer, grad_in, input_sizes, dim, size, step);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unique_dim_consecutive");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  Tensor result0;
  Tensor result1;
  Tensor result2;
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_dim_consecutive", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, int64_t, bool, bool)>();
  std::tie(result0, result1, result2) =c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Tracer, self, dim, return_inverse, return_counts);
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
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bilinear2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, self, output_size, align_corners, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_bilinear2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bilinear2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & upsample_linear1d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Tracer, out, self, output_size, align_corners, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor upsample_nearest1d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Tracer, self, output_size, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor & upsample_nearest1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Tracer, grad_input, grad_output, output_size, input_size, scales);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  #endif
  return grad_input;
}
Tensor & upsample_nearest2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
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
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Tracer, out, self, output_size, scales_h, scales_w);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  #endif
  return out;
}
Tensor var(const Tensor & self, bool unbiased) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var", "")
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
Tensor var_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
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
      .findSchemaOrThrow("aten::var", "dim")
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
Tensor var_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::var");
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
      .findSchemaOrThrow("aten::var", "names_dim")
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
Tensor where_self(const Tensor & condition, const Tensor & self, const Tensor & other) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::where", "self")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Tracer, condition, self, other);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
std::vector<Tensor> where(const Tensor & condition) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::where");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "condition", condition);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #endif
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::where", "")
      .typed<std::vector<Tensor> (const Tensor &)>();
  auto result =c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &>(op, c10::DispatchKey::Tracer, condition);
  #if !defined(PYTORCH_DISABLE_TRACING)
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  #endif
  return result;
}
Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  #if !defined(PYTORCH_DISABLE_TRACING)
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros_like");
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
      .findSchemaOrThrow("aten::zeros_like", "")
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
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl_UNBOXED("_batch_norm_impl_index_backward", &TraceType::_batch_norm_impl_index_backward);
  m.impl("_cast_Char", TORCH_FN(TraceType::_cast_Char));
  m.impl("_cast_Float", TORCH_FN(TraceType::_cast_Float));
  m.impl("_cholesky_solve_helper", TORCH_FN(TraceType::_cholesky_solve_helper));
  m.impl_UNBOXED("_convolution_nogroup", &TraceType::_convolution_nogroup);
  m.impl("_copy_from", TORCH_FN(TraceType::_copy_from));
  m.impl("_ctc_loss_backward", TORCH_FN(TraceType::_ctc_loss_backward));
  m.impl("_cudnn_ctc_loss", TORCH_FN(TraceType::_cudnn_ctc_loss));
  m.impl("_cufft_set_plan_cache_max_size", TORCH_FN(TraceType::_cufft_set_plan_cache_max_size));
  m.impl_UNBOXED("_cummin_helper", &TraceType::_cummin_helper);
  m.impl("_euclidean_dist", TORCH_FN(TraceType::_euclidean_dist));
  m.impl_UNBOXED("_index_copy_", &TraceType::_index_copy_);
  m.impl("_inverse_helper", TORCH_FN(TraceType::_inverse_helper));
  m.impl("_masked_scale", TORCH_FN(TraceType::_masked_scale));
  m.impl("_nnpack_available", TORCH_FN(TraceType::_nnpack_available));
  m.impl("_pdist_backward", TORCH_FN(TraceType::_pdist_backward));
  m.impl("_s_where", TORCH_FN(TraceType::_s_where));
  m.impl_UNBOXED("_sample_dirichlet", &TraceType::_sample_dirichlet);
  m.impl_UNBOXED("_sobol_engine_scramble_", &TraceType::_sobol_engine_scramble_);
  m.impl("_sparse_addmm", TORCH_FN(TraceType::_sparse_addmm));
  m.impl_UNBOXED("_sparse_coo_tensor_unsafe", &TraceType::_sparse_coo_tensor_unsafe);
  m.impl_UNBOXED("_sparse_coo_tensor_with_dims_and_tensors", &TraceType::_sparse_coo_tensor_with_dims_and_tensors);
  m.impl_UNBOXED("_sparse_softmax.int", &TraceType::_sparse_softmax_int);
  m.impl_UNBOXED("_sparse_softmax.Dimname", &TraceType::_sparse_softmax_Dimname);
  m.impl("_sparse_softmax", TORCH_FN(TraceType::_sparse_softmax));
  m.impl_UNBOXED("_standard_gamma", &TraceType::_standard_gamma);
  m.impl("_symeig_helper", TORCH_FN(TraceType::_symeig_helper));
  m.impl_UNBOXED("_thnn_fused_lstm_cell", &TraceType::_thnn_fused_lstm_cell);
  m.impl("_triangular_solve_helper", TORCH_FN(TraceType::_triangular_solve_helper));
  m.impl_UNBOXED("abs.out", &TraceType::abs_out_out);
  m.impl("adaptive_max_pool2d", TORCH_FN(TraceType::adaptive_max_pool2d));
  m.impl_UNBOXED("adaptive_max_pool2d_backward.grad_input", &TraceType::adaptive_max_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("adaptive_max_pool3d.out", &TraceType::adaptive_max_pool3d_out_out);
  m.impl("addcdiv", TORCH_FN(TraceType::addcdiv));
  m.impl_UNBOXED("addcdiv_", &TraceType::addcdiv_);
  m.impl_UNBOXED("addcmul.out", &TraceType::addcmul_out_out);
  m.impl_UNBOXED("addmm.out", &TraceType::addmm_out_out);
  m.impl("addmv", TORCH_FN(TraceType::addmv));
  m.impl_UNBOXED("addmv_", &TraceType::addmv_);
  m.impl("alpha_dropout", TORCH_FN(TraceType::alpha_dropout));
  m.impl_UNBOXED("alpha_dropout_", &TraceType::alpha_dropout_);
  m.impl("angle", TORCH_FN(TraceType::angle));
  m.impl_UNBOXED("atan2.out", &TraceType::atan2_out_out);
  m.impl_UNBOXED("atan.out", &TraceType::atan_out_out);
  m.impl("atanh", TORCH_FN(TraceType::atanh));
  m.impl_UNBOXED("atanh_", &TraceType::atanh_);
  m.impl("avg_pool1d", TORCH_FN(TraceType::avg_pool1d));
  m.impl_UNBOXED("avg_pool2d.out", &TraceType::avg_pool2d_out_out);
  m.impl_UNBOXED("batch_norm_backward_elemt", &TraceType::batch_norm_backward_elemt);
  m.impl_UNBOXED("batch_norm_gather_stats_with_counts", &TraceType::batch_norm_gather_stats_with_counts);
  m.impl_UNBOXED("binary_cross_entropy", &TraceType::binary_cross_entropy);
  m.impl_UNBOXED("binary_cross_entropy_backward.grad_input", &TraceType::binary_cross_entropy_backward_out_grad_input);
  m.impl("bitwise_or.Scalar", TORCH_FN(TraceType::bitwise_or_Scalar));
  m.impl("bitwise_or.Tensor", TORCH_FN(TraceType::bitwise_or_Tensor));
  m.impl_UNBOXED("bitwise_or_.Scalar", &TraceType::bitwise_or__Scalar);
  m.impl_UNBOXED("bitwise_or_.Tensor", &TraceType::bitwise_or__Tensor);
  m.impl("bitwise_xor.Scalar", TORCH_FN(TraceType::bitwise_xor_Scalar));
  m.impl("bitwise_xor.Tensor", TORCH_FN(TraceType::bitwise_xor_Tensor));
  m.impl_UNBOXED("bitwise_xor_.Scalar", &TraceType::bitwise_xor__Scalar);
  m.impl_UNBOXED("bitwise_xor_.Tensor", &TraceType::bitwise_xor__Tensor);
  m.impl_UNBOXED("blackman_window", &TraceType::blackman_window);
  m.impl_UNBOXED("blackman_window.periodic", &TraceType::blackman_window_periodic);
  m.impl("broadcast_tensors", TORCH_FN(TraceType::broadcast_tensors));
  m.impl_UNBOXED("cholesky_inverse.out", &TraceType::cholesky_inverse_out_out);
  m.impl("cholesky_solve", TORCH_FN(TraceType::cholesky_solve));
  m.impl("clamp_min", TORCH_FN(TraceType::clamp_min));
  m.impl_UNBOXED("clamp_min_", &TraceType::clamp_min_);
  m.impl_UNBOXED("clamp.out", &TraceType::clamp_out_out);
  m.impl_UNBOXED("clone", &TraceType::clone);
  m.impl_UNBOXED("contiguous", &TraceType::contiguous);
  m.impl_UNBOXED("conv3d", &TraceType::conv3d);
  m.impl_UNBOXED("conv_transpose2d.input", &TraceType::conv_transpose2d_input);
  m.impl_UNBOXED("convolution_overrideable", &TraceType::convolution_overrideable);
  m.impl_UNBOXED("cos.out", &TraceType::cos_out_out);
  m.impl("cosh", TORCH_FN(TraceType::cosh));
  m.impl_UNBOXED("cosh_", &TraceType::cosh_);
  m.impl("cross", TORCH_FN(TraceType::cross));
  m.impl_UNBOXED("cudnn_batch_norm", &TraceType::cudnn_batch_norm);
  m.impl("cudnn_convolution_transpose_backward", TORCH_FN(TraceType::cudnn_convolution_transpose_backward));
  m.impl("cudnn_convolution_transpose_backward_input", TORCH_FN(TraceType::cudnn_convolution_transpose_backward_input));
  m.impl("cudnn_grid_sampler", TORCH_FN(TraceType::cudnn_grid_sampler));
  m.impl("cudnn_is_acceptable", TORCH_FN(TraceType::cudnn_is_acceptable));
  m.impl("cummin", TORCH_FN(TraceType::cummin));
  m.impl_UNBOXED("cummin.dimname", &TraceType::cummin_dimname);
  m.impl_UNBOXED("deg2rad.out", &TraceType::deg2rad_out_out);
  m.impl("diag_embed", TORCH_FN(TraceType::diag_embed));
  m.impl_UNBOXED("diag.out", &TraceType::diag_out_out);
  m.impl_UNBOXED("digamma.out", &TraceType::digamma_out_out);
  m.impl("eig", TORCH_FN(TraceType::eig));
  m.impl_UNBOXED("empty.names", &TraceType::empty_names);
  m.impl_UNBOXED("empty.memory_format", &TraceType::empty_memory_format);
  m.impl("eq.Scalar", TORCH_FN(TraceType::eq_Scalar));
  m.impl("eq.Tensor", TORCH_FN(TraceType::eq_Tensor));
  m.impl_UNBOXED("eq_.Scalar", &TraceType::eq__Scalar);
  m.impl_UNBOXED("eq_.Tensor", &TraceType::eq__Tensor);
  m.impl_UNBOXED("erfinv.out", &TraceType::erfinv_out_out);
  m.impl("fake_quantize_per_tensor_affine", TORCH_FN(TraceType::fake_quantize_per_tensor_affine));
  m.impl("fbgemm_pack_quantized_matrix", TORCH_FN(TraceType::fbgemm_pack_quantized_matrix));
  m.impl("fbgemm_pack_quantized_matrix.KN", TORCH_FN(TraceType::fbgemm_pack_quantized_matrix_KN));
  m.impl("feature_dropout", TORCH_FN(TraceType::feature_dropout));
  m.impl_UNBOXED("feature_dropout_", &TraceType::feature_dropout_);
  m.impl_UNBOXED("fill_diagonal_", &TraceType::fill_diagonal_);
  m.impl("fliplr", TORCH_FN(TraceType::fliplr));
  m.impl("flipud", TORCH_FN(TraceType::flipud));
  m.impl_UNBOXED("floor_divide.out", &TraceType::floor_divide_out_out);
  m.impl("fractional_max_pool3d_backward", TORCH_FN(TraceType::fractional_max_pool3d_backward));
  m.impl_UNBOXED("full.out", &TraceType::full_out_out);
  m.impl_UNBOXED("gather.out", &TraceType::gather_out_out);
  m.impl_UNBOXED("gather.dimname_out", &TraceType::gather_out_dimname_out);
  m.impl("ge.Scalar", TORCH_FN(TraceType::ge_Scalar));
  m.impl("ge.Tensor", TORCH_FN(TraceType::ge_Tensor));
  m.impl_UNBOXED("ge_.Scalar", &TraceType::ge__Scalar);
  m.impl_UNBOXED("ge_.Tensor", &TraceType::ge__Tensor);
  m.impl("gelu", TORCH_FN(TraceType::gelu));
  m.impl_UNBOXED("geometric_", &TraceType::geometric_);
  m.impl("glu_backward", TORCH_FN(TraceType::glu_backward));
  m.impl("grid_sampler_2d_backward", TORCH_FN(TraceType::grid_sampler_2d_backward));
  m.impl("grid_sampler_3d", TORCH_FN(TraceType::grid_sampler_3d));
  m.impl("gru.input", TORCH_FN(TraceType::gru_input));
  m.impl("gru.data", TORCH_FN(TraceType::gru_data));
  m.impl("gt.Scalar", TORCH_FN(TraceType::gt_Scalar));
  m.impl("gt.Tensor", TORCH_FN(TraceType::gt_Tensor));
  m.impl_UNBOXED("gt_.Scalar", &TraceType::gt__Scalar);
  m.impl_UNBOXED("gt_.Tensor", &TraceType::gt__Tensor);
  m.impl("hardsigmoid_backward", TORCH_FN(TraceType::hardsigmoid_backward));
  m.impl("hardswish_backward", TORCH_FN(TraceType::hardswish_backward));
  m.impl("hinge_embedding_loss", TORCH_FN(TraceType::hinge_embedding_loss));
  m.impl("histc", TORCH_FN(TraceType::histc));
  m.impl("hspmm", TORCH_FN(TraceType::hspmm));
  m.impl("imag", TORCH_FN(TraceType::imag));
  m.impl("index_copy", TORCH_FN(TraceType::index_copy));
  m.impl_UNBOXED("index_copy.dimname", &TraceType::index_copy_dimname);
  m.impl_UNBOXED("index_copy_", &TraceType::index_copy_);
  m.impl_UNBOXED("index_copy_.dimname", &TraceType::index_copy__dimname);
  m.impl("index_fill.int_Scalar", TORCH_FN(TraceType::index_fill_int_Scalar));
  m.impl("index_fill.int_Tensor", TORCH_FN(TraceType::index_fill_int_Tensor));
  m.impl_UNBOXED("index_fill.Dimname_Scalar", &TraceType::index_fill_Dimname_Scalar);
  m.impl_UNBOXED("index_fill.Dimname_Tensor", &TraceType::index_fill_Dimname_Tensor);
  m.impl_UNBOXED("index_fill_.int_Scalar", &TraceType::index_fill__int_Scalar);
  m.impl_UNBOXED("index_fill_.int_Tensor", &TraceType::index_fill__int_Tensor);
  m.impl_UNBOXED("index_fill_.Dimname_Scalar", &TraceType::index_fill__Dimname_Scalar);
  m.impl_UNBOXED("index_fill_.Dimname_Tensor", &TraceType::index_fill__Dimname_Tensor);
  m.impl("inverse", TORCH_FN(TraceType::inverse));
  m.impl("irfft", TORCH_FN(TraceType::irfft));
  m.impl("is_nonzero", TORCH_FN(TraceType::is_nonzero));
  m.impl("is_set_to", TORCH_FN(TraceType::is_set_to));
  m.impl("is_signed", TORCH_FN(TraceType::is_signed));
  m.impl("isclose", TORCH_FN(TraceType::isclose));
  m.impl("isfinite", TORCH_FN(TraceType::isfinite));
  m.impl_UNBOXED("istft", &TraceType::istft);
  m.impl("kl_div_backward", TORCH_FN(TraceType::kl_div_backward));
  m.impl("le.Scalar", TORCH_FN(TraceType::le_Scalar));
  m.impl("le.Tensor", TORCH_FN(TraceType::le_Tensor));
  m.impl_UNBOXED("le_.Scalar", &TraceType::le__Scalar);
  m.impl_UNBOXED("le_.Tensor", &TraceType::le__Tensor);
  m.impl("leaky_relu", TORCH_FN(TraceType::leaky_relu));
  m.impl_UNBOXED("leaky_relu_", &TraceType::leaky_relu_);
  m.impl_UNBOXED("lerp.Scalar_out", &TraceType::lerp_out_Scalar_out);
  m.impl_UNBOXED("lerp.Tensor_out", &TraceType::lerp_out_Tensor_out);
  m.impl("log10", TORCH_FN(TraceType::log10));
  m.impl_UNBOXED("log10_", &TraceType::log10_);
  m.impl_UNBOXED("log_sigmoid.out", &TraceType::log_sigmoid_out_out);
  m.impl("logdet", TORCH_FN(TraceType::logdet));
  m.impl_UNBOXED("lstm_cell", &TraceType::lstm_cell);
  m.impl("lt.Scalar", TORCH_FN(TraceType::lt_Scalar));
  m.impl("lt.Tensor", TORCH_FN(TraceType::lt_Tensor));
  m.impl_UNBOXED("lt_.Scalar", &TraceType::lt__Scalar);
  m.impl_UNBOXED("lt_.Tensor", &TraceType::lt__Tensor);
  m.impl("masked_select", TORCH_FN(TraceType::masked_select));
  m.impl("matrix_rank.tol", TORCH_FN(TraceType::matrix_rank_tol));
  m.impl("matrix_rank", TORCH_FN(TraceType::matrix_rank));
  m.impl("max_pool3d", TORCH_FN(TraceType::max_pool3d));
  m.impl("max_pool3d_with_indices_backward", TORCH_FN(TraceType::max_pool3d_with_indices_backward));
  m.impl_UNBOXED("max_unpool2d.out", &TraceType::max_unpool2d_out_out);
  m.impl("min.dim", TORCH_FN(TraceType::min_dim));
  m.impl_UNBOXED("min.names_dim", &TraceType::min_names_dim);
  m.impl("min.other", TORCH_FN(TraceType::min_other));
  m.impl("min", TORCH_FN(TraceType::min));
  m.impl_UNBOXED("miopen_convolution", &TraceType::miopen_convolution);
  m.impl("miopen_convolution_transpose_backward_weight", TORCH_FN(TraceType::miopen_convolution_transpose_backward_weight));
  m.impl("mkldnn_convolution_backward_weights", TORCH_FN(TraceType::mkldnn_convolution_backward_weights));
  m.impl_UNBOXED("mkldnn_linear", &TraceType::mkldnn_linear);
  m.impl("mse_loss", TORCH_FN(TraceType::mse_loss));
  m.impl_UNBOXED("mse_loss_backward.grad_input", &TraceType::mse_loss_backward_out_grad_input);
  m.impl("mul.Tensor", TORCH_FN(TraceType::mul_Tensor));
  m.impl("mul.Scalar", TORCH_FN(TraceType::mul_Scalar));
  m.impl_UNBOXED("mul_.Tensor", &TraceType::mul__Tensor);
  m.impl_UNBOXED("mul_.Scalar", &TraceType::mul__Scalar);
  m.impl("multilabel_margin_loss", TORCH_FN(TraceType::multilabel_margin_loss));
  m.impl_UNBOXED("multilabel_margin_loss_backward.grad_input", &TraceType::multilabel_margin_loss_backward_out_grad_input);
  m.impl_UNBOXED("multinomial.out", &TraceType::multinomial_out_out);
  m.impl_UNBOXED("native_batch_norm.out", &TraceType::native_batch_norm_out_out);
  m.impl_UNBOXED("native_group_norm_backward", &TraceType::native_group_norm_backward);
  m.impl("neg", TORCH_FN(TraceType::neg));
  m.impl_UNBOXED("neg_", &TraceType::neg_);
  m.impl_UNBOXED("new_empty", &TraceType::new_empty);
  m.impl_UNBOXED("nll_loss2d.out", &TraceType::nll_loss2d_out_out);
  m.impl_UNBOXED("nll_loss.out", &TraceType::nll_loss_out_out);
  m.impl("nonzero", TORCH_FN(TraceType::nonzero));
  m.impl_UNBOXED("norm.ScalarOpt_dtype", &TraceType::norm_ScalarOpt_dtype);
  m.impl("norm.Scalar", TORCH_FN(TraceType::norm_Scalar));
  m.impl_UNBOXED("norm.ScalarOpt_dim_dtype", &TraceType::norm_ScalarOpt_dim_dtype);
  m.impl("norm.ScalarOpt_dim", TORCH_FN(TraceType::norm_ScalarOpt_dim));
  m.impl_UNBOXED("norm.names_ScalarOpt_dim_dtype", &TraceType::norm_names_ScalarOpt_dim_dtype);
  m.impl_UNBOXED("norm.names_ScalarOpt_dim", &TraceType::norm_names_ScalarOpt_dim);
  m.impl_UNBOXED("normal.Tensor_float", &TraceType::normal_Tensor_float);
  m.impl_UNBOXED("normal.float_Tensor", &TraceType::normal_float_Tensor);
  m.impl_UNBOXED("normal.Tensor_Tensor", &TraceType::normal_Tensor_Tensor);
  m.impl_UNBOXED("normal.float_float", &TraceType::normal_float_float);
  m.impl_UNBOXED("normal_", &TraceType::normal_);
  m.impl_UNBOXED("nuclear_norm.out", &TraceType::nuclear_norm_out_out);
  m.impl_UNBOXED("nuclear_norm.dim_out", &TraceType::nuclear_norm_out_dim_out);
  m.impl_UNBOXED("orgqr.out", &TraceType::orgqr_out_out);
  m.impl_UNBOXED("poisson", &TraceType::poisson);
  m.impl("poisson_nll_loss", TORCH_FN(TraceType::poisson_nll_loss));
  m.impl_UNBOXED("promote_types", &TraceType::promote_types);
  m.impl("q_per_channel_scales", TORCH_FN(TraceType::q_per_channel_scales));
  m.impl_UNBOXED("quantized_batch_norm", &TraceType::quantized_batch_norm);
  m.impl("quantized_max_pool2d", TORCH_FN(TraceType::quantized_max_pool2d));
  m.impl_UNBOXED("rad2deg.out", &TraceType::rad2deg_out_out);
  m.impl_UNBOXED("range.out", &TraceType::range_out_out);
  m.impl_UNBOXED("reciprocal.out", &TraceType::reciprocal_out_out);
  m.impl_UNBOXED("reflection_pad1d.out", &TraceType::reflection_pad1d_out_out);
  m.impl("renorm", TORCH_FN(TraceType::renorm));
  m.impl_UNBOXED("renorm_", &TraceType::renorm_);
  m.impl("replication_pad3d_backward", TORCH_FN(TraceType::replication_pad3d_backward));
  m.impl("reshape", TORCH_FN(TraceType::reshape));
  m.impl("rfft", TORCH_FN(TraceType::rfft));
  m.impl("rnn_relu.input", TORCH_FN(TraceType::rnn_relu_input));
  m.impl("rnn_relu.data", TORCH_FN(TraceType::rnn_relu_data));
  m.impl_UNBOXED("rrelu", &TraceType::rrelu);
  m.impl_UNBOXED("rrelu_", &TraceType::rrelu_);
  m.impl_UNBOXED("rrelu_with_noise.out", &TraceType::rrelu_with_noise_out_out);
  m.impl("rsub.Tensor", TORCH_FN(TraceType::rsub_Tensor));
  m.impl("rsub.Scalar", TORCH_FN(TraceType::rsub_Scalar));
  m.impl("sigmoid_backward", TORCH_FN(TraceType::sigmoid_backward));
  m.impl_UNBOXED("sin.out", &TraceType::sin_out_out);
  m.impl("sinh", TORCH_FN(TraceType::sinh));
  m.impl_UNBOXED("sinh_", &TraceType::sinh_);
  m.impl("slogdet", TORCH_FN(TraceType::slogdet));
  m.impl_UNBOXED("slow_conv3d_forward", &TraceType::slow_conv3d_forward);
  m.impl("slow_conv_dilated2d_backward", TORCH_FN(TraceType::slow_conv_dilated2d_backward));
  m.impl_UNBOXED("slow_conv_dilated3d", &TraceType::slow_conv_dilated3d);
  m.impl("slow_conv_transpose3d_backward.output_mask", TORCH_FN(TraceType::slow_conv_transpose3d_backward_output_mask));
  m.impl("softshrink", TORCH_FN(TraceType::softshrink));
  m.impl_UNBOXED("softshrink_backward.grad_input", &TraceType::softshrink_backward_out_grad_input);
  m.impl("stack", TORCH_FN(TraceType::stack));
  m.impl_UNBOXED("stft", &TraceType::stft);
  m.impl_UNBOXED("sub.out", &TraceType::sub_out_out);
  m.impl("symeig", TORCH_FN(TraceType::symeig));
  m.impl("tensordot", TORCH_FN(TraceType::tensordot));
  m.impl("thnn_conv2d_backward.output_mask", TORCH_FN(TraceType::thnn_conv2d_backward_output_mask));
  m.impl_UNBOXED("thnn_conv2d_forward.output", &TraceType::thnn_conv2d_forward_out_output);
  m.impl("threshold_backward", TORCH_FN(TraceType::threshold_backward));
  m.impl("to_dense", TORCH_FN(TraceType::to_dense));
  m.impl("triangular_solve", TORCH_FN(TraceType::triangular_solve));
  m.impl("triplet_margin_loss", TORCH_FN(TraceType::triplet_margin_loss));
  m.impl_UNBOXED("unfold_backward", &TraceType::unfold_backward);
  m.impl("unique_dim_consecutive", TORCH_FN(TraceType::unique_dim_consecutive));
  m.impl("upsample_bilinear2d", TORCH_FN(TraceType::upsample_bilinear2d));
  m.impl_UNBOXED("upsample_bilinear2d_backward.grad_input", &TraceType::upsample_bilinear2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_linear1d.out", &TraceType::upsample_linear1d_out_out);
  m.impl("upsample_nearest1d", TORCH_FN(TraceType::upsample_nearest1d));
  m.impl_UNBOXED("upsample_nearest1d_backward.grad_input", &TraceType::upsample_nearest1d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest2d.out", &TraceType::upsample_nearest2d_out_out);
  m.impl("var", TORCH_FN(TraceType::var));
  m.impl("var.dim", TORCH_FN(TraceType::var_dim));
  m.impl_UNBOXED("var.names_dim", &TraceType::var_names_dim);
  m.impl("where.self", TORCH_FN(TraceType::where_self));
  m.impl("where", TORCH_FN(TraceType::where));
  m.impl_UNBOXED("zeros_like", &TraceType::zeros_like);;
}

}  // namespace

} // namespace torch
