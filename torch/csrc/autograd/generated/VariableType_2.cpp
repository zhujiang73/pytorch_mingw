#include "torch/csrc/autograd/VariableTypeUtils.h"

// @generated from tools/autograd\templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

Tensor VariableType::__and__(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("__and__", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__and__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__and__(self_, other);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::__and__(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__and__", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__and__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::__and__(self_, other_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::__iand__(Tensor & self, Scalar other) {
  RECORD_FUNCTION("__iand__", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__iand__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__iand__(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::__iand__(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("__iand__", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::__iand__");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.__iand__(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::_adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) {
  RECORD_FUNCTION("_adaptive_avg_pool2d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AdaptiveAvgPool2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool2DBackwardBackward>(new AdaptiveAvgPool2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_adaptive_avg_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_adaptive_avg_pool2d_backward(grad_output_, self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cast_Long(const Tensor & self, bool non_blocking) {
  RECORD_FUNCTION("_cast_Long", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_cast_Long");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_cast_Long(self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_cat(TensorList tensors, int64_t dim) {
  RECORD_FUNCTION("_cat", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cat"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (Tensor tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cat(tensors_, dim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("_convolution_double_backward", std::vector<c10::IValue>({ggI, ggW, ggb, gO, weight, self}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_convolution_double_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "ggI", ggI);
    jit::tracer::addInputs(node, "ggW", ggW);
    jit::tracer::addInputs(node, "ggb", ggb);
    jit::tracer::addInputs(node, "gO", gO);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
void VariableType::_cufft_clear_plan_cache(int64_t device_index) {
  RECORD_FUNCTION("_cufft_clear_plan_cache", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  TypeDefault::_cufft_clear_plan_cache(device_index);
}
int64_t VariableType::_cufft_get_plan_cache_max_size(int64_t device_index) {
  RECORD_FUNCTION("_cufft_get_plan_cache_max_size", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::_cufft_get_plan_cache_max_size(device_index);
  return result;
}
Tensor VariableType::_cumprod(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("_cumprod", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cumprod"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_cumprod(self_, dim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::_debug_has_internal_overlap(const Tensor & self) {
  RECORD_FUNCTION("_debug_has_internal_overlap", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::_debug_has_internal_overlap(self);
  return result;
}
Tensor VariableType::_dequantize_linear(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) {
  RECORD_FUNCTION("_dequantize_linear", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_dequantize_linear"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_dequantize_linear");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_dequantize_linear(self_, scale, zero_point, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::_dimI(const Tensor & self) {
  RECORD_FUNCTION("_dimI", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_._dimI();
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor VariableType::_empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) {
  RECORD_FUNCTION("_empty_affine_quantized", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_empty_affine_quantized");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_empty_affine_quantized(size, options_, scale, zero_point, memory_format);
  })();
  auto result = as_variable(tmp);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes) {
  RECORD_FUNCTION("_fft_with_size", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FftWithSizeBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FftWithSizeBackward>(new FftWithSizeBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->signal_ndim = signal_ndim;
    grad_fn->complex_input = complex_input;
    grad_fn->complex_output = complex_output;
    grad_fn->inverse = inverse;
    grad_fn->checked_signal_sizes = checked_signal_sizes.vec();
    grad_fn->normalized = normalized;
    grad_fn->onesided = onesided;
    grad_fn->output_sizes = output_sizes.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_fft_with_size");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "signal_ndim", signal_ndim);
    jit::tracer::addInputs(node, "complex_input", complex_input);
    jit::tracer::addInputs(node, "complex_output", complex_output);
    jit::tracer::addInputs(node, "inverse", inverse);
    jit::tracer::addInputs(node, "checked_signal_sizes", checked_signal_sizes);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "output_sizes", output_sizes);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_fft_with_size(self_, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::_log_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  RECORD_FUNCTION("_log_softmax", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogSoftmaxBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogSoftmaxBackward>(new LogSoftmaxBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_log_softmax(self_, dim, half_to_float);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_lu_with_info(const Tensor & self, bool pivot, bool check_errors) {
  RECORD_FUNCTION("_lu_with_info", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LuWithInfoBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LuWithInfoBackward>(new LuWithInfoBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_lu_with_info");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_lu_with_info(self_, pivot, check_errors);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::_multinomial_alias_draw(const Tensor & J, const Tensor & q, int64_t num_samples, Generator * generator) {
  RECORD_FUNCTION("_multinomial_alias_draw", std::vector<c10::IValue>({J, q}), Function::peek_at_next_sequence_nr());
  auto& J_ = unpack(J, "J", 0);
  auto& q_ = unpack(q, "q", 1);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( J, q )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_multinomial_alias_draw"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( J, q ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_multinomial_alias_draw");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "J", J);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "num_samples", num_samples);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> J__storage_saved =
    J_.has_storage() ? c10::optional<Storage>(J_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> J__impl_saved;
  if (J_.defined()) J__impl_saved = J_.getIntrusivePtr();
  c10::optional<Storage> q__storage_saved =
    q_.has_storage() ? c10::optional<Storage>(q_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> q__impl_saved;
  if (q_.defined()) q__impl_saved = q_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_multinomial_alias_draw(J_, q_, num_samples, generator);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (J__storage_saved.has_value())
    AT_ASSERT(J__storage_saved.value().is_alias_of(J_.storage()));
  if (J__impl_saved) AT_ASSERT(J__impl_saved == J_.getIntrusivePtr());
  if (q__storage_saved.has_value())
    AT_ASSERT(q__storage_saved.value().is_alias_of(q_.storage()));
  if (q__impl_saved) AT_ASSERT(q__impl_saved == q_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::_nnpack_spatial_convolution_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward", std::vector<c10::IValue>({input, grad_output, weight}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::_nnpack_spatial_convolution_backward(input, grad_output, weight, padding, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::_nnpack_spatial_convolution_backward_input(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding) {
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward_input", std::vector<c10::IValue>({input, grad_output, weight}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_nnpack_spatial_convolution_backward_input");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_nnpack_spatial_convolution_backward_input(input, grad_output, weight, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t VariableType::_nnz(const Tensor & self) {
  RECORD_FUNCTION("_nnz", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_._nnz();
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
std::tuple<Tensor,Tensor> VariableType::_pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first) {
  RECORD_FUNCTION("_pack_padded_sequence", std::vector<c10::IValue>({input, lengths}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& lengths_ = unpack(lengths, "lengths", 1);
  check_no_requires_grad(lengths, "lengths");
  std::shared_ptr<PackPaddedSequenceBackward> grad_fn;
  if (compute_requires_grad( input )) {
    grad_fn = std::shared_ptr<PackPaddedSequenceBackward>(new PackPaddedSequenceBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->batch_first = batch_first;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pack_padded_sequence");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "lengths", lengths);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> lengths__storage_saved =
    lengths_.has_storage() ? c10::optional<Storage>(lengths_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> lengths__impl_saved;
  if (lengths_.defined()) lengths__impl_saved = lengths_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_pack_padded_sequence(input_, lengths_, batch_first);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (lengths__storage_saved.has_value())
    AT_ASSERT(lengths__storage_saved.value().is_alias_of(lengths_.storage()));
  if (lengths__impl_saved) AT_ASSERT(lengths__impl_saved == lengths_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::_pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) {
  RECORD_FUNCTION("_pad_packed_sequence", std::vector<c10::IValue>({data, batch_sizes, padding_value}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_pad_packed_sequence");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "padding_value", padding_value);
    jit::tracer::addInputs(node, "total_length", total_length);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1) = TypeDefault::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::_qr_helper(const Tensor & self, bool some) {
  RECORD_FUNCTION("_qr_helper", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_qr_helper"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_qr_helper");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::_qr_helper(self_, some);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::_reshape_from_tensor(const Tensor & self, const Tensor & shape) {
  RECORD_FUNCTION("_reshape_from_tensor", std::vector<c10::IValue>({self, shape}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_reshape_from_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_reshape_from_tensor(self, shape);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sobol_engine_ff_(Tensor & self, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
  RECORD_FUNCTION("_sobol_engine_ff_", std::vector<c10::IValue>({self, sobolstate}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_ff");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_ff_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "sobolstate", sobolstate);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "num_generated", num_generated);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sobol_engine_ff_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::_sobol_engine_ff_(self, n, sobolstate, dimension, num_generated);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_sobol_engine_initialize_state_(Tensor & self, int64_t dimension) {
  RECORD_FUNCTION("_sobol_engine_initialize_state_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_initialize_state");
    } else {
      op_name = jit::Symbol::fromQualString("aten::_sobol_engine_initialize_state_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sobol_engine_initialize_state_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::_sobol_engine_initialize_state_(self, dimension);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::_sparse_dense_add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("_sparse_dense_add_out", std::vector<c10::IValue>({out, self, other, alpha}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_dense_add");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_sparse_dense_add");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_dense_add");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_dense_add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_sparse_dense_add_out(out_, self_, other_, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::_sparse_mm(const Tensor & sparse, const Tensor & dense) {
  RECORD_FUNCTION("_sparse_mm", std::vector<c10::IValue>({sparse, dense}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "dense", dense);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_sparse_mm(sparse, dense);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::_sparse_mul_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("_sparse_mul_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("_sparse_mul");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_sparse_mul");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_mul");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_mul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_sparse_mul_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::_sparse_mul_scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  RECORD_FUNCTION("_sparse_mul_scalar_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_sparse_mul_scalar");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_sparse_mul_scalar");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_sparse_mul_scalar");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_mul_scalar_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::_sparse_mul_scalar_out(out_, self_, other);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::_weight_norm(const Tensor & v, const Tensor & g, int64_t dim) {
  RECORD_FUNCTION("_weight_norm", std::vector<c10::IValue>({v, g}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::_weight_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::_weight_norm(v, g, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::acos(const Tensor & self) {
  RECORD_FUNCTION("acos", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AcosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AcosBackward>(new AcosBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::acos");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::acos(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::acos_(Tensor & self) {
  RECORD_FUNCTION("acos_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AcosBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AcosBackward>(new AcosBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::acos");
    } else {
      op_name = jit::Symbol::fromQualString("aten::acos_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::acos_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size) {
  RECORD_FUNCTION("adaptive_avg_pool3d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AdaptiveAvgPool3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackward>(new AdaptiveAvgPool3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::adaptive_avg_pool3d(self_, output_size);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
  RECORD_FUNCTION("adaptive_avg_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::adaptive_avg_pool3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::adaptive_avg_pool3d_backward_out(grad_input_, grad_output_, self_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::add(const Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("add", std::vector<c10::IValue>({self, other, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<AddBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::add(self_, other_, alpha);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::add(const Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("add", std::vector<c10::IValue>({self, other, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::add(self_, other, alpha);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::add_(Tensor & self, const Tensor & other, Scalar alpha) {
  RECORD_FUNCTION("add_", std::vector<c10::IValue>({self, other, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<AddBackward0> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->alpha = alpha;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.add_(other_, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::add_(Tensor & self, Scalar other, Scalar alpha) {
  RECORD_FUNCTION("add_", std::vector<c10::IValue>({self, other, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AddBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::add");
    } else {
      op_name = jit::Symbol::fromQualString("aten::add_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.add_(other, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::addbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("addbmm_out", std::vector<c10::IValue>({out, self, batch1, batch2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& batch1_ = unpack(batch1, "batch1", 2);
  auto& batch2_ = unpack(batch2, "batch2", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("addbmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addbmm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::addbmm");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("addbmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::addbmm_out(out_, self_, batch1_, batch2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::affine_grid_generator_backward(const Tensor & grad, IntArrayRef size) {
  RECORD_FUNCTION("affine_grid_generator_backward", std::vector<c10::IValue>({grad}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::affine_grid_generator_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::affine_grid_generator_backward(grad, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::alias(const Tensor & self) {
  RECORD_FUNCTION("alias", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AliasBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AliasBackward>(new AliasBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::alias");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::alias(self_);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::all_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("all_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("all");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("all");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::all");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("all_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::all_out(out_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
bool VariableType::allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
  RECORD_FUNCTION("allclose", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::allclose(self, other, rtol, atol, equal_nan);
  return result;
}
Tensor & VariableType::any_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("any_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("any");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("any");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::any");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("any_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::any_out(out_, self_, dim, keepdim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::arange(Scalar end, const TensorOptions & options) {
  RECORD_FUNCTION("arange", std::vector<c10::IValue>({end}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::arange");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::arange(end, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::arange(Scalar start, Scalar end, const TensorOptions & options) {
  RECORD_FUNCTION("arange", std::vector<c10::IValue>({start, end}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::arange(start, end, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::arange(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  RECORD_FUNCTION("arange", std::vector<c10::IValue>({start, end, step}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::arange(start, end, step, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::asin(const Tensor & self) {
  RECORD_FUNCTION("asin", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<AsinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsinBackward>(new AsinBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::asin");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::asin(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::asin_(Tensor & self) {
  RECORD_FUNCTION("asin_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<AsinBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<AsinBackward>(new AsinBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::asin");
    } else {
      op_name = jit::Symbol::fromQualString("aten::asin_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::asin_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  RECORD_FUNCTION("avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<AvgPool3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<AvgPool3DBackwardBackward>(new AvgPool3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->self_info = self;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::avg_pool3d_backward(grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::batch_norm_gather_stats(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, int64_t count) {
  RECORD_FUNCTION("batch_norm_gather_stats", std::vector<c10::IValue>({input, mean, invstd, running_mean, running_var}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& invstd_ = unpack(invstd, "invstd", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input, mean, invstd, running_mean, running_var )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_gather_stats"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, mean, invstd, running_mean, running_var ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_gather_stats");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "count", count);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::batch_norm_gather_stats(input_, mean_, invstd_, running_mean_, running_var_, momentum, eps, count);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
std::tuple<Tensor,Tensor> VariableType::batch_norm_stats(const Tensor & input, double eps) {
  RECORD_FUNCTION("batch_norm_stats", std::vector<c10::IValue>({input}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( input )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_stats"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input ));
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::batch_norm_stats");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::batch_norm_stats(input_, eps);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor VariableType::bernoulli(const Tensor & self, Generator * generator) {
  RECORD_FUNCTION("bernoulli", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<BernoulliBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BernoulliBackward0>(new BernoulliBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::bernoulli(self_, generator);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::bernoulli(const Tensor & self, double p, Generator * generator) {
  RECORD_FUNCTION("bernoulli", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::bernoulli(self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::bernoulli_(Tensor & self, const Tensor & p, Generator * generator) {
  RECORD_FUNCTION("bernoulli_", std::vector<c10::IValue>({self, p}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& p_ = unpack(p, "p", 1);
  check_inplace(self);
  std::shared_ptr<BernoulliBackward1> grad_fn;
  if (compute_requires_grad( self, p )) {
    grad_fn = std::shared_ptr<BernoulliBackward1>(new BernoulliBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, p ));
    grad_fn->p_info = p;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bernoulli");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bernoulli_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> p__storage_saved =
    p_.has_storage() ? c10::optional<Storage>(p_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> p__impl_saved;
  if (p_.defined()) p__impl_saved = p_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.bernoulli_(p_, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (p__storage_saved.has_value())
    AT_ASSERT(p__storage_saved.value().is_alias_of(p_.storage()));
  if (p__impl_saved) AT_ASSERT(p__impl_saved == p_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::bernoulli_(Tensor & self, double p, Generator * generator) {
  RECORD_FUNCTION("bernoulli_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<BernoulliBackward2> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<BernoulliBackward2>(new BernoulliBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::bernoulli");
    } else {
      op_name = jit::Symbol::fromQualString("aten::bernoulli_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.bernoulli_(p, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::binary_cross_entropy_with_logits_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {
  RECORD_FUNCTION("binary_cross_entropy_with_logits_backward", std::vector<c10::IValue>({grad_output, self, target, weight, pos_weight}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::binary_cross_entropy_with_logits_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "pos_weight", pos_weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cat(TensorList tensors, int64_t dim) {
  RECORD_FUNCTION("cat", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto tensors_ = unpack(tensors, "tensors", 0);
  std::shared_ptr<CatBackward> grad_fn;
  if (compute_requires_grad( tensors )) {
    grad_fn = std::shared_ptr<CatBackward>(new CatBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->tensors_args_sizes = to_args_sizes(tensors);
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (Tensor tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cat(tensors_, dim);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::cauchy_(Tensor & self, double median, double sigma, Generator * generator) {
  RECORD_FUNCTION("cauchy_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<CauchyBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CauchyBackward>(new CauchyBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::cauchy");
    } else {
      op_name = jit::Symbol::fromQualString("aten::cauchy_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cauchy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.cauchy_(median, sigma, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::ceil_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("ceil_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("ceil");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("ceil");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ceil");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ceil_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::ceil_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::vector<Tensor> VariableType::chunk(const Tensor & self, int64_t chunks, int64_t dim) {
  RECORD_FUNCTION("chunk", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::chunk");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "chunks", chunks);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::chunk(self, chunks, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  RECORD_FUNCTION("col2im", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Col2ImBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Col2ImBackward>(new Col2ImBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::col2im(self_, output_size, kernel_size, dilation, padding, stride);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::col2im_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  RECORD_FUNCTION("col2im_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("col2im_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("col2im_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::col2im_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("col2im_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::col2im_backward_out(grad_input_, grad_output_, kernel_size, dilation, padding, stride);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  RECORD_FUNCTION("conv1d", std::vector<c10::IValue>({input, weight, bias}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::conv1d(input, weight, bias, stride, padding, dilation, groups);
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  RECORD_FUNCTION("conv_dilated3d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<ConvDilated3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ConvDilated3DBackwardBackward>(new ConvDilated3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conv_dilated3d_backward");
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
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::conv_dilated3d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
  })();
  std::tie(grad_input, grad_weight, grad_bias) = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
std::tuple<Tensor,Tensor,Tensor> VariableType::conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) {
  RECORD_FUNCTION("conv_tbc_backward", std::vector<c10::IValue>({self, input, weight, bias}), Function::peek_at_next_sequence_nr());
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::conv_tbc_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  std::tie(result0, result1, result2) = TypeDefault::conv_tbc_backward(self, input, weight, bias, pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
  RECORD_FUNCTION("cosine_embedding_loss", std::vector<c10::IValue>({input1, input2, target}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cosine_embedding_loss");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::cosine_embedding_loss(input1, input2, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  RECORD_FUNCTION("cudnn_affine_grid_generator", std::vector<c10::IValue>({theta}), Function::peek_at_next_sequence_nr());
  auto& theta_ = unpack(theta, "theta", 0);
  std::shared_ptr<CudnnAffineGridGeneratorBackward> grad_fn;
  if (compute_requires_grad( theta )) {
    grad_fn = std::shared_ptr<CudnnAffineGridGeneratorBackward>(new CudnnAffineGridGeneratorBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( theta ));
    grad_fn->N = N;
    grad_fn->C = C;
    grad_fn->H = H;
    grad_fn->W = W;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_affine_grid_generator");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "theta", theta);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> theta__storage_saved =
    theta_.has_storage() ? c10::optional<Storage>(theta_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> theta__impl_saved;
  if (theta_.defined()) theta__impl_saved = theta_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_affine_grid_generator(theta_, N, C, H, W);
  })();
  auto grid_return = as_variable(tmp);
  #ifndef NDEBUG
  if (theta__storage_saved.has_value())
    AT_ASSERT(theta__storage_saved.value().is_alias_of(theta_.storage()));
  if (theta__impl_saved) AT_ASSERT(theta__impl_saved == theta_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grid_return ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grid_return);
  }
  return grid_return;
}
Tensor VariableType::cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("cudnn_convolution", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<CudnnConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<CudnnConvolutionBackward>(new CudnnConvolutionBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_convolution(self_, weight_, bias_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cudnn_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("cudnn_convolution_transpose_backward_weight", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_weight"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::cudnn_convolution_transpose_backward_weight");
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
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cudnn_convolution_transpose_backward_weight(weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::cumprod(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("cumprod", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<CumprodBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<CumprodBackward>(new CumprodBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::cumprod(self_, dim, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::dequantize(const Tensor & self) {
  RECORD_FUNCTION("dequantize", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("dequantize"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::dequantize(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::det(const Tensor & self) {
  RECORD_FUNCTION("det", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<DetBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<DetBackward>(new DetBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::det");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::det(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::div_out(Tensor & out, const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("div_out", std::vector<c10::IValue>({out, self, other}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& other_ = unpack(other, "other", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("div");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("div");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::div");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::div_out(out_, self_, other_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::dot(const Tensor & self, const Tensor & tensor) {
  RECORD_FUNCTION("dot", std::vector<c10::IValue>({self, tensor}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  std::shared_ptr<DotBackward> grad_fn;
  if (compute_requires_grad( self, tensor )) {
    grad_fn = std::shared_ptr<DotBackward>(new DotBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, tensor ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->tensor_ = SavedVariable(tensor, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor__storage_saved =
    tensor_.has_storage() ? c10::optional<Storage>(tensor_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor__impl_saved;
  if (tensor_.defined()) tensor__impl_saved = tensor_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::dot(self_, tensor_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor__storage_saved.has_value())
    AT_ASSERT(tensor__storage_saved.value().is_alias_of(tensor_.storage()));
  if (tensor__impl_saved) AT_ASSERT(tensor__impl_saved == tensor_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::einsum(std::string equation, TensorList tensors) {
  RECORD_FUNCTION("einsum", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::einsum");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "equation", equation);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::einsum(equation, tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  RECORD_FUNCTION("elu_backward", std::vector<c10::IValue>({grad_output, alpha, scale, input_scale, output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<EluBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    grad_fn = std::shared_ptr<EluBackwardBackward>(new EluBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
    grad_fn->input_scale = input_scale;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::elu_backward(grad_output_, alpha, scale, input_scale, output_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  RECORD_FUNCTION("embedding", std::vector<c10::IValue>({weight, indices}), Function::peek_at_next_sequence_nr());
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  std::shared_ptr<EmbeddingBackward> grad_fn;
  if (compute_requires_grad( weight )) {
    grad_fn = std::shared_ptr<EmbeddingBackward>(new EmbeddingBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( weight ));
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->padding_idx = padding_idx;
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->sparse = sparse;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "sparse", sparse);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::embedding(weight_, indices_, padding_idx, scale_grad_by_freq, sparse);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  RECORD_FUNCTION("embedding_sparse_backward", std::vector<c10::IValue>({grad, indices}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::embedding_sparse_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::erf(const Tensor & self) {
  RECORD_FUNCTION("erf", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ErfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfBackward>(new ErfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::erf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::erf(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::erf_(Tensor & self) {
  RECORD_FUNCTION("erf_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<ErfBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ErfBackward>(new ErfBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::erf");
    } else {
      op_name = jit::Symbol::fromQualString("aten::erf_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erf_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::erf_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::exp_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("exp_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("exp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("exp");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::exp");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::exp_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::expand_as(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("expand_as", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::expand_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::expand_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::eye_out(Tensor & out, int64_t n) {
  RECORD_FUNCTION("eye_out", std::vector<c10::IValue>({out}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eye_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::eye_out(out_, n);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::eye_out(Tensor & out, int64_t n, int64_t m) {
  RECORD_FUNCTION("eye_out", std::vector<c10::IValue>({out}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::eye");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "m", m);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eye_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::eye_out(out_, n, m);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::fbgemm_linear_int8_weight(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) {
  RECORD_FUNCTION("fbgemm_linear_int8_weight", std::vector<c10::IValue>({input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fbgemm_linear_int8_weight");
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
  auto result = TypeDefault::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::feature_alpha_dropout(const Tensor & input, double p, bool train) {
  RECORD_FUNCTION("feature_alpha_dropout", std::vector<c10::IValue>({input}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::feature_alpha_dropout");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::feature_alpha_dropout(input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::feature_alpha_dropout_(Tensor & self, double p, bool train) {
  RECORD_FUNCTION("feature_alpha_dropout_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::feature_alpha_dropout");
    } else {
      op_name = jit::Symbol::fromQualString("aten::feature_alpha_dropout_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("feature_alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::feature_alpha_dropout_(self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::flip(const Tensor & self, IntArrayRef dims) {
  RECORD_FUNCTION("flip", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FlipBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FlipBackward>(new FlipBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dims = dims.vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::flip");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::flip(self_, dims);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::fmod(const Tensor & self, Scalar other) {
  RECORD_FUNCTION("fmod", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FmodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FmodBackward0>(new FmodBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::fmod(self_, other);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::fmod(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("fmod", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  std::shared_ptr<FmodBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<FmodBackward1>(new FmodBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::fmod(self_, other_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::fmod_(Tensor & self, Scalar other) {
  RECORD_FUNCTION("fmod_", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FmodBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FmodBackward0>(new FmodBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::fmod");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fmod_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.fmod_(other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::fmod_(Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("fmod_", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  check_inplace(self);
  std::shared_ptr<FmodBackward1> grad_fn;
  if (compute_requires_grad( self, other )) {
    grad_fn = std::shared_ptr<FmodBackward1>(new FmodBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::fmod");
    } else {
      op_name = jit::Symbol::fromQualString("aten::fmod_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.fmod_(other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::frac(const Tensor & self) {
  RECORD_FUNCTION("frac", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<FracBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FracBackward>(new FracBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frac");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::frac(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::frac_(Tensor & self) {
  RECORD_FUNCTION("frac_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<FracBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FracBackward>(new FracBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::frac");
    } else {
      op_name = jit::Symbol::fromQualString("aten::frac_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frac_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::frac_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  RECORD_FUNCTION("fractional_max_pool2d", std::vector<c10::IValue>({self, random_samples}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  check_no_requires_grad(random_samples, "random_samples");
  std::shared_ptr<FractionalMaxPool2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackward>(new FractionalMaxPool2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->output_size = output_size.vec();
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::fractional_max_pool2d(self_, kernel_size, output_size, random_samples_);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  RECORD_FUNCTION("fractional_max_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::fractional_max_pool2d_backward_out(grad_input_, grad_output_, self_, kernel_size, output_size, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::fractional_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  RECORD_FUNCTION("fractional_max_pool3d_out", std::vector<c10::IValue>({output, indices, self, random_samples}), Function::peek_at_next_sequence_nr());
  auto& output_ = unpack(output, "output", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& random_samples_ = unpack(random_samples, "random_samples", 5);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, random_samples )) {
    throw_error_out_requires_grad("fractional_max_pool3d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("fractional_max_pool3d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::fractional_max_pool3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::fractional_max_pool3d_out(output_, indices_, self_, kernel_size, output_size, random_samples_);
  }
  #ifndef NDEBUG
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  #endif
  increment_version(output);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
Tensor VariableType::frobenius_norm(const Tensor & self) {
  RECORD_FUNCTION("frobenius_norm", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::frobenius_norm(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::frobenius_norm(const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("frobenius_norm", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::frobenius_norm(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options) {
  RECORD_FUNCTION("from_file", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::from_file");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "filename", filename);
    jit::tracer::addInputs(node, "shared", shared);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_file(filename, shared, size, options_);
  })();
  auto result = as_variable(tmp);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::gels(const Tensor & self, const Tensor & A) {
  RECORD_FUNCTION("gels", std::vector<c10::IValue>({self, A}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  std::shared_ptr<GelsBackward> grad_fn;
  if (compute_requires_grad( self, A )) {
    grad_fn = std::shared_ptr<GelsBackward>(new GelsBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  Tensor solution;
  Tensor QR;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::gels");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::gels(self_, A_);
  })();
  std::tie(solution, QR) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution, QR ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, QR);
  }
  return std::make_tuple(std::move(solution), std::move(QR));
}
std::tuple<Tensor &,Tensor &> VariableType::geqrf_out(Tensor & a, Tensor & tau, const Tensor & self) {
  RECORD_FUNCTION("geqrf_out", std::vector<c10::IValue>({a, tau, self}), Function::peek_at_next_sequence_nr());
  auto& a_ = unpack(a, "a", 0);
  auto& tau_ = unpack(tau, "tau", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("geqrf");
  }
  if (compute_requires_grad( a, tau )) {
    throw_error_out_requires_grad("geqrf");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::geqrf");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tau", tau);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "a", a);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("geqrf_out", a);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> a__storage_saved =
    a_.has_storage() ? c10::optional<Storage>(a_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> a__impl_saved;
  if (a_.defined()) a__impl_saved = a_.getIntrusivePtr();
  c10::optional<Storage> tau__storage_saved =
    tau_.has_storage() ? c10::optional<Storage>(tau_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tau__impl_saved;
  if (tau_.defined()) tau__impl_saved = tau_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::geqrf_out(a_, tau_, self_);
  }
  #ifndef NDEBUG
  if (a__storage_saved.has_value())
    AT_ASSERT(a__storage_saved.value().is_alias_of(a_.storage()));
  if (a__impl_saved) AT_ASSERT(a__impl_saved == a_.getIntrusivePtr());
  if (tau__storage_saved.has_value())
    AT_ASSERT(tau__storage_saved.value().is_alias_of(tau_.storage()));
  if (tau__impl_saved) AT_ASSERT(tau__impl_saved == tau_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(a);
  increment_version(tau);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( a, tau ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, a);
    jit::tracer::addOutput(node, tau);
  }
  return std::forward_as_tuple(a, tau);
}
Tensor & VariableType::ger_out(Tensor & out, const Tensor & self, const Tensor & vec2) {
  RECORD_FUNCTION("ger_out", std::vector<c10::IValue>({out, self, vec2}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& vec2_ = unpack(vec2, "vec2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, vec2 )) {
    throw_error_out_requires_grad("ger");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("ger");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ger");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ger_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec2__storage_saved =
    vec2_.has_storage() ? c10::optional<Storage>(vec2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec2__impl_saved;
  if (vec2_.defined()) vec2__impl_saved = vec2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::ger_out(out_, self_, vec2_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec2__storage_saved.has_value())
    AT_ASSERT(vec2__storage_saved.value().is_alias_of(vec2_.storage()));
  if (vec2__impl_saved) AT_ASSERT(vec2__impl_saved == vec2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::glu_out(Tensor & out, const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("glu_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("glu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("glu");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::glu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::glu_out(out_, self_, dim);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {
  RECORD_FUNCTION("hardtanh", std::vector<c10::IValue>({self, min_val, max_val}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<HardtanhBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardtanhBackward0>(new HardtanhBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::hardtanh(self_, min_val, max_val);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) {
  RECORD_FUNCTION("hardtanh_", std::vector<c10::IValue>({self, min_val, max_val}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<HardtanhBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<HardtanhBackward1>(new HardtanhBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::hardtanh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::hardtanh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::hardtanh_(self_, min_val, max_val);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  RECORD_FUNCTION("hardtanh_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, min_val, max_val}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::hardtanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::hardtanh_backward_out(grad_input_, grad_output_, self_, min_val, max_val);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  RECORD_FUNCTION("im2col", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Im2ColBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Im2ColBackward>(new Im2ColBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_argsize_2 = self.size(2);
    grad_fn->self_argsize_3 = self.size(3);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::im2col(self_, kernel_size, dilation, padding, stride);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::im2col_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  RECORD_FUNCTION("im2col_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("im2col_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("im2col_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::im2col_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("im2col_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::im2col_backward_out(grad_input_, grad_output_, input_size, kernel_size, dilation, padding, stride);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  RECORD_FUNCTION("instance_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::instance_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "use_input_stats", use_input_stats);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::int_repr(const Tensor & self) {
  RECORD_FUNCTION("int_repr", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("int_repr"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::int_repr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::int_repr(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool VariableType::is_distributed(const Tensor & self) {
  auto result = TypeDefault::is_distributed(self);
  return result;
}
Tensor VariableType::isnan(const Tensor & self) {
  RECORD_FUNCTION("isnan", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::isnan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::isnan(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("kthvalue_out", std::vector<c10::IValue>({values, indices, self}), Function::peek_at_next_sequence_nr());
  auto& values_ = unpack(values, "values", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("kthvalue");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("kthvalue");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::kthvalue_out(values_, indices_, self_, k, dim, keepdim);
  }
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(values);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
Tensor VariableType::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<L1LossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<L1LossBackwardBackward>(new L1LossBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::l1_loss_backward(grad_output_, self_, target_, reduction);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lgamma_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("lgamma_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("lgamma");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lgamma");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lgamma");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lgamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::lgamma_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) {
  RECORD_FUNCTION("linspace", std::vector<c10::IValue>({start, end}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::linspace(start, end, steps, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::log(const Tensor & self) {
  RECORD_FUNCTION("log", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogBackward>(new LogBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::log(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log1p_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("log1p_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log1p");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("log1p");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log1p");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log1p_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::log1p_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::log2(const Tensor & self) {
  RECORD_FUNCTION("log2", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<Log2Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log2Backward>(new Log2Backward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log2");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::log2(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::log2_(Tensor & self) {
  RECORD_FUNCTION("log2_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<Log2Backward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<Log2Backward>(new Log2Backward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log2");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log2_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::log2_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::log_(Tensor & self) {
  RECORD_FUNCTION("log_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<LogBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogBackward>(new LogBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::log");
    } else {
      op_name = jit::Symbol::fromQualString("aten::log_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::log_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
std::tuple<Tensor,Tensor> VariableType::log_sigmoid_forward(const Tensor & self) {
  RECORD_FUNCTION("log_sigmoid_forward", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<LogSigmoidBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<LogSigmoidBackward>(new LogSigmoidBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  Tensor output;
  Tensor buffer;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_sigmoid_forward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::log_sigmoid_forward(self_);
  })();
  std::tie(output, buffer) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  if (grad_fn) {
    grad_fn->buffer_ = SavedVariable(buffer, true);
  }
  return std::make_tuple(std::move(output), std::move(buffer));
}
Tensor VariableType::log_softmax(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("log_softmax", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::log_softmax");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::log_softmax(self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::lu_solve_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  RECORD_FUNCTION("lu_solve_out", std::vector<c10::IValue>({out, self, LU_data, LU_pivots}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& LU_data_ = unpack(LU_data, "LU_data", 2);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, LU_data, LU_pivots )) {
    throw_error_out_requires_grad("lu_solve");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lu_solve");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::lu_solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "LU_data", LU_data);
    jit::tracer::addInputs(node, "LU_pivots", LU_pivots);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lu_solve_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> LU_data__storage_saved =
    LU_data_.has_storage() ? c10::optional<Storage>(LU_data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_data__impl_saved;
  if (LU_data_.defined()) LU_data__impl_saved = LU_data_.getIntrusivePtr();
  c10::optional<Storage> LU_pivots__storage_saved =
    LU_pivots_.has_storage() ? c10::optional<Storage>(LU_pivots_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_pivots__impl_saved;
  if (LU_pivots_.defined()) LU_pivots__impl_saved = LU_pivots_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::lu_solve_out(out_, self_, LU_data_, LU_pivots_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (LU_data__storage_saved.has_value())
    AT_ASSERT(LU_data__storage_saved.value().is_alias_of(LU_data_.storage()));
  if (LU_data__impl_saved) AT_ASSERT(LU_data__impl_saved == LU_data_.getIntrusivePtr());
  if (LU_pivots__storage_saved.has_value())
    AT_ASSERT(LU_pivots__storage_saved.value().is_alias_of(LU_pivots_.storage()));
  if (LU_pivots__impl_saved) AT_ASSERT(LU_pivots__impl_saved == LU_pivots_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::masked_fill(const Tensor & self, const Tensor & mask, Scalar value) {
  RECORD_FUNCTION("masked_fill", std::vector<c10::IValue>({self, mask, value}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::masked_fill(self, mask, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::masked_fill(const Tensor & self, const Tensor & mask, const Tensor & value) {
  RECORD_FUNCTION("masked_fill", std::vector<c10::IValue>({self, mask, value}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_fill");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::masked_fill(self, mask, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::masked_fill_(Tensor & self, const Tensor & mask, Scalar value) {
  RECORD_FUNCTION("masked_fill_", std::vector<c10::IValue>({self, mask, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  check_inplace(self);
  std::shared_ptr<MaskedFillBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaskedFillBackward0>(new MaskedFillBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::masked_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::masked_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.masked_fill_(mask_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) {
  RECORD_FUNCTION("masked_fill_", std::vector<c10::IValue>({self, mask, value}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& value_ = unpack(value, "value", 2);
  check_inplace(self);
  std::shared_ptr<MaskedFillBackward1> grad_fn;
  if (compute_requires_grad( self, value )) {
    grad_fn = std::shared_ptr<MaskedFillBackward1>(new MaskedFillBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::masked_fill");
    } else {
      op_name = jit::Symbol::fromQualString("aten::masked_fill_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.masked_fill_(mask_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) {
  RECORD_FUNCTION("masked_scatter", std::vector<c10::IValue>({self, mask, source}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::masked_scatter");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::masked_scatter(self, mask, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) {
  RECORD_FUNCTION("masked_scatter_", std::vector<c10::IValue>({self, mask, source}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& source_ = unpack(source, "source", 2);
  check_inplace(self);
  std::shared_ptr<MaskedScatterBackward> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<MaskedScatterBackward>(new MaskedScatterBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->mask_ = SavedVariable(mask, false);
    grad_fn->source_sizes = source.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::masked_scatter");
    } else {
      op_name = jit::Symbol::fromQualString("aten::masked_scatter_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.masked_scatter_(mask_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  RECORD_FUNCTION("max_pool1d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool1d");
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
  auto result = TypeDefault::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  RECORD_FUNCTION("max_pool2d_with_indices", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MaxPool2DWithIndicesBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackward>(new MaxPool2DWithIndicesBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  Tensor result0;
  Tensor result1;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::max_pool2d_with_indices(self_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  std::tie(result0, result1) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
Tensor & VariableType::max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  RECORD_FUNCTION("max_pool2d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& indices_ = unpack(indices, "indices", 8);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_pool2d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool2d_with_indices_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool2d_with_indices_backward");
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
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_pool2d_with_indices_backward_out(grad_input_, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
std::tuple<Tensor &,Tensor &> VariableType::max_pool3d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  RECORD_FUNCTION("max_pool3d_with_indices_out", std::vector<c10::IValue>({out, indices, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool3d_with_indices");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_pool3d_with_indices");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_pool3d_with_indices");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool3d_with_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::max_pool3d_with_indices_out(out_, indices_, self_, kernel_size, stride, padding, dilation, ceil_mode);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(out, indices);
}
Tensor VariableType::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("max_unpool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("max_unpool3d_backward"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, indices ));
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::max_unpool3d_backward(grad_output_, self_, indices_, output_size, stride, padding);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::max_values(const Tensor & self, IntArrayRef dim, bool keepdim) {
  RECORD_FUNCTION("max_values", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::max_values");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::max_values(self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("mean", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward0>(new MeanBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_numel = self.numel();
    grad_fn->self_scalar_type = self.scalar_type();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mean");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mean(self_, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mean(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("mean", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MeanBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MeanBackward1>(new MeanBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mean(self_, dim, keepdim, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::median(const Tensor & self, int64_t dim, bool keepdim) {
  RECORD_FUNCTION("median", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MedianBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MedianBackward1>(new MedianBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  Tensor values;
  Tensor indices;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::median(self_, dim, keepdim);
  })();
  std::tie(values, indices) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
Tensor VariableType::median(const Tensor & self) {
  RECORD_FUNCTION("median", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<MedianBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MedianBackward0>(new MedianBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::median");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::median(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::miopen_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) {
  RECORD_FUNCTION("miopen_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto running_mean_ = unpack_opt(running_mean, "running_mean", 3);
  auto running_var_ = unpack_opt(running_var, "running_var", 4);
  auto save_mean_ = unpack_opt(save_mean, "save_mean", 5);
  auto save_var_ = unpack_opt(save_var, "save_var", 6);
  check_no_requires_grad(running_mean, "running_mean");
  check_no_requires_grad(running_var, "running_var");
  std::shared_ptr<MiopenBatchNormBackwardBackward> grad_fn;
  if (compute_requires_grad( input, grad_output, weight, save_mean, save_var )) {
    grad_fn = std::shared_ptr<MiopenBatchNormBackwardBackward>(new MiopenBatchNormBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, grad_output, weight, save_mean, save_var ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_var_ = SavedVariable(save_var, false);
    grad_fn->epsilon = epsilon;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_batch_norm_backward");
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> running_mean__storage_saved =
    running_mean_.has_storage() ? c10::optional<Storage>(running_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_mean__impl_saved;
  if (running_mean_.defined()) running_mean__impl_saved = running_mean_.getIntrusivePtr();
  c10::optional<Storage> running_var__storage_saved =
    running_var_.has_storage() ? c10::optional<Storage>(running_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> running_var__impl_saved;
  if (running_var_.defined()) running_var__impl_saved = running_var_.getIntrusivePtr();
  c10::optional<Storage> save_mean__storage_saved =
    save_mean_.has_storage() ? c10::optional<Storage>(save_mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_mean__impl_saved;
  if (save_mean_.defined()) save_mean__impl_saved = save_mean_.getIntrusivePtr();
  c10::optional<Storage> save_var__storage_saved =
    save_var_.has_storage() ? c10::optional<Storage>(save_var_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> save_var__impl_saved;
  if (save_var_.defined()) save_var__impl_saved = save_var_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_batch_norm_backward(input_, grad_output_, weight_, running_mean_, running_var_, save_mean_, save_var_, epsilon);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (running_mean__storage_saved.has_value())
    AT_ASSERT(running_mean__storage_saved.value().is_alias_of(running_mean_.storage()));
  if (running_mean__impl_saved) AT_ASSERT(running_mean__impl_saved == running_mean_.getIntrusivePtr());
  if (running_var__storage_saved.has_value())
    AT_ASSERT(running_var__storage_saved.value().is_alias_of(running_var_.storage()));
  if (running_var__impl_saved) AT_ASSERT(running_var__impl_saved == running_var_.getIntrusivePtr());
  if (save_mean__storage_saved.has_value())
    AT_ASSERT(save_mean__storage_saved.value().is_alias_of(save_mean_.storage()));
  if (save_mean__impl_saved) AT_ASSERT(save_mean__impl_saved == save_mean_.getIntrusivePtr());
  if (save_var__storage_saved.has_value())
    AT_ASSERT(save_var__storage_saved.value().is_alias_of(save_var_.storage()));
  if (save_var__impl_saved) AT_ASSERT(save_var__impl_saved == save_var_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::miopen_depthwise_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  RECORD_FUNCTION("miopen_depthwise_convolution", std::vector<c10::IValue>({self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<MiopenDepthwiseConvolutionBackward> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    grad_fn = std::shared_ptr<MiopenDepthwiseConvolutionBackward>(new MiopenDepthwiseConvolutionBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::miopen_depthwise_convolution");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::miopen_depthwise_convolution(self_, weight_, bias_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  RECORD_FUNCTION("mkldnn_max_pool2d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("mkldnn_max_pool2d"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mkldnn_max_pool2d");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mkldnn_max_pool2d(self_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::mm_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  RECORD_FUNCTION("mm_out", std::vector<c10::IValue>({out, self, mat2}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("mm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mm");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::mm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::mm_out(out_, self_, mat2_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  RECORD_FUNCTION("multi_margin_loss", std::vector<c10::IValue>({self, target, p, margin, weight}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 4);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<MultiMarginLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<MultiMarginLossBackward>(new MultiMarginLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->p = p;
    grad_fn->margin = margin;
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::multi_margin_loss(self_, target_, p, margin, weight_, reduction);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  RECORD_FUNCTION("multi_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, p, margin, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  auto weight_ = unpack_opt(weight, "weight", 6);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    throw_error_out_requires_grad("multi_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multi_margin_loss_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::multi_margin_loss_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multi_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::multi_margin_loss_backward_out(grad_input_, grad_output_, self_, target_, p, margin, weight_, reduction);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::mv(const Tensor & self, const Tensor & vec) {
  RECORD_FUNCTION("mv", std::vector<c10::IValue>({self, vec}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  std::shared_ptr<MvBackward> grad_fn;
  if (compute_requires_grad( self, vec )) {
    grad_fn = std::shared_ptr<MvBackward>(new MvBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, vec ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->vec_ = SavedVariable(vec, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::mv(self_, vec_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::native_layer_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps) {
  RECORD_FUNCTION("native_layer_norm", std::vector<c10::IValue>({input, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack_opt(weight, "weight", 1);
  auto bias_ = unpack_opt(bias, "bias", 2);
  std::shared_ptr<NativeLayerNormBackward> grad_fn;
  if (compute_requires_grad( input, weight, bias )) {
    grad_fn = std::shared_ptr<NativeLayerNormBackward>(new NativeLayerNormBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->M = M;
    grad_fn->N = N;
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::native_layer_norm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "M", M);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::native_layer_norm(input_, weight_, bias_, M, N, eps);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
std::tuple<Tensor,Tensor> VariableType::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss2d_forward", std::vector<c10::IValue>({self, target, weight}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<NllLoss2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NllLoss2DBackward>(new NllLoss2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  Tensor output;
  Tensor total_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss2d_forward");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::nll_loss2d_forward(self_, target_, weight_, reduction, ignore_index);
  })();
  std::tie(output, total_weight) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
std::tuple<Tensor,Tensor> VariableType::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  RECORD_FUNCTION("nll_loss_forward", std::vector<c10::IValue>({self, target, weight}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto weight_ = unpack_opt(weight, "weight", 2);
  check_no_requires_grad(weight, "weight");
  std::shared_ptr<NllLossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NllLossBackward>(new NllLossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  Tensor output;
  Tensor total_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::nll_loss_forward");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::nll_loss_forward(self_, target_, weight_, reduction, ignore_index);
  })();
  std::tie(output, total_weight) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
Tensor VariableType::norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) {
  RECORD_FUNCTION("norm_except_dim", std::vector<c10::IValue>({v}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::norm_except_dim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "pow", pow);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::norm_except_dim(v, pow, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::ones(IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("ones", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::ones");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::ones(size, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pin_memory(const Tensor & self) {
  RECORD_FUNCTION("pin_memory", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::pin_memory");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::pin_memory(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::polygamma(int64_t n, const Tensor & self) {
  RECORD_FUNCTION("polygamma", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<PolygammaBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PolygammaBackward>(new PolygammaBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->n = n;
    grad_fn->self_ = SavedVariable(self, false);
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::polygamma(n, self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::polygamma_(Tensor & self, int64_t n) {
  RECORD_FUNCTION("polygamma_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("polygamma_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::polygamma");
    } else {
      op_name = jit::Symbol::fromQualString("aten::polygamma_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polygamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.polygamma_(n);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::pow(const Tensor & self, Scalar exponent) {
  RECORD_FUNCTION("pow", std::vector<c10::IValue>({self, exponent}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<PowBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PowBackward0>(new PowBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent = exponent;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::pow(self_, exponent);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pow(const Tensor & self, const Tensor & exponent) {
  RECORD_FUNCTION("pow", std::vector<c10::IValue>({self, exponent}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<PowBackward1> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::shared_ptr<PowBackward1>(new PowBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::pow(self_, exponent_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::pow(Scalar self, const Tensor & exponent) {
  RECORD_FUNCTION("pow", std::vector<c10::IValue>({self, exponent}), Function::peek_at_next_sequence_nr());
  auto& exponent_ = unpack(exponent, "exponent", 1);
  std::shared_ptr<PowBackward2> grad_fn;
  if (compute_requires_grad( exponent )) {
    grad_fn = std::shared_ptr<PowBackward2>(new PowBackward2(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( exponent ));
    grad_fn->self = self;
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::pow(self, exponent_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::pow_(Tensor & self, Scalar exponent) {
  RECORD_FUNCTION("pow_", std::vector<c10::IValue>({self, exponent}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<PowBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<PowBackward0>(new PowBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->exponent = exponent;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::pow");
    } else {
      op_name = jit::Symbol::fromQualString("aten::pow_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.pow_(exponent);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::pow_(Tensor & self, const Tensor & exponent) {
  RECORD_FUNCTION("pow_", std::vector<c10::IValue>({self, exponent}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  check_inplace(self);
  std::shared_ptr<PowBackward1> grad_fn;
  if (compute_requires_grad( self, exponent )) {
    grad_fn = std::shared_ptr<PowBackward1>(new PowBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self.clone(), false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::pow");
    } else {
      op_name = jit::Symbol::fromQualString("aten::pow_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("pow_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.pow_(exponent_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::prelu(const Tensor & self, const Tensor & weight) {
  RECORD_FUNCTION("prelu", std::vector<c10::IValue>({self, weight}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  std::shared_ptr<PreluBackward> grad_fn;
  if (compute_requires_grad( self, weight )) {
    grad_fn = std::shared_ptr<PreluBackward>(new PreluBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prelu");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::prelu(self_, weight_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::prod(const Tensor & self, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("prod", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward0>(new ProdBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->self_ = SavedVariable(self, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::prod");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::prod(self_, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor VariableType::prod(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  RECORD_FUNCTION("prod", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ProdBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ProdBackward1>(new ProdBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::prod(self_, dim, keepdim, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
std::tuple<Tensor,Tensor> VariableType::qr(const Tensor & self, bool some) {
  RECORD_FUNCTION("qr", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<QrBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<QrBackward>(new QrBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->some = some;
  }
  Tensor Q;
  Tensor R;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::qr");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::qr(self_, some);
  })();
  std::tie(Q, R) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( Q, R ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  if (grad_fn) {
    grad_fn->Q_ = SavedVariable(Q, true);
    grad_fn->R_ = SavedVariable(R, true);
  }
  return std::make_tuple(std::move(Q), std::move(R));
}
QScheme VariableType::qscheme(const Tensor & self) {
  RECORD_FUNCTION("qscheme", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.qscheme();
  })();
  auto result = tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
Tensor VariableType::quantize_linear_per_channel(const Tensor & self, const Tensor & scales, const Tensor & zero_points, IntArrayRef axis, ScalarType dtype) {
  RECORD_FUNCTION("quantize_linear_per_channel", std::vector<c10::IValue>({self, scales, zero_points}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& scales_ = unpack(scales, "scales", 1);
  auto& zero_points_ = unpack(zero_points, "zero_points", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, scales, zero_points )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("quantize_linear_per_channel"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, scales, zero_points ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::quantize_linear_per_channel");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> scales__storage_saved =
    scales_.has_storage() ? c10::optional<Storage>(scales_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> scales__impl_saved;
  if (scales_.defined()) scales__impl_saved = scales_.getIntrusivePtr();
  c10::optional<Storage> zero_points__storage_saved =
    zero_points_.has_storage() ? c10::optional<Storage>(zero_points_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> zero_points__impl_saved;
  if (zero_points_.defined()) zero_points__impl_saved = zero_points_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::quantize_linear_per_channel(self_, scales_, zero_points_, axis, dtype);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (scales__storage_saved.has_value())
    AT_ASSERT(scales__storage_saved.value().is_alias_of(scales_.storage()));
  if (scales__impl_saved) AT_ASSERT(scales__impl_saved == scales_.getIntrusivePtr());
  if (zero_points__storage_saved.has_value())
    AT_ASSERT(zero_points__storage_saved.value().is_alias_of(zero_points_.storage()));
  if (zero_points__impl_saved) AT_ASSERT(zero_points__impl_saved == zero_points_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::rand_like(const Tensor & self) {
  RECORD_FUNCTION("rand_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::rand_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::rand_like(const Tensor & self, const TensorOptions & options) {
  RECORD_FUNCTION("rand_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::rand_like(self, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint(high, size, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint(high, size, generator, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint(low, high, size, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options) {
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randint(low, high, size, generator, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randn_like(const Tensor & self) {
  RECORD_FUNCTION("randn_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randn_like(self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randn_like(const Tensor & self, const TensorOptions & options) {
  RECORD_FUNCTION("randn_like", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randn_like(self, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randperm(int64_t n, const TensorOptions & options) {
  RECORD_FUNCTION("randperm", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::randperm");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randperm(n, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::randperm(int64_t n, Generator * generator, const TensorOptions & options) {
  RECORD_FUNCTION("randperm", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
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
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::randperm(n, generator, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("reflection_pad2d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReflectionPad2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReflectionPad2DBackwardBackward>(new ReflectionPad2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::reflection_pad2d_backward(grad_output_, self_, padding);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::repeat_interleave(const Tensor & repeats) {
  RECORD_FUNCTION("repeat_interleave", std::vector<c10::IValue>({repeats}), Function::peek_at_next_sequence_nr());
  auto& repeats_ = unpack(repeats, "repeats", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( repeats )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("repeat_interleave"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( repeats ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "repeats", repeats);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> repeats__storage_saved =
    repeats_.has_storage() ? c10::optional<Storage>(repeats_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> repeats__impl_saved;
  if (repeats_.defined()) repeats__impl_saved = repeats_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::repeat_interleave(repeats_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (repeats__storage_saved.has_value())
    AT_ASSERT(repeats__storage_saved.value().is_alias_of(repeats_.storage()));
  if (repeats__impl_saved) AT_ASSERT(repeats__impl_saved == repeats_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::repeat_interleave(const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim) {
  RECORD_FUNCTION("repeat_interleave", std::vector<c10::IValue>({self, repeats}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::repeat_interleave(self, repeats, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::repeat_interleave(const Tensor & self, int64_t repeats, c10::optional<int64_t> dim) {
  RECORD_FUNCTION("repeat_interleave", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::repeat_interleave(self, repeats, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad1d_backward", std::vector<c10::IValue>({grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<ReplicationPad1DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<ReplicationPad1DBackwardBackward>(new ReplicationPad1DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::replication_pad1d_backward(grad_output_, self_, padding);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::replication_pad2d(const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad2d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<ReplicationPad2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackward>(new ReplicationPad2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::replication_pad2d(self_, padding);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad2d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::replication_pad2d_backward_out(grad_input_, grad_output_, self_, padding);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::replication_pad3d_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  RECORD_FUNCTION("replication_pad3d_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("replication_pad3d");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::replication_pad3d");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::replication_pad3d_out(out_, self_, padding);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  RECORD_FUNCTION("rnn_tanh_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Function::peek_at_next_sequence_nr());
  auto result = TypeDefault::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
Tensor VariableType::round(const Tensor & self) {
  RECORD_FUNCTION("round", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RoundBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RoundBackward>(new RoundBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::round");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::round(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::round_(Tensor & self) {
  RECORD_FUNCTION("round_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RoundBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RoundBackward>(new RoundBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::round");
    } else {
      op_name = jit::Symbol::fromQualString("aten::round_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::round_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::rsqrt(const Tensor & self) {
  RECORD_FUNCTION("rsqrt", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RsqrtBackward>(new RsqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::rsqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::rsqrt(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::rsqrt_(Tensor & self) {
  RECORD_FUNCTION("rsqrt_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<RsqrtBackward>(new RsqrtBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::rsqrt");
    } else {
      op_name = jit::Symbol::fromQualString("aten::rsqrt_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rsqrt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::rsqrt_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor VariableType::s_native_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("s_native_addmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("s_native_addmm"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::s_native_addmm");
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
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::s_native_addmm(self_, mat1_, mat2_, beta, alpha);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::s_native_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("s_native_addmm_", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mat1_ = unpack(mat1, "mat1", 1);
  auto& mat2_ = unpack(mat2, "mat2", 2);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, mat1, mat2 )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("s_native_addmm_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, mat1, mat2 ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::s_native_addmm");
    } else {
      op_name = jit::Symbol::fromQualString("aten::s_native_addmm_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("s_native_addmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat1__storage_saved =
    mat1_.has_storage() ? c10::optional<Storage>(mat1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat1__impl_saved;
  if (mat1_.defined()) mat1__impl_saved = mat1_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::s_native_addmm_(self_, mat1_, mat2_, beta, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat1__storage_saved.has_value())
    AT_ASSERT(mat1__storage_saved.value().is_alias_of(mat1_.storage()));
  if (mat1__impl_saved) AT_ASSERT(mat1__impl_saved == mat1_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::scalar_tensor(Scalar s, const TensorOptions & options) {
  RECORD_FUNCTION("scalar_tensor", std::vector<c10::IValue>({s}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::scalar_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::scalar_tensor(s, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::set_(Tensor & self, Storage source) {
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.set_(source);
  }
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & VariableType::set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.set_(source, storage_offset, size, stride);
  }
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & VariableType::set_(Tensor & self, const Tensor & source) {
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self, source}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& source_ = unpack(source, "source", 1);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self, source )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::set");
    } else {
      op_name = jit::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.set_(source_);
  }
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::set_(Tensor & self) {
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::set");
    } else {
      op_name = jit::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.set_();
  }
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::set_quantizer_(Tensor & self, ConstQuantizerPtr quantizer) {
  RECORD_FUNCTION("set_quantizer_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_quantizer_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.set_quantizer_(quantizer);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  return self;
}
Tensor & VariableType::sigmoid_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("sigmoid_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sigmoid");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sigmoid");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sigmoid_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::sign_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("sign_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sign");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sign");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sign");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sign_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sign_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("smooth_l1_loss", std::vector<c10::IValue>({self, target}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SmoothL1LossBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SmoothL1LossBackward>(new SmoothL1LossBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::smooth_l1_loss(self_, target_, reduction);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("smooth_l1_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& target_ = unpack(target, "target", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::smooth_l1_loss_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("smooth_l1_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::smooth_l1_loss_backward_out(grad_input_, grad_output_, self_, target_, reduction);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  RECORD_FUNCTION("soft_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  check_no_requires_grad(target, "target");
  std::shared_ptr<SoftMarginLossBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftMarginLossBackwardBackward>(new SoftMarginLossBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::soft_margin_loss_backward(grad_output_, self_, target_, reduction);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  RECORD_FUNCTION("softplus_backward", std::vector<c10::IValue>({grad_output, self, beta, threshold, output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 4);
  std::shared_ptr<SoftplusBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    grad_fn = std::shared_ptr<SoftplusBackwardBackward>(new SoftplusBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->beta = beta;
    grad_fn->threshold = threshold;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::softplus_backward(grad_output_, self_, beta, threshold, output_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &> VariableType::solve_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) {
  RECORD_FUNCTION("solve_out", std::vector<c10::IValue>({solution, lu, self, A}), Function::peek_at_next_sequence_nr());
  auto& solution_ = unpack(solution, "solution", 0);
  auto& lu_ = unpack(lu, "lu", 1);
  auto& self_ = unpack(self, "self", 2);
  auto& A_ = unpack(A, "A", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("solve");
  }
  if (compute_requires_grad( solution )) {
    throw_error_out_requires_grad("solve");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::solve");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "lu", lu);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "solution", solution);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("solve_out", solution);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> solution__storage_saved =
    solution_.has_storage() ? c10::optional<Storage>(solution_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> solution__impl_saved;
  if (solution_.defined()) solution__impl_saved = solution_.getIntrusivePtr();
  c10::optional<Storage> lu__storage_saved =
    lu_.has_storage() ? c10::optional<Storage>(lu_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> lu__impl_saved;
  if (lu_.defined()) lu__impl_saved = lu_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::solve_out(solution_, lu_, self_, A_);
  }
  #ifndef NDEBUG
  if (solution__storage_saved.has_value())
    AT_ASSERT(solution__storage_saved.value().is_alias_of(solution_.storage()));
  if (solution__impl_saved) AT_ASSERT(solution__impl_saved == solution_.getIntrusivePtr());
  if (lu__storage_saved.has_value())
    AT_ASSERT(lu__storage_saved.value().is_alias_of(lu_.storage()));
  if (lu__impl_saved) AT_ASSERT(lu__impl_saved == lu_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  increment_version(solution);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( solution ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, solution);
    jit::tracer::addOutput(node, lu);
  }
  return std::forward_as_tuple(solution, lu);
}
Tensor VariableType::sparse_coo_tensor(IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("sparse_coo_tensor", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::sparse_coo_tensor(size, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  RECORD_FUNCTION("sparse_coo_tensor", std::vector<c10::IValue>({indices, values}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::sparse_coo_tensor(indices, values, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  RECORD_FUNCTION("sparse_coo_tensor", std::vector<c10::IValue>({indices, values}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::sparse_coo_tensor(indices, values, size, options);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::sparse_mask(const Tensor & self, const Tensor & mask) {
  RECORD_FUNCTION("sparse_mask", std::vector<c10::IValue>({self, mask}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  std::shared_ptr<SparseMaskBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<SparseMaskBackward>(new SparseMaskBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sparse_mask");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return self_.sparse_mask(mask_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  RECORD_FUNCTION("sparse_resize_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("sparse_resize_"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize");
    } else {
      op_name = jit::Symbol::fromQualString("aten::sparse_resize_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_resize_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.sparse_resize_(size, sparse_dim, dense_dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor & VariableType::sqrt_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("sqrt_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sqrt");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sqrt");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::sqrt");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sqrt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::sqrt_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  RECORD_FUNCTION("sspaddmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Function::peek_at_next_sequence_nr());
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::sspaddmm(self, mat1, mat2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor &,Tensor &,Tensor &> VariableType::svd_out(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) {
  RECORD_FUNCTION("svd_out", std::vector<c10::IValue>({U, S, V, self}), Function::peek_at_next_sequence_nr());
  auto& U_ = unpack(U, "U", 0);
  auto& S_ = unpack(S, "S", 1);
  auto& V_ = unpack(V, "V", 2);
  auto& self_ = unpack(self, "self", 3);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("svd");
  }
  if (compute_requires_grad( U, S, V )) {
    throw_error_out_requires_grad("svd");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::svd");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "S", S);
    jit::tracer::addInputs(node, "V", V);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "U", U);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("svd_out", U);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> U__storage_saved =
    U_.has_storage() ? c10::optional<Storage>(U_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> U__impl_saved;
  if (U_.defined()) U__impl_saved = U_.getIntrusivePtr();
  c10::optional<Storage> S__storage_saved =
    S_.has_storage() ? c10::optional<Storage>(S_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> S__impl_saved;
  if (S_.defined()) S__impl_saved = S_.getIntrusivePtr();
  c10::optional<Storage> V__storage_saved =
    V_.has_storage() ? c10::optional<Storage>(V_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> V__impl_saved;
  if (V_.defined()) V__impl_saved = V_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::svd_out(U_, S_, V_, self_, some, compute_uv);
  }
  #ifndef NDEBUG
  if (U__storage_saved.has_value())
    AT_ASSERT(U__storage_saved.value().is_alias_of(U_.storage()));
  if (U__impl_saved) AT_ASSERT(U__impl_saved == U_.getIntrusivePtr());
  if (S__storage_saved.has_value())
    AT_ASSERT(S__storage_saved.value().is_alias_of(S_.storage()));
  if (S__impl_saved) AT_ASSERT(S__impl_saved == S_.getIntrusivePtr());
  if (V__storage_saved.has_value())
    AT_ASSERT(V__storage_saved.value().is_alias_of(V_.storage()));
  if (V__impl_saved) AT_ASSERT(V__impl_saved == V_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(U);
  increment_version(S);
  increment_version(V);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( U, S, V ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, V);
  }
  return std::forward_as_tuple(U, S, V);
}
Tensor & VariableType::tan_out(Tensor & out, const Tensor & self) {
  RECORD_FUNCTION("tan_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tan");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("tan");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tan");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::tan_out(out_, self_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::tanh(const Tensor & self) {
  RECORD_FUNCTION("tanh", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TanhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanhBackward>(new TanhBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::tanh(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
Tensor & VariableType::tanh_(Tensor & self) {
  RECORD_FUNCTION("tanh_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TanhBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TanhBackward>(new TanhBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::tanh");
    } else {
      op_name = jit::Symbol::fromQualString("aten::tanh_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::tanh_(self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true);
  }
  return self;
}
Tensor & VariableType::tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  RECORD_FUNCTION("tanh_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& output_ = unpack(output, "output", 2);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("tanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("tanh_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tanh_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::tanh_backward_out(grad_input_, grad_output_, output_);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::thnn_conv2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  RECORD_FUNCTION("thnn_conv2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv2d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::thnn_conv2d_out(out, self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
std::tuple<Tensor,Tensor> VariableType::thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) {
  RECORD_FUNCTION("thnn_conv_depthwise2d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  std::shared_ptr<ThnnConvDepthwise2DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    grad_fn = std::shared_ptr<ThnnConvDepthwise2DBackwardBackward>(new ThnnConvDepthwise2DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_argsize_1 = self.size(1);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  Tensor grad_input;
  Tensor grad_weight;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_backward");
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
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::thnn_conv_depthwise2d_backward(grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
  })();
  std::tie(grad_input, grad_weight) = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}
Tensor & VariableType::thnn_conv_depthwise2d_forward_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  RECORD_FUNCTION("thnn_conv_depthwise2d_forward_out", std::vector<c10::IValue>({out, self, weight, bias}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto bias_ = unpack_opt(bias, "bias", 4);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::thnn_conv_depthwise2d_forward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("thnn_conv_depthwise2d_forward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> bias__storage_saved =
    bias_.has_storage() ? c10::optional<Storage>(bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> bias__impl_saved;
  if (bias_.defined()) bias__impl_saved = bias_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::thnn_conv_depthwise2d_forward_out(out_, self_, weight_, kernel_size, bias_, stride, padding, dilation);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (bias__storage_saved.has_value())
    AT_ASSERT(bias__storage_saved.value().is_alias_of(bias_.storage()));
  if (bias__impl_saved) AT_ASSERT(bias__impl_saved == bias_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor & VariableType::threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value) {
  RECORD_FUNCTION("threshold_out", std::vector<c10::IValue>({out, self, threshold, value}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("threshold");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("threshold");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::threshold");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "threshold", threshold);
    jit::tracer::addInputs(node, "value", value);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("threshold_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::threshold_out(out_, self_, threshold, value);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::to(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy) {
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "options", options);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, options, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy) {
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, device, dtype, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy) {
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, dtype, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to(const Tensor & self, const Tensor & other, bool non_blocking, bool copy) {
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to(self, other, non_blocking, copy);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::to_mkldnn_backward(const Tensor & grad, const Tensor & input) {
  RECORD_FUNCTION("to_mkldnn_backward", std::vector<c10::IValue>({grad, input}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::to_mkldnn_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::to_mkldnn_backward(grad, input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::trace(const Tensor & self) {
  RECORD_FUNCTION("trace", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TraceBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TraceBackward>(new TraceBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::trace");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::trace(self_);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::tril_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  RECORD_FUNCTION("tril_out", std::vector<c10::IValue>({out, self}), Function::peek_at_next_sequence_nr());
  auto& out_ = unpack(out, "out", 0);
  auto& self_ = unpack(self, "self", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tril");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("tril");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::tril");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::tril_out(out_, self_, diagonal);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(out);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
Tensor VariableType::triu(const Tensor & self, int64_t diagonal) {
  RECORD_FUNCTION("triu", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<TriuBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TriuBackward>(new TriuBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::triu(self_, diagonal);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::triu_(Tensor & self, int64_t diagonal) {
  RECORD_FUNCTION("triu_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<TriuBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<TriuBackward>(new TriuBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::triu");
    } else {
      op_name = jit::Symbol::fromQualString("aten::triu_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("triu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.triu_(diagonal);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  RECORD_FUNCTION("triu_indices", std::vector<c10::IValue>({}), Function::peek_at_next_sequence_nr());
  auto options_ = TensorOptions(options).is_variable(false);
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::triu_indices");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "row", row);
    jit::tracer::addInputs(node, "col", col);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "options", options);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::triu_indices(row, col, offset, options_);
  })();
  auto result = as_variable(tmp);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::type_as(const Tensor & self, const Tensor & other) {
  RECORD_FUNCTION("type_as", std::vector<c10::IValue>({self, other}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::type_as");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result = TypeDefault::type_as(self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
std::tuple<Tensor,Tensor,Tensor> VariableType::unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  RECORD_FUNCTION("unique_dim", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<NotImplemented> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("unique_dim"), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  Tensor result0;
  Tensor result1;
  Tensor result2;
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unique_dim");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::unique_dim(self_, dim, sorted, return_inverse, return_counts);
  })();
  std::tie(result0, result1, result2) = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
Tensor VariableType::unsqueeze(const Tensor & self, int64_t dim) {
  RECORD_FUNCTION("unsqueeze", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UnsqueezeBackward0> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsqueezeBackward0>(new UnsqueezeBackward0(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::unsqueeze");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::unsqueeze(self_, dim);
  })();
  auto result = as_view(self, tmp, true);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::unsqueeze_(Tensor & self, int64_t dim) {
  RECORD_FUNCTION("unsqueeze_", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  std::shared_ptr<UnsqueezeBackward1> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UnsqueezeBackward1>(new UnsqueezeBackward1(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    if (tracer_state->force_outplace) {
      op_name = jit::Symbol::fromQualString("aten::unsqueeze");
    } else {
      op_name = jit::Symbol::fromQualString("aten::unsqueeze_");
    }
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unsqueeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    self_.unsqueeze_(dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  increment_version(self);
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
Tensor VariableType::upsample_bicubic2d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  RECORD_FUNCTION("upsample_bicubic2d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleBicubic2DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackward>(new UpsampleBicubic2DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_bicubic2d(self_, output_size, align_corners);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  RECORD_FUNCTION("upsample_bicubic2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bicubic2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bicubic2d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_bicubic2d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bicubic2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_bicubic2d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor VariableType::upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size) {
  RECORD_FUNCTION("upsample_nearest3d_backward", std::vector<c10::IValue>({grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  std::shared_ptr<UpsampleNearest3DBackwardBackward> grad_fn;
  if (compute_requires_grad( grad_output )) {
    grad_fn = std::shared_ptr<UpsampleNearest3DBackwardBackward>(new UpsampleNearest3DBackwardBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_nearest3d_backward(grad_output_, output_size, input_size);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor VariableType::upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners) {
  RECORD_FUNCTION("upsample_trilinear3d", std::vector<c10::IValue>({self}), Function::peek_at_next_sequence_nr());
  auto& self_ = unpack(self, "self", 0);
  std::shared_ptr<UpsampleTrilinear3DBackward> grad_fn;
  if (compute_requires_grad( self )) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackward>(new UpsampleTrilinear3DBackward(), deleteFunction);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
  }
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
    tracer_state->graph->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto tmp = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::upsample_trilinear3d(self_, output_size, align_corners);
  })();
  auto result = as_variable(tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
Tensor & VariableType::upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners) {
  RECORD_FUNCTION("upsample_trilinear3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Function::peek_at_next_sequence_nr());
  auto& grad_input_ = unpack(grad_input, "grad_input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  std::shared_ptr<Function> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::upsample_trilinear3d_backward");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_trilinear3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    at::upsample_trilinear3d_backward_out(grad_input_, grad_output_, output_size, input_size, align_corners);
  }
  #ifndef NDEBUG
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  increment_version(grad_input);
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
Tensor & VariableType::zeros_out(Tensor & out, IntArrayRef size) {
  RECORD_FUNCTION("zeros_out", std::vector<c10::IValue>({out}), Function::peek_at_next_sequence_nr());
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = jit::Symbol::fromQualString("aten::zeros");
    node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", out.options());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->graph->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("zeros_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  TypeDefault::zeros_out(out, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}

static auto& registerer = globalATenDispatch()
  .registerVariableOp<Tensor (const Tensor &, Scalar)>("aten::__and__(Tensor self, Scalar other) -> Tensor", &VariableType::__and__)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::__and__(Tensor self, Tensor other) -> Tensor", &VariableType::__and__)
  .registerVariableOp<Tensor & (Tensor &, Scalar)>("aten::__iand__(Tensor(a!) self, Scalar other) -> Tensor(a!)", &VariableType::__iand__)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::__iand__(Tensor(a!) self, Tensor other) -> Tensor(a!)", &VariableType::__iand__)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor", &VariableType::_adaptive_avg_pool2d_backward)
  .registerVariableOp<Tensor (const Tensor &, bool)>("aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor", &VariableType::_cast_Long)
  .registerVariableOp<Tensor (TensorList, int64_t)>("aten::_cat(Tensor[] tensors, int dim=0) -> Tensor", &VariableType::_cat)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>)>("aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", &VariableType::_convolution_double_backward)
  .registerVariableOp<void (int64_t)>("aten::_cufft_clear_plan_cache(int device_index) -> void", &VariableType::_cufft_clear_plan_cache)
  .registerVariableOp<int64_t (int64_t)>("aten::_cufft_get_plan_cache_max_size(int device_index) -> int", &VariableType::_cufft_get_plan_cache_max_size)
  .registerVariableOp<Tensor (const Tensor &, int64_t)>("aten::_cumprod(Tensor self, int dim) -> Tensor", &VariableType::_cumprod)
  .registerVariableOp<int64_t (const Tensor &)>("aten::_debug_has_internal_overlap(Tensor self) -> int", &VariableType::_debug_has_internal_overlap)
  .registerVariableOp<Tensor (const Tensor &, double, int64_t, ScalarType)>("aten::_dequantize_linear(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor", &VariableType::_dequantize_linear)
  .registerVariableOp<int64_t (const Tensor &)>("aten::_dimI(Tensor self) -> int", &VariableType::_dimI)
  .registerVariableOp<Tensor (IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>)>("aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor", &VariableType::_empty_affine_quantized)
  .registerVariableOp<Tensor (const Tensor &, int64_t, bool, bool, bool, IntArrayRef, bool, bool, IntArrayRef)>("aten::_fft_with_size(Tensor self, int signal_ndim, bool complex_input, bool complex_output, bool inverse, int[] checked_signal_sizes, bool normalized, bool onesided, int[] output_sizes) -> Tensor", &VariableType::_fft_with_size)
  .registerVariableOp<Tensor (const Tensor &, int64_t, bool)>("aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor", &VariableType::_log_softmax)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>("aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor, Tensor, Tensor)", &VariableType::_lu_with_info)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, int64_t, Generator *)>("aten::_multinomial_alias_draw(Tensor J, Tensor q, int num_samples, *, Generator? generator=None) -> Tensor", &VariableType::_multinomial_alias_draw)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, std::array<bool,3>)>("aten::_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", &VariableType::_nnpack_spatial_convolution_backward)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("aten::_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor", &VariableType::_nnpack_spatial_convolution_backward_input)
  .registerVariableOp<int64_t (const Tensor &)>("aten::_nnz(Tensor self) -> int", &VariableType::_nnz)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool)>("aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)", &VariableType::_pack_padded_sequence)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, Scalar, int64_t)>("aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)", &VariableType::_pad_packed_sequence)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>("aten::_qr_helper(Tensor self, bool some) -> (Tensor, Tensor)", &VariableType::_qr_helper)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor", &VariableType::_reshape_from_tensor)
  .registerVariableOp<Tensor & (Tensor &, int64_t, const Tensor &, int64_t, int64_t)>("aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)", &VariableType::_sobol_engine_ff_)
  .registerVariableOp<Tensor & (Tensor &, int64_t)>("aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)", &VariableType::_sobol_engine_initialize_state_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>("aten::_sparse_dense_add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", &VariableType::_sparse_dense_add_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor", &VariableType::_sparse_mm)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::_sparse_mul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::_sparse_mul_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, Scalar)>("aten::_sparse_mul_scalar(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::_sparse_mul_scalar_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, int64_t)>("aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor", &VariableType::_weight_norm)
  .registerVariableOp<Tensor (const Tensor &)>("aten::acos(Tensor self) -> Tensor", &VariableType::acos)
  .registerVariableOp<Tensor & (Tensor &)>("aten::acos_(Tensor(a!) self) -> Tensor(a!)", &VariableType::acos_)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef)>("aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor", &VariableType::adaptive_avg_pool3d)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::adaptive_avg_pool3d_backward_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, Scalar)>("aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", &VariableType::add)
  .registerVariableOp<Tensor (const Tensor &, Scalar, Scalar)>("aten::add(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", &VariableType::add)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, Scalar)>("aten::add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)", &VariableType::add_)
  .registerVariableOp<Tensor & (Tensor &, Scalar, Scalar)>("aten::add_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)", &VariableType::add_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", &VariableType::addbmm_out)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef)>("aten::affine_grid_generator_backward(Tensor grad, int[] size) -> Tensor", &VariableType::affine_grid_generator_backward)
  .registerVariableOp<Tensor (const Tensor &)>("aten::alias(Tensor(a) self) -> Tensor(a)", &VariableType::alias)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, int64_t, bool)>("aten::all(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::all_out)
  .registerVariableOp<bool (const Tensor &, const Tensor &, double, double, bool)>("aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool", &VariableType::allclose)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, int64_t, bool)>("aten::any(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::any_out)
  .registerVariableOp<Tensor (Scalar, const TensorOptions &)>("aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::arange)
  .registerVariableOp<Tensor (Scalar, Scalar, const TensorOptions &)>("aten::arange(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::arange)
  .registerVariableOp<Tensor (Scalar, Scalar, Scalar, const TensorOptions &)>("aten::arange(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::arange)
  .registerVariableOp<Tensor (const Tensor &)>("aten::asin(Tensor self) -> Tensor", &VariableType::asin)
  .registerVariableOp<Tensor & (Tensor &)>("aten::asin_(Tensor(a!) self) -> Tensor(a!)", &VariableType::asin_)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>("aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad) -> Tensor", &VariableType::avg_pool3d_backward)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>("aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)", &VariableType::batch_norm_gather_stats)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, double)>("aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)", &VariableType::batch_norm_stats)
  .registerVariableOp<Tensor (const Tensor &, Generator *)>("aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor", &VariableType::bernoulli)
  .registerVariableOp<Tensor (const Tensor &, double, Generator *)>("aten::bernoulli(Tensor self, float p, *, Generator? generator=None) -> Tensor", &VariableType::bernoulli)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, Generator *)>("aten::bernoulli_(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)", &VariableType::bernoulli_)
  .registerVariableOp<Tensor & (Tensor &, double, Generator *)>("aten::bernoulli_(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)", &VariableType::bernoulli_)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor", &VariableType::binary_cross_entropy_with_logits_backward)
  .registerVariableOp<Tensor (TensorList, int64_t)>("aten::cat(Tensor[] tensors, int dim=0) -> Tensor", &VariableType::cat)
  .registerVariableOp<Tensor & (Tensor &, double, double, Generator *)>("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)", &VariableType::cauchy_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::ceil(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::ceil_out)
  .registerVariableOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>("aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]", &VariableType::chunk)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor", &VariableType::col2im)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::col2im_backward_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>("aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor", &VariableType::conv1d)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>)>("aten::conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", &VariableType::conv_dilated3d_backward)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)", &VariableType::conv_tbc_backward)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t)>("aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor", &VariableType::cosine_embedding_loss)
  .registerVariableOp<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>("aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid", &VariableType::cudnn_affine_grid_generator)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", &VariableType::cudnn_convolution)
  .registerVariableOp<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", &VariableType::cudnn_convolution_transpose_backward_weight)
  .registerVariableOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>("aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor", &VariableType::cumprod)
  .registerVariableOp<Tensor (const Tensor &)>("aten::dequantize(Tensor self) -> Tensor", &VariableType::dequantize)
  .registerVariableOp<Tensor (const Tensor &)>("aten::det(Tensor self) -> Tensor", &VariableType::det)
  .registerVariableOp<Tensor (const Tensor &)>("aten::detach(Tensor self) -> Tensor", &VariableType::detach)
  .registerVariableOp<Tensor & (Tensor &)>("aten::detach_(Tensor(a!) self) -> Tensor(a!)", &VariableType::detach_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::div(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::div_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::dot(Tensor self, Tensor tensor) -> Tensor", &VariableType::dot)
  .registerVariableOp<Tensor (std::string, TensorList)>("aten::einsum(str equation, Tensor[] tensors) -> Tensor", &VariableType::einsum)
  .registerVariableOp<Tensor (const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>("aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor", &VariableType::elu_backward)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, int64_t, bool, bool)>("aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor", &VariableType::embedding)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>("aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor", &VariableType::embedding_sparse_backward)
  .registerVariableOp<Tensor (const Tensor &)>("aten::erf(Tensor self) -> Tensor", &VariableType::erf)
  .registerVariableOp<Tensor & (Tensor &)>("aten::erf_(Tensor(a!) self) -> Tensor(a!)", &VariableType::erf_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::exp(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::exp_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::expand_as(Tensor self, Tensor other) -> Tensor", &VariableType::expand_as)
  .registerVariableOp<Tensor & (Tensor &, int64_t)>("aten::eye(int n, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::eye_out)
  .registerVariableOp<Tensor & (Tensor &, int64_t, int64_t)>("aten::eye(int n, int m, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::eye_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor", &VariableType::fbgemm_linear_int8_weight)
  .registerVariableOp<Tensor (const Tensor &, double, bool)>("aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor", &VariableType::feature_alpha_dropout)
  .registerVariableOp<Tensor & (Tensor &, double, bool)>("aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)", &VariableType::feature_alpha_dropout_)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef)>("aten::flip(Tensor self, int[] dims) -> Tensor", &VariableType::flip)
  .registerVariableOp<Tensor (const Tensor &, Scalar)>("aten::fmod(Tensor self, Scalar other) -> Tensor", &VariableType::fmod)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::fmod(Tensor self, Tensor other) -> Tensor", &VariableType::fmod)
  .registerVariableOp<Tensor & (Tensor &, Scalar)>("aten::fmod_(Tensor(a!) self, Scalar other) -> Tensor(a!)", &VariableType::fmod_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::fmod_(Tensor(a!) self, Tensor other) -> Tensor(a!)", &VariableType::fmod_)
  .registerVariableOp<Tensor (const Tensor &)>("aten::frac(Tensor self) -> Tensor", &VariableType::frac)
  .registerVariableOp<Tensor & (Tensor &)>("aten::frac_(Tensor(a!) self) -> Tensor(a!)", &VariableType::frac_)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)", &VariableType::fractional_max_pool2d)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::fractional_max_pool2d_backward_out)
  .registerVariableOp<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>("aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))", &VariableType::fractional_max_pool3d_out)
  .registerVariableOp<Tensor (const Tensor &)>("aten::frobenius_norm(Tensor self) -> Tensor", &VariableType::frobenius_norm)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, bool)>("aten::frobenius_norm(Tensor self, int[1] dim, bool keepdim=False) -> Tensor", &VariableType::frobenius_norm)
  .registerVariableOp<Tensor (std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &)>("aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::from_file)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>("aten::gels(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)", &VariableType::gels)
  .registerVariableOp<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &)>("aten::geqrf(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)", &VariableType::geqrf_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::ger(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::ger_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, int64_t)>("aten::glu(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::glu_out)
  .registerVariableOp<Tensor (const Tensor &, Scalar, Scalar)>("aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor", &VariableType::hardtanh)
  .registerVariableOp<Tensor & (Tensor &, Scalar, Scalar)>("aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)", &VariableType::hardtanh_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::hardtanh_backward_out)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor", &VariableType::im2col)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>("aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::im2col_backward_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>("aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor", &VariableType::instance_norm)
  .registerVariableOp<Tensor (const Tensor &)>("aten::int_repr(Tensor self) -> Tensor", &VariableType::int_repr)
  .registerVariableOp<bool (const Tensor &)>("aten::is_distributed(Tensor self) -> bool", &VariableType::is_distributed)
  .registerVariableOp<Tensor (const Tensor &)>("aten::isnan(Tensor self) -> Tensor", &VariableType::isnan)
  .registerVariableOp<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool)>("aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)", &VariableType::kthvalue_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>("aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor", &VariableType::l1_loss_backward)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::lgamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::lgamma_out)
  .registerVariableOp<Tensor (Scalar, Scalar, int64_t, const TensorOptions &)>("aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::linspace)
  .registerVariableOp<Tensor (const Tensor &)>("aten::log(Tensor self) -> Tensor", &VariableType::log)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::log1p(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::log1p_out)
  .registerVariableOp<Tensor (const Tensor &)>("aten::log2(Tensor self) -> Tensor", &VariableType::log2)
  .registerVariableOp<Tensor & (Tensor &)>("aten::log2_(Tensor(a!) self) -> Tensor(a!)", &VariableType::log2_)
  .registerVariableOp<Tensor & (Tensor &)>("aten::log_(Tensor(a!) self) -> Tensor(a!)", &VariableType::log_)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &)>("aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)", &VariableType::log_sigmoid_forward)
  .registerVariableOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>("aten::log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor", &VariableType::log_softmax)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>("aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::lu_solve_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, Scalar)>("aten::masked_fill(Tensor self, Tensor mask, Scalar value) -> Tensor", &VariableType::masked_fill)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>("aten::masked_fill(Tensor self, Tensor mask, Tensor value) -> Tensor", &VariableType::masked_fill)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, Scalar)>("aten::masked_fill_(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)", &VariableType::masked_fill_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::masked_fill_(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)", &VariableType::masked_fill_)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>("aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor", &VariableType::masked_scatter)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)", &VariableType::masked_scatter_)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor", &VariableType::max_pool1d)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)", &VariableType::max_pool2d_with_indices)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>("aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::max_pool2d_with_indices_backward_out)
  .registerVariableOp<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))", &VariableType::max_pool3d_with_indices_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor", &VariableType::max_unpool3d_backward)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, bool)>("aten::max_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor", &VariableType::max_values)
  .registerVariableOp<Tensor (const Tensor &, c10::optional<ScalarType>)>("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", &VariableType::mean)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>("aten::mean(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", &VariableType::mean)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>("aten::median(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", &VariableType::median)
  .registerVariableOp<Tensor (const Tensor &)>("aten::median(Tensor self) -> Tensor", &VariableType::median)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>("aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)", &VariableType::miopen_batch_norm_backward)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>("aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor", &VariableType::miopen_depthwise_convolution)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>("aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor", &VariableType::mkldnn_max_pool2d)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::mm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::mm_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor", &VariableType::multi_margin_loss)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>("aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::multi_margin_loss_backward_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::mv(Tensor self, Tensor vec) -> Tensor", &VariableType::mv)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double)>("aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)", &VariableType::native_layer_norm)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)", &VariableType::nll_loss2d_forward)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>("aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)", &VariableType::nll_loss_forward)
  .registerVariableOp<Tensor (const Tensor &, int64_t, int64_t)>("aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor", &VariableType::norm_except_dim)
  .registerVariableOp<Tensor (IntArrayRef, const TensorOptions &)>("aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::ones)
  .registerVariableOp<Tensor (const Tensor &)>("aten::pin_memory(Tensor self) -> Tensor", &VariableType::pin_memory)
  .registerVariableOp<Tensor (int64_t, const Tensor &)>("aten::polygamma(int n, Tensor self) -> Tensor", &VariableType::polygamma)
  .registerVariableOp<Tensor & (Tensor &, int64_t)>("aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)", &VariableType::polygamma_)
  .registerVariableOp<Tensor (const Tensor &, Scalar)>("aten::pow(Tensor self, Scalar exponent) -> Tensor", &VariableType::pow)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::pow(Tensor self, Tensor exponent) -> Tensor", &VariableType::pow)
  .registerVariableOp<Tensor (Scalar, const Tensor &)>("aten::pow(Scalar self, Tensor exponent) -> Tensor", &VariableType::pow)
  .registerVariableOp<Tensor & (Tensor &, Scalar)>("aten::pow_(Tensor(a!) self, Scalar exponent) -> Tensor(a!)", &VariableType::pow_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::pow_(Tensor(a!) self, Tensor exponent) -> Tensor(a!)", &VariableType::pow_)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::prelu(Tensor self, Tensor weight) -> Tensor", &VariableType::prelu)
  .registerVariableOp<Tensor (const Tensor &, c10::optional<ScalarType>)>("aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor", &VariableType::prod)
  .registerVariableOp<Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>)>("aten::prod(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", &VariableType::prod)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>("aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)", &VariableType::qr)
  .registerVariableOp<QScheme (const Tensor &)>("aten::qscheme(Tensor self) -> QScheme", &VariableType::qscheme)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, ScalarType)>("aten::quantize_linear_per_channel(Tensor self, Tensor scales, Tensor zero_points, int[] axis, ScalarType dtype) -> Tensor", &VariableType::quantize_linear_per_channel)
  .registerVariableOp<Tensor (const Tensor &)>("aten::rand_like(Tensor self) -> Tensor", &VariableType::rand_like)
  .registerVariableOp<Tensor (const Tensor &, const TensorOptions &)>("aten::rand_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor", &VariableType::rand_like)
  .registerVariableOp<Tensor (int64_t, IntArrayRef, const TensorOptions &)>("aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::randint)
  .registerVariableOp<Tensor (int64_t, IntArrayRef, Generator *, const TensorOptions &)>("aten::randint(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::randint)
  .registerVariableOp<Tensor (int64_t, int64_t, IntArrayRef, const TensorOptions &)>("aten::randint(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::randint)
  .registerVariableOp<Tensor (int64_t, int64_t, IntArrayRef, Generator *, const TensorOptions &)>("aten::randint(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::randint)
  .registerVariableOp<Tensor (const Tensor &)>("aten::randn_like(Tensor self) -> Tensor", &VariableType::randn_like)
  .registerVariableOp<Tensor (const Tensor &, const TensorOptions &)>("aten::randn_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor", &VariableType::randn_like)
  .registerVariableOp<Tensor (int64_t, const TensorOptions &)>("aten::randperm(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::randperm)
  .registerVariableOp<Tensor (int64_t, Generator *, const TensorOptions &)>("aten::randperm(int n, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::randperm)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, IntArrayRef)>("aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor", &VariableType::reflection_pad2d_backward)
  .registerVariableOp<Tensor (const Tensor &)>("aten::repeat_interleave(Tensor repeats) -> Tensor", &VariableType::repeat_interleave)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>("aten::repeat_interleave(Tensor self, Tensor repeats, int? dim=None) -> Tensor", &VariableType::repeat_interleave)
  .registerVariableOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>)>("aten::repeat_interleave(Tensor self, int repeats, int? dim=None) -> Tensor", &VariableType::repeat_interleave)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, IntArrayRef)>("aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor", &VariableType::replication_pad1d_backward)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef)>("aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor", &VariableType::replication_pad2d)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>("aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::replication_pad2d_backward_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, IntArrayRef)>("aten::replication_pad3d(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::replication_pad3d_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>("aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor", &VariableType::rnn_tanh_cell)
  .registerVariableOp<Tensor (const Tensor &)>("aten::round(Tensor self) -> Tensor", &VariableType::round)
  .registerVariableOp<Tensor & (Tensor &)>("aten::round_(Tensor(a!) self) -> Tensor(a!)", &VariableType::round_)
  .registerVariableOp<Tensor (const Tensor &)>("aten::rsqrt(Tensor self) -> Tensor", &VariableType::rsqrt)
  .registerVariableOp<Tensor & (Tensor &)>("aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)", &VariableType::rsqrt_)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("aten::s_native_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", &VariableType::s_native_addmm)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("aten::s_native_addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)", &VariableType::s_native_addmm_)
  .registerVariableOp<Tensor (Scalar, const TensorOptions &)>("aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::scalar_tensor)
  .registerVariableOp<Tensor & (Tensor &, Storage)>("aten::set_(Tensor(a!) self, Storage source) -> Tensor(a!)", &VariableType::set_)
  .registerVariableOp<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>("aten::set_(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)", &VariableType::set_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::set_(Tensor(a!) self, Tensor source) -> Tensor(a!)", &VariableType::set_)
  .registerVariableOp<Tensor & (Tensor &)>("aten::set_(Tensor(a!) self) -> Tensor(a!)", &VariableType::set_)
  .registerVariableOp<Tensor & (Tensor &, ConstQuantizerPtr)>("aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)", &VariableType::set_quantizer_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::sigmoid(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::sigmoid_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::sign(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::sign_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, int64_t)>("aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor", &VariableType::smooth_l1_loss)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>("aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::smooth_l1_loss_backward_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>("aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor", &VariableType::soft_margin_loss_backward)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>("aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor", &VariableType::softplus_backward)
  .registerVariableOp<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &)>("aten::solve(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)", &VariableType::solve_out)
  .registerVariableOp<Tensor (IntArrayRef, const TensorOptions &)>("aten::sparse_coo_tensor(int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor", &VariableType::sparse_coo_tensor)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const TensorOptions &)>("aten::sparse_coo_tensor(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::sparse_coo_tensor)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &)>("aten::sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::sparse_coo_tensor)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::sparse_mask(Tensor self, Tensor mask) -> Tensor", &VariableType::sparse_mask)
  .registerVariableOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>("aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)", &VariableType::sparse_resize_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::sqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::sqrt_out)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>("aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", &VariableType::sspaddmm)
  .registerVariableOp<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool)>("aten::svd(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)", &VariableType::svd_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &)>("aten::tan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::tan_out)
  .registerVariableOp<Tensor (const Tensor &)>("aten::tanh(Tensor self) -> Tensor", &VariableType::tanh)
  .registerVariableOp<Tensor & (Tensor &)>("aten::tanh_(Tensor(a!) self) -> Tensor(a!)", &VariableType::tanh_)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>("aten::tanh_backward(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::tanh_backward_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>("aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::thnn_conv2d_out)
  .registerVariableOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>)>("aten::thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)", &VariableType::thnn_conv_depthwise2d_backward)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>("aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::thnn_conv_depthwise2d_forward_out)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>("aten::threshold(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::threshold_out)
  .registerVariableOp<Tensor (const Tensor &, const TensorOptions &, bool, bool)>("aten::to(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor", &VariableType::to)
  .registerVariableOp<Tensor (const Tensor &, Device, ScalarType, bool, bool)>("aten::to(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor", &VariableType::to)
  .registerVariableOp<Tensor (const Tensor &, ScalarType, bool, bool)>("aten::to(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor", &VariableType::to)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &, bool, bool)>("aten::to(Tensor self, Tensor other, bool non_blocking=False, bool copy=False) -> Tensor", &VariableType::to)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor", &VariableType::to_mkldnn_backward)
  .registerVariableOp<Tensor (const Tensor &)>("aten::trace(Tensor self) -> Tensor", &VariableType::trace)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, int64_t)>("aten::tril(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::tril_out)
  .registerVariableOp<Tensor (const Tensor &, int64_t)>("aten::triu(Tensor self, int diagonal=0) -> Tensor", &VariableType::triu)
  .registerVariableOp<Tensor & (Tensor &, int64_t)>("aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)", &VariableType::triu_)
  .registerVariableOp<Tensor (int64_t, int64_t, int64_t, const TensorOptions &)>("aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", &VariableType::triu_indices)
  .registerVariableOp<Tensor (const Tensor &, const Tensor &)>("aten::type_as(Tensor self, Tensor other) -> Tensor", &VariableType::type_as)
  .registerVariableOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, int64_t, bool, bool, bool)>("aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)", &VariableType::unique_dim)
  .registerVariableOp<Tensor (const Tensor &, int64_t)>("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", &VariableType::unsqueeze)
  .registerVariableOp<Tensor & (Tensor &, int64_t)>("aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)", &VariableType::unsqueeze_)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, bool)>("aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor", &VariableType::upsample_bicubic2d)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::upsample_bicubic2d_backward_out)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef)>("aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size) -> Tensor", &VariableType::upsample_nearest3d_backward)
  .registerVariableOp<Tensor (const Tensor &, IntArrayRef, bool)>("aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners) -> Tensor", &VariableType::upsample_trilinear3d)
  .registerVariableOp<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool)>("aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)", &VariableType::upsample_trilinear3d_backward_out)
  .registerVariableOp<Tensor & (Tensor &, IntArrayRef)>("aten::zeros(int[] size, *, Tensor(a!) out) -> Tensor(a!)", &VariableType::zeros_out);
}} // namespace torch::autograd
