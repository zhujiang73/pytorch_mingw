#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

// @generated from tools\autograd\templates/ProfiledType.cpp

// NOTE See [Sharded File] comment in VariableType

using namespace at;
using namespace torch::autograd::generated;
using torch::autograd::Node;

namespace torch {

namespace ProfiledType {

namespace {
Tensor & __irshift___Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__irshift__", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("__irshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __irshift___Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__irshift__", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__irshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __rshift___Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__rshift__", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("__rshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __rshift___Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__rshift__", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__rshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor _addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_addr", std::vector<c10::IValue>({self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, vec1, vec2, beta, alpha);
}
Tensor & _addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_addr_", std::vector<c10::IValue>({self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, vec1, vec2, beta, alpha);
}
Tensor _amp_update_scale(Tensor & growth_tracker, const Tensor & current_scale, const Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_amp_update_scale", "")
      .typed<Tensor (Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  RECORD_FUNCTION("_amp_update_scale", std::vector<c10::IValue>({growth_tracker, current_scale, found_inf}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Profiler, growth_tracker, current_scale, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
}
Tensor _bmm(const Tensor & self, const Tensor & mat2, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_bmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_bmm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, mat2, deterministic);
}
Tensor _cast_Byte(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Byte", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Byte", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Half(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Half", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Half", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Int(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Int", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Int", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor & _cat_out_out(Tensor & out, TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cat", "out")
      .typed<Tensor & (Tensor &, TensorList, int64_t)>();
  RECORD_FUNCTION("_cat_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, int64_t>(op, c10::DispatchKey::Profiler, out, tensors, dim);
}
Tensor _cdist_backward(const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cdist_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, const Tensor &)>();
  RECORD_FUNCTION("_cdist_backward", std::vector<c10::IValue>({grad, x1, x2, cdist}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Profiler, grad, x1, x2, p, cdist);
}
Tensor _cholesky_helper(const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cholesky_helper", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cholesky_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, upper);
}
Tensor & _coalesced_(Tensor & self, bool coalesced) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_coalesced_", "")
      .typed<Tensor & (Tensor &, bool)>();
  RECORD_FUNCTION("_coalesced_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, bool>(op, c10::DispatchKey::Profiler, self, coalesced);
}
Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool)>();
  RECORD_FUNCTION("_convolution", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>)>();
  RECORD_FUNCTION("_cudnn_rnn_backward", std::vector<c10::IValue>({input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, dropout_state, reserve}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}
int64_t _cufft_get_plan_cache_size(int64_t device_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_get_plan_cache_size", "")
      .typed<int64_t (int64_t)>();
  RECORD_FUNCTION("_cufft_get_plan_cache_size", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, int64_t>(op, c10::DispatchKey::Profiler, device_index);
}
void _cummax_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cummax_helper", "")
      .typed<void (const Tensor &, Tensor &, Tensor &, int64_t)>();
  RECORD_FUNCTION("_cummax_helper", std::vector<c10::IValue>({self, values, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &, Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, values, indices, dim);
}
Tensor & _cumprod_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumprod", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_cumprod_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
}
Tensor _cumsum(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumsum", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("_cumsum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor _dim_arange(const Tensor & like, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dim_arange", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("_dim_arange", std::vector<c10::IValue>({like}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, like, dim);
}
Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dirichlet_grad", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_dirichlet_grad", std::vector<c10::IValue>({x, alpha, total}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, x, alpha, total);
}
Tensor _embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &)>();
  RECORD_FUNCTION("_embedding_bag_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, maximum_indices, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
}
std::tuple<Tensor,Tensor> _fused_dropout(const Tensor & self, double p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_fused_dropout", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("_fused_dropout", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
Tensor _logcumsumexp(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_logcumsumexp", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("_logcumsumexp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor _make_per_tensor_quantized_tensor(const Tensor & self, double scale, int64_t zero_point) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_make_per_tensor_quantized_tensor", "")
      .typed<Tensor (const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("_make_per_tensor_quantized_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point);
}
std::tuple<Tensor,Tensor> _mode(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mode", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_mode", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor _nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("_nnpack_spatial_convolution", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, padding, stride);
}
Tensor _softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output, dim, self);
}
Tensor _sparse_log_softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax", "int")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("_sparse_log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor _sparse_log_softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("_sparse_log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor _sparse_log_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_sparse_log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, half_to_float);
}
Tensor _sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_sparse_sum_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad, self, dim);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_gru_cell_backward(const Tensor & grad_hy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_differentiable_gru_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_thnn_differentiable_gru_cell_backward", std::vector<c10::IValue>({grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward(const Tensor & grad_hy, const Tensor & workspace, bool has_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_gru_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_thnn_fused_gru_cell_backward", std::vector<c10::IValue>({grad_hy, workspace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, grad_hy, workspace, has_bias);
}
bool _use_cudnn_rnn_flatten_weight() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_use_cudnn_rnn_flatten_weight", "")
      .typed<bool ()>();
  RECORD_FUNCTION("_use_cudnn_rnn_flatten_weight", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Profiler);
}
Tensor _values(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_values", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_cuda_interface_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm_cuda_interface_backward", std::vector<c10::IValue>({grad_w, saved_v, saved_g, saved_norms}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_w, saved_v, saved_g, saved_norms, dim);
}
Tensor & acos_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acos", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("acos_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor acosh(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acosh", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("acosh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & acosh_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acosh_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("acosh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor & adaptive_avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, output_size);
}
Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, indices);
}
Tensor & add_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("add_out", std::vector<c10::IValue>({out, self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other, alpha);
}
Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addr", std::vector<c10::IValue>({self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, vec1, vec2, beta, alpha);
}
Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addr_", std::vector<c10::IValue>({self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, vec1, vec2, beta, alpha);
}
Tensor affine_grid_generator(const Tensor & theta, IntArrayRef size, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::affine_grid_generator", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("affine_grid_generator", std::vector<c10::IValue>({theta}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, theta, size, align_corners);
}
Tensor & arange_out_out(Tensor & out, Scalar end) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "out")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("arange_out", std::vector<c10::IValue>({out, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, end);
}
Tensor & arange_out_start_out(Tensor & out, Scalar start, Scalar end, Scalar step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "start_out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("arange_out", std::vector<c10::IValue>({out, start, end, step}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, start, end, step);
}
Tensor & asin_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asin", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("asin_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor asinh(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asinh", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("asinh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & asinh_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asinh_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("asinh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
void backward(const Tensor & self, const Tensor & gradient, c10::optional<bool> retain_graph, bool create_graph) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::backward", "")
      .typed<void (const Tensor &, const Tensor &, c10::optional<bool>, bool)>();
  RECORD_FUNCTION("backward", std::vector<c10::IValue>({self, gradient}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &, const Tensor &, c10::optional<bool>, bool>(op, c10::DispatchKey::Profiler, self, gradient, retain_graph, create_graph);
}
Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::baddbmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("baddbmm", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, batch1, batch2, beta, alpha);
}
Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::baddbmm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("baddbmm_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, batch1, batch2, beta, alpha);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, bool input_g, bool weight_g, bool bias_g) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_backward_reduce", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("batch_norm_backward_reduce", std::vector<c10::IValue>({grad_out, input, mean, invstd, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}
Tensor & bernoulli_out_out(Tensor & out, const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli", "out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("bernoulli_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, self, generator);
}
Tensor binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_with_logits", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_with_logits", std::vector<c10::IValue>({self, target, weight, pos_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, pos_weight, reduction);
}
Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bincount", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("bincount", std::vector<c10::IValue>({self, weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, weights, minlength);
}
Tensor bitwise_and_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_and", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor bitwise_and_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_and", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & bitwise_and__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_and_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & bitwise_and__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_and_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor bitwise_not(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("bitwise_not", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & bitwise_not_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("bitwise_not_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor bmm(const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bmm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bmm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mat2);
}
Tensor bucketize_Tensor(const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("bucketize", std::vector<c10::IValue>({self, boundaries}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, boundaries, out_int32, right);
}
Tensor bucketize_Scalar(Scalar self, const Tensor & boundaries, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Scalar")
      .typed<Tensor (Scalar, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("bucketize", std::vector<c10::IValue>({self, boundaries}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, boundaries, out_int32, right);
}
Tensor cartesian_prod(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cartesian_prod", "")
      .typed<Tensor (TensorList)>();
  RECORD_FUNCTION("cartesian_prod", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
Tensor & cat_out_out(Tensor & out, TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cat", "out")
      .typed<Tensor & (Tensor &, TensorList, int64_t)>();
  RECORD_FUNCTION("cat_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, int64_t>(op, c10::DispatchKey::Profiler, out, tensors, dim);
}
Tensor & cat_out_names_out(Tensor & out, TensorList tensors, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cat", "names_out")
      .typed<Tensor & (Tensor &, TensorList, Dimname)>();
  RECORD_FUNCTION("cat_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, Dimname>(op, c10::DispatchKey::Profiler, out, tensors, dim);
}
Tensor chain_matmul(TensorList matrices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::chain_matmul", "")
      .typed<Tensor (TensorList)>();
  RECORD_FUNCTION("chain_matmul", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Profiler, matrices);
}
Tensor cholesky(const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, upper);
}
Tensor clamp_max(const Tensor & self, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_max", std::vector<c10::IValue>({self, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, max);
}
Tensor & clamp_max_(Tensor & self, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_max_", std::vector<c10::IValue>({self, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, max);
}
Tensor coalesce(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::coalesce", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("coalesce", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & col2im_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, output_size, kernel_size, dilation, padding, stride);
}
Tensor combinations(const Tensor & self, int64_t r, bool with_replacement) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::combinations", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("combinations", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, r, with_replacement);
}
Tensor conj(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conj", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("conj", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_tbc", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("conv_tbc", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, weight, bias, pad);
}
Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("convolution", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
Tensor cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosine_similarity", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, double)>();
  RECORD_FUNCTION("cosine_similarity", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, double>(op, c10::DispatchKey::Profiler, x1, x2, dim, eps);
}
Tensor cudnn_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor> cummax(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t)>();
  RECORD_FUNCTION("cummax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
std::tuple<Tensor,Tensor> cummax_dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname)>();
  RECORD_FUNCTION("cummax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & cumprod_out_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumprod", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumprod_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, dtype);
}
Tensor & cumprod_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumprod", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumprod_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, dtype);
}
Tensor cumsum(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumsum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor cumsum_dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumsum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
int64_t dense_dim(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dense_dim", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("dense_dim", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diagonal", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("diagonal", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, offset, dim1, dim2);
}
Tensor diagonal_Dimname(const Tensor & self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diagonal", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, Dimname, Dimname, int64_t)>();
  RECORD_FUNCTION("diagonal", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, Dimname, Dimname, int64_t>(op, c10::DispatchKey::Profiler, self, outdim, dim1, dim2, offset);
}
Tensor dist(const Tensor & self, const Tensor & other, Scalar p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dist", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("dist", std::vector<c10::IValue>({self, other, p}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, p);
}
Tensor & dot_out_out(Tensor & out, const Tensor & self, const Tensor & tensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dot", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("dot_out", std::vector<c10::IValue>({out, self, tensor}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, tensor);
}
Tensor dropout(const Tensor & input, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dropout", "")
      .typed<Tensor (const Tensor &, double, bool)>();
  RECORD_FUNCTION("dropout", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, p, train);
}
Tensor & dropout_(Tensor & self, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dropout_", "")
      .typed<Tensor & (Tensor &, double, bool)>();
  RECORD_FUNCTION("dropout_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, bool>(op, c10::DispatchKey::Profiler, self, p, train);
}
Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("elu", std::vector<c10::IValue>({self, alpha, scale, input_scale}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, alpha, scale, input_scale);
}
Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_", "")
      .typed<Tensor & (Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("elu_", std::vector<c10::IValue>({self, alpha, scale, input_scale}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, alpha, scale, input_scale);
}
Tensor & elu_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("elu_backward_out", std::vector<c10::IValue>({grad_input, grad_output, alpha, scale, input_scale, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, alpha, scale, input_scale, output);
}
Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_renorm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, double, double)>();
  RECORD_FUNCTION("embedding_renorm_", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, double, double>(op, c10::DispatchKey::Profiler, self, indices, max_norm, norm_type);
}
bool equal(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::equal", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("equal", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & erf_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erf", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("erf_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor erfc(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfc", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("erfc", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & erfc_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfc_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("erfc_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor expm1(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("expm1", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & expm1_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("expm1_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & exponential_(Tensor & self, double lambd, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exponential_", "")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("exponential_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, lambd, generator);
}
Tensor fake_quantize_per_channel_affine(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_channel_affine", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_channel_affine", std::vector<c10::IValue>({self, scale, zero_point}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point, axis, quant_min, quant_max);
}
Tensor fbgemm_linear_fp16_weight_fp32_activation(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_fp16_weight_fp32_activation", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_fp16_weight_fp32_activation", std::vector<c10::IValue>({input, packed_weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, packed_weight, bias);
}
Tensor fbgemm_linear_int8_weight_fp32_activation(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_int8_weight_fp32_activation", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_int8_weight_fp32_activation", std::vector<c10::IValue>({input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}
std::tuple<Tensor,Tensor,double,int64_t> fbgemm_linear_quantize_weight(const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_quantize_weight", "")
      .typed<std::tuple<Tensor,Tensor,double,int64_t> (const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_quantize_weight", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,double,int64_t>, const Tensor &>(op, c10::DispatchKey::Profiler, input);
}
Tensor floor(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("floor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & floor_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("floor_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & fmod_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("fmod_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & fmod_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fmod_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & frac_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frac", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("frac_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out_output(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d_out", std::vector<c10::IValue>({output, indices, self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, output, indices, self, kernel_size, output_size, random_samples);
}
Tensor & frobenius_norm_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frobenius_norm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("frobenius_norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full_like", "")
      .typed<Tensor (const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("full_like", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, fill_value, options, memory_format);
}
Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::group_norm", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool)>();
  RECORD_FUNCTION("group_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, num_groups, weight, bias, eps, cudnn_enabled);
}
Tensor hamming_window(int64_t window_length, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("hamming_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, options);
}
Tensor hamming_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "periodic")
      .typed<Tensor (int64_t, bool, const TensorOptions &)>();
  RECORD_FUNCTION("hamming_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, periodic, options);
}
Tensor hamming_window_periodic_alpha(int64_t window_length, bool periodic, double alpha, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "periodic_alpha")
      .typed<Tensor (int64_t, bool, double, const TensorOptions &)>();
  RECORD_FUNCTION("hamming_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, double, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, periodic, alpha, options);
}
Tensor hamming_window_periodic_alpha_beta(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hamming_window", "periodic_alpha_beta")
      .typed<Tensor (int64_t, bool, double, double, const TensorOptions &)>();
  RECORD_FUNCTION("hamming_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, double, double, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, periodic, alpha, beta, options);
}
Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardshrink_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("hardshrink_backward", std::vector<c10::IValue>({grad_out, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_out, self, lambd);
}
Tensor & hardtanh_out_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_out", std::vector<c10::IValue>({out, self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, min_val, max_val);
}
Tensor & im2col_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, kernel_size, dilation, padding, stride);
}
Tensor index_Tensor(const Tensor & self, TensorList indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index", "Tensor")
      .typed<Tensor (const Tensor &, TensorList)>();
  RECORD_FUNCTION("index", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, TensorList>(op, c10::DispatchKey::Profiler, self, indices);
}
Tensor index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_put", "")
      .typed<Tensor (const Tensor &, TensorList, const Tensor &, bool)>();
  RECORD_FUNCTION("index_put", std::vector<c10::IValue>({self, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, TensorList, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, indices, values, accumulate);
}
Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_put_", "")
      .typed<Tensor & (Tensor &, TensorList, const Tensor &, bool)>();
  RECORD_FUNCTION("index_put_", std::vector<c10::IValue>({self, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, indices, values, accumulate);
}
Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("index_select", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index);
}
Tensor index_select_dimname(const Tensor & self, Dimname dim, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &)>();
  RECORD_FUNCTION("index_select", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index);
}
bool is_coalesced(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_coalesced", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_coalesced", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_floating_point(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_floating_point", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_floating_point", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_vulkan_available() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_vulkan_available", "")
      .typed<bool ()>();
  RECORD_FUNCTION("is_vulkan_available", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Profiler);
}
Scalar item(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::item", "")
      .typed<Scalar (const Tensor &)>();
  RECORD_FUNCTION("item", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor & l1_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
}
Tensor & linspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linspace", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, int64_t)>();
  RECORD_FUNCTION("linspace_out", std::vector<c10::IValue>({out, start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, int64_t>(op, c10::DispatchKey::Profiler, out, start, end, steps);
}
Tensor & log2_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log2", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log2_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & log_normal_(Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_normal_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("log_normal_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, mean, std, generator);
}
Tensor & log_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_backward", std::vector<c10::IValue>({grad_output, self, buffer}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, buffer);
}
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out_output(Tensor & output, Tensor & buffer, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_forward_out", std::vector<c10::IValue>({output, buffer, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, output, buffer, self);
}
Tensor & logaddexp2_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logaddexp2", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logaddexp2_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & logaddexp_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logaddexp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logaddexp_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor logcumsumexp(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("logcumsumexp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor logcumsumexp_dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "dimname")
      .typed<Tensor (const Tensor &, Dimname)>();
  RECORD_FUNCTION("logcumsumexp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor logical_or(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_or", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & logical_or_(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or_", "")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_or_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor logical_xor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_xor", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & logical_xor_(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor_", "")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_xor_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logspace", "")
      .typed<Tensor (Scalar, Scalar, int64_t, double, const TensorOptions &)>();
  RECORD_FUNCTION("logspace", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, int64_t, double, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, steps, base, options);
}
Tensor logsumexp(const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("logsumexp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor logsumexp_names(const Tensor & self, DimnameList dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "names")
      .typed<Tensor (const Tensor &, DimnameList, bool)>();
  RECORD_FUNCTION("logsumexp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> lstsq_out_X(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstsq", "X")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lstsq_out", std::vector<c10::IValue>({X, qr, self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, X, qr, self, A);
}
Tensor matmul(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matmul", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("matmul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
std::tuple<Tensor,Tensor> max_dim(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("max", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> max_names_dim(const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("max", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor max_other(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "other")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("max", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor max(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("max", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool1d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool1d_with_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool2d_with_indices_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, indices, output_size);
}
Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, indices, output_size, stride, padding);
}
Tensor & max_unpool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices, output_size, stride, padding);
}
Tensor & mean_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("mean_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim, dtype);
}
Tensor & mean_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("mean_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim, dtype);
}
std::tuple<Tensor &,Tensor &> median_out_dim_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "dim_values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("median_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> median_out_names_dim_values(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "names_dim_values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("median_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, keepdim);
}
std::vector<Tensor> meshgrid(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::meshgrid", "")
      .typed<std::vector<Tensor> (TensorList)>();
  RECORD_FUNCTION("meshgrid", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("miopen_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>();
  RECORD_FUNCTION("miopen_convolution_transpose_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Profiler, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor miopen_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward_input", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_transpose_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor miopen_depthwise_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_depthwise_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("mkldnn_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, self, grad_output, weight, padding, stride, dilation, groups, output_mask);
}
Tensor mkldnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution_backward_input", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool)>();
  RECORD_FUNCTION("mkldnn_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool>(op, c10::DispatchKey::Profiler, self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
}
std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("mode", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> mode_dimname(const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("mode", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor & multi_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss_out", std::vector<c10::IValue>({out, self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, p, margin, weight, reduction);
}
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss_forward", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor & mv_out_out(Tensor & out, const Tensor & self, const Tensor & vec) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mv_out", std::vector<c10::IValue>({out, self, vec}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, vec);
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>)>();
  RECORD_FUNCTION("native_batch_norm_backward", std::vector<c10::IValue>({grad_out, input, weight, running_mean, running_var, save_mean, save_invstd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}
Tensor native_norm(const Tensor & self, Scalar p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_norm", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("native_norm", std::vector<c10::IValue>({self, p}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, p);
}
Tensor ne_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("ne", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor ne_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ne", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & ne__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("ne_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & ne__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ne_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss2d_backward", std::vector<c10::IValue>({grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out_output(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d_forward_out", std::vector<c10::IValue>({output, total_weight, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, output, total_weight, self, target, weight, reduction, ignore_index);
}
Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss_backward", std::vector<c10::IValue>({grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
std::tuple<Tensor &,Tensor &> nll_loss_forward_out_output(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss_forward_out", std::vector<c10::IValue>({output, total_weight, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, output, total_weight, self, target, weight, reduction, ignore_index);
}
Tensor & ones_out_out(Tensor & out, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones", "out")
      .typed<Tensor & (Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("ones_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, size);
}
Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ormqr", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("ormqr", std::vector<c10::IValue>({self, input2, input3}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, input2, input3, left, transpose);
}
Tensor pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pairwise_distance", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, double, bool)>();
  RECORD_FUNCTION("pairwise_distance", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, double, bool>(op, c10::DispatchKey::Profiler, x1, x2, p, eps, keepdim);
}
Tensor pinverse(const Tensor & self, double rcond) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pinverse", "")
      .typed<Tensor (const Tensor &, double)>();
  RECORD_FUNCTION("pinverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, rcond);
}
Tensor & polygamma_out_out(Tensor & out, int64_t n, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::polygamma", "out")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("polygamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, out, n, self);
}
Tensor & pow_out_Tensor_Scalar_out(Tensor & out, const Tensor & self, Scalar exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Tensor_Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("pow_out", std::vector<c10::IValue>({out, self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, exponent);
}
Tensor & pow_out_Tensor_Tensor_out(Tensor & out, const Tensor & self, const Tensor & exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Tensor_Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("pow_out", std::vector<c10::IValue>({out, self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, exponent);
}
Tensor & pow_out_Scalar_out(Tensor & out, Scalar self, const Tensor & exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Scalar_out")
      .typed<Tensor & (Tensor &, Scalar, const Tensor &)>();
  RECORD_FUNCTION("pow_out", std::vector<c10::IValue>({out, self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, exponent);
}
Tensor & prod_out_int_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "int_out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("prod_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim, dtype);
}
Tensor & prod_out_Dimname_out(Tensor & out, const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "Dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("prod_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim, dtype);
}
int64_t q_per_channel_axis(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_axis", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("q_per_channel_axis", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor q_per_channel_zero_points(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_zero_points", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("q_per_channel_zero_points", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor &,Tensor &> qr_out_Q(Tensor & Q, Tensor & R, const Tensor & self, bool some) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::qr", "Q")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("qr_out", std::vector<c10::IValue>({Q, R, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, Q, R, self, some);
}
Tensor quantized_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_gru_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_gru_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor quantized_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_rnn_relu_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_rnn_relu_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor rand_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("rand", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, names, options);
}
Tensor rand_generator_with_names(IntArrayRef size, c10::optional<Generator> generator, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "generator_with_names")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("rand", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, generator, names, options);
}
Tensor rand(IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("rand", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, options);
}
Tensor rand_generator(IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "generator")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  RECORD_FUNCTION("rand", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, generator, options);
}
Tensor & randint_out_out(Tensor & out, int64_t high, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "out")
      .typed<Tensor & (Tensor &, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("randint_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, out, high, size);
}
Tensor & randint_out_generator_out(Tensor & out, int64_t high, IntArrayRef size, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "generator_out")
      .typed<Tensor & (Tensor &, int64_t, IntArrayRef, c10::optional<Generator>)>();
  RECORD_FUNCTION("randint_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, high, size, generator);
}
Tensor & randint_out_low_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "low_out")
      .typed<Tensor & (Tensor &, int64_t, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("randint_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, out, low, high, size);
}
Tensor & randint_out_low_generator_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "low_generator_out")
      .typed<Tensor & (Tensor &, int64_t, int64_t, IntArrayRef, c10::optional<Generator>)>();
  RECORD_FUNCTION("randint_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, low, high, size, generator);
}
Tensor randn(IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("randn", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, options);
}
Tensor randn_generator(IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "generator")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  RECORD_FUNCTION("randn", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, generator, options);
}
Tensor randn_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("randn", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, names, options);
}
Tensor randn_generator_with_names(IntArrayRef size, c10::optional<Generator> generator, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "generator_with_names")
      .typed<Tensor (IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("randn", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<Generator>, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, generator, names, options);
}
Tensor & random__from(Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::random_", "from")
      .typed<Tensor & (Tensor &, int64_t, c10::optional<int64_t>, c10::optional<Generator>)>();
  RECORD_FUNCTION("random_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, c10::optional<int64_t>, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, from, to, generator);
}
Tensor & random__to(Tensor & self, int64_t to, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::random_", "to")
      .typed<Tensor & (Tensor &, int64_t, c10::optional<Generator>)>();
  RECORD_FUNCTION("random_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, to, generator);
}
Tensor & random_(Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::random_", "")
      .typed<Tensor & (Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("random_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, generator);
}
Tensor & randperm_out_out(Tensor & out, int64_t n) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randperm", "out")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("randperm_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, n);
}
Tensor & randperm_out_generator_out(Tensor & out, int64_t n, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randperm", "generator_out")
      .typed<Tensor & (Tensor &, int64_t, c10::optional<Generator>)>();
  RECORD_FUNCTION("randperm_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, n, generator);
}
Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor reflection_pad2d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor & reflection_pad2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor remainder_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("remainder", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor remainder_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("remainder", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & remainder__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("remainder_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & remainder__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("remainder_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor repeat(const Tensor & self, IntArrayRef repeats) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::repeat", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("repeat", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, repeats);
}
Tensor replication_pad1d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor & replication_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & replication_pad2d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor reshape_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reshape_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("reshape_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
ScalarType result_type_Tensor(const Tensor & tensor, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Tensor")
      .typed<ScalarType (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("result_type", std::vector<c10::IValue>({tensor, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<ScalarType, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, tensor, other);
}
ScalarType result_type_Scalar(const Tensor & tensor, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Scalar")
      .typed<ScalarType (const Tensor &, Scalar)>();
  RECORD_FUNCTION("result_type", std::vector<c10::IValue>({tensor, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<ScalarType, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, tensor, other);
}
ScalarType result_type_Scalar_Tensor(Scalar scalar, const Tensor & tensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Scalar_Tensor")
      .typed<ScalarType (Scalar, const Tensor &)>();
  RECORD_FUNCTION("result_type", std::vector<c10::IValue>({scalar, tensor}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<ScalarType, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, scalar, tensor);
}
ScalarType result_type_Scalar_Scalar(Scalar scalar1, Scalar scalar2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::result_type", "Scalar_Scalar")
      .typed<ScalarType (Scalar, Scalar)>();
  RECORD_FUNCTION("result_type", std::vector<c10::IValue>({scalar1, scalar2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<ScalarType, Scalar, Scalar>(op, c10::DispatchKey::Profiler, scalar1, scalar2);
}
std::tuple<Tensor,Tensor> rnn_tanh_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_tanh", "input")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>();
  RECORD_FUNCTION("rnn_tanh", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool>(op, c10::DispatchKey::Profiler, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> rnn_tanh_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_tanh", "data")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>();
  RECORD_FUNCTION("rnn_tanh", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool>(op, c10::DispatchKey::Profiler, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::roll", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("roll", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, shifts, dims);
}
Tensor rot90(const Tensor & self, int64_t k, IntArrayRef dims) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rot90", "")
      .typed<Tensor (const Tensor &, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("rot90", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, self, k, dims);
}
Tensor & round_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::round", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("round_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, bool self_is_result) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, bool)>();
  RECORD_FUNCTION("rrelu_with_noise_backward", std::vector<c10::IValue>({grad_output, self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, bool>(op, c10::DispatchKey::Profiler, grad_output, self, noise, lower, upper, training, self_is_result);
}
Tensor & rsqrt_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsqrt", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rsqrt_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & searchsorted_out_Tensor_out(Tensor & out, const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::searchsorted", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("searchsorted_out", std::vector<c10::IValue>({out, sorted_sequence, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, out, sorted_sequence, self, out_int32, right);
}
Tensor selu(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::selu", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("selu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & selu_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::selu_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("selu_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & slow_conv3d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding);
}
Tensor & slow_conv_transpose2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
Tensor & smooth_l1_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
}
Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("soft_margin_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor & soft_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("soft_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
}
Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("softplus", std::vector<c10::IValue>({self, beta, threshold}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, beta, threshold);
}
Tensor & softplus_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("softplus_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, beta, threshold, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, beta, threshold, output);
}
std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("sort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, descending);
}
std::tuple<Tensor,Tensor> sort_dimname(const Tensor & self, Dimname dim, bool descending) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("sort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, descending);
}
std::vector<Tensor> split_Tensor(const Tensor & self, int64_t split_size, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::split", "Tensor")
      .typed<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("split", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, split_size, dim);
}
Tensor & sspaddmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sspaddmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("sspaddmm_out", std::vector<c10::IValue>({out, self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, mat1, mat2, beta, alpha);
}
Tensor std(const Tensor & self, bool unbiased) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("std", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, unbiased);
}
Tensor std_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("std", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
Tensor std_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "names_dim")
      .typed<Tensor (const Tensor &, DimnameList, bool, bool)>();
  RECORD_FUNCTION("std", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
int64_t stride_int(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stride", "int")
      .typed<int64_t (const Tensor &, int64_t)>();
  RECORD_FUNCTION("stride", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
int64_t stride_Dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stride", "Dimname")
      .typed<int64_t (const Tensor &, Dimname)>();
  RECORD_FUNCTION("stride", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "")
      .typed<Tensor (const Tensor &, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dtype);
}
Tensor sum_dim_IntList(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "dim_IntList")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, keepdim, dtype);
}
Tensor sum_dim_DimnameList(const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "dim_DimnameList")
      .typed<Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, keepdim, dtype);
}
Tensor sum_to_size(const Tensor & self, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum_to_size", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("sum_to_size", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, size);
}
Tensor t(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::t", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("t", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & t_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::t_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("t_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor take(const Tensor & self, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::take", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("take", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, index);
}
Tensor & tanh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tanh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}
Tensor to_mkldnn(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_mkldnn", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("to_mkldnn", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor to_sparse_sparse_dim(const Tensor & self, int64_t sparse_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_sparse", "sparse_dim")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("to_sparse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, sparse_dim);
}
Tensor to_sparse(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_sparse", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("to_sparse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::topk", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("topk", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, k, dim, largest, sorted);
}
Tensor transpose_int(const Tensor & self, int64_t dim0, int64_t dim1) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::transpose", "int")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("transpose", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim0, dim1);
}
Tensor transpose_Dimname(const Tensor & self, Dimname dim0, Dimname dim1) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::transpose", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, Dimname)>();
  RECORD_FUNCTION("transpose", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, Dimname>(op, c10::DispatchKey::Profiler, self, dim0, dim1);
}
Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::transpose_", "")
      .typed<Tensor & (Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("transpose_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim0, dim1);
}
Tensor trapz_x(const Tensor & y, const Tensor & x, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trapz", "x")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("trapz", std::vector<c10::IValue>({y, x}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, y, x, dim);
}
Tensor trapz_dx(const Tensor & y, double dx, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trapz", "dx")
      .typed<Tensor (const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("trapz", std::vector<c10::IValue>({y}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, y, dx, dim);
}
Tensor & triu_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("triu_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, diagonal);
}
Tensor & true_divide_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::true_divide", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("true_divide_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor trunc(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("trunc", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & trunc_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("trunc_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & upsample_bicubic2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, align_corners, scales_h, scales_w);
}
Tensor upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales);
}
Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, scales_h, scales_w);
}
Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, scales_d, scales_h, scales_w);
}
Tensor & upsample_nearest3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}
Tensor & upsample_trilinear3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, align_corners, scales_d, scales_h, scales_w);
}
Tensor values(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::values", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> var_mean(const Tensor & self, bool unbiased) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var_mean", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("var_mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, unbiased);
}
std::tuple<Tensor,Tensor> var_mean_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var_mean", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("var_mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
std::tuple<Tensor,Tensor> var_mean_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var_mean", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, DimnameList, bool, bool)>();
  RECORD_FUNCTION("var_mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
}  // namespace
}  // namespace ProfiledType

namespace {

TORCH_LIBRARY_IMPL(aten, Profiler, m) {
  m.impl_UNBOXED("__irshift__.Scalar", &ProfiledType::__irshift___Scalar);
  m.impl_UNBOXED("__irshift__.Tensor", &ProfiledType::__irshift___Tensor);
  m.impl("__rshift__.Scalar", TORCH_FN(ProfiledType::__rshift___Scalar));
  m.impl("__rshift__.Tensor", TORCH_FN(ProfiledType::__rshift___Tensor));
  m.impl("_adaptive_avg_pool2d", TORCH_FN(ProfiledType::_adaptive_avg_pool2d));
  m.impl("_addr", TORCH_FN(ProfiledType::_addr));
  m.impl_UNBOXED("_addr_", &ProfiledType::_addr_);
  m.impl_UNBOXED("_amp_update_scale", &ProfiledType::_amp_update_scale);
  m.impl("_bmm", TORCH_FN(ProfiledType::_bmm));
  m.impl("_cast_Byte", TORCH_FN(ProfiledType::_cast_Byte));
  m.impl("_cast_Half", TORCH_FN(ProfiledType::_cast_Half));
  m.impl("_cast_Int", TORCH_FN(ProfiledType::_cast_Int));
  m.impl_UNBOXED("_cat.out", &ProfiledType::_cat_out_out);
  m.impl("_cdist_backward", TORCH_FN(ProfiledType::_cdist_backward));
  m.impl("_cholesky_helper", TORCH_FN(ProfiledType::_cholesky_helper));
  m.impl_UNBOXED("_coalesced_", &ProfiledType::_coalesced_);
  m.impl_UNBOXED("_convolution", &ProfiledType::_convolution);
  m.impl_UNBOXED("_cudnn_rnn_backward", &ProfiledType::_cudnn_rnn_backward);
  m.impl("_cufft_get_plan_cache_size", TORCH_FN(ProfiledType::_cufft_get_plan_cache_size));
  m.impl_UNBOXED("_cummax_helper", &ProfiledType::_cummax_helper);
  m.impl_UNBOXED("_cumprod.out", &ProfiledType::_cumprod_out_out);
  m.impl("_cumsum", TORCH_FN(ProfiledType::_cumsum));
  m.impl("_dim_arange", TORCH_FN(ProfiledType::_dim_arange));
  m.impl("_dirichlet_grad", TORCH_FN(ProfiledType::_dirichlet_grad));
  m.impl_UNBOXED("_embedding_bag_backward", &ProfiledType::_embedding_bag_backward);
  m.impl_UNBOXED("_fused_dropout", &ProfiledType::_fused_dropout);
  m.impl("_logcumsumexp", TORCH_FN(ProfiledType::_logcumsumexp));
  m.impl("_make_per_tensor_quantized_tensor", TORCH_FN(ProfiledType::_make_per_tensor_quantized_tensor));
  m.impl("_mode", TORCH_FN(ProfiledType::_mode));
  m.impl_UNBOXED("_nnpack_spatial_convolution", &ProfiledType::_nnpack_spatial_convolution);
  m.impl("_softmax_backward_data", TORCH_FN(ProfiledType::_softmax_backward_data));
  m.impl_UNBOXED("_sparse_log_softmax.int", &ProfiledType::_sparse_log_softmax_int);
  m.impl_UNBOXED("_sparse_log_softmax.Dimname", &ProfiledType::_sparse_log_softmax_Dimname);
  m.impl("_sparse_log_softmax", TORCH_FN(ProfiledType::_sparse_log_softmax));
  m.impl("_sparse_sum_backward", TORCH_FN(ProfiledType::_sparse_sum_backward));
  m.impl_UNBOXED("_thnn_differentiable_gru_cell_backward", &ProfiledType::_thnn_differentiable_gru_cell_backward);
  m.impl("_thnn_fused_gru_cell_backward", TORCH_FN(ProfiledType::_thnn_fused_gru_cell_backward));
  m.impl("_use_cudnn_rnn_flatten_weight", TORCH_FN(ProfiledType::_use_cudnn_rnn_flatten_weight));
  m.impl("_values", TORCH_FN(ProfiledType::_values));
  m.impl("_weight_norm_cuda_interface_backward", TORCH_FN(ProfiledType::_weight_norm_cuda_interface_backward));
  m.impl_UNBOXED("acos.out", &ProfiledType::acos_out_out);
  m.impl("acosh", TORCH_FN(ProfiledType::acosh));
  m.impl_UNBOXED("acosh_", &ProfiledType::acosh_);
  m.impl("adaptive_avg_pool2d", TORCH_FN(ProfiledType::adaptive_avg_pool2d));
  m.impl_UNBOXED("adaptive_avg_pool3d.out", &ProfiledType::adaptive_avg_pool3d_out_out);
  m.impl("adaptive_max_pool3d_backward", TORCH_FN(ProfiledType::adaptive_max_pool3d_backward));
  m.impl_UNBOXED("add.out", &ProfiledType::add_out_out);
  m.impl("addr", TORCH_FN(ProfiledType::addr));
  m.impl_UNBOXED("addr_", &ProfiledType::addr_);
  m.impl("affine_grid_generator", TORCH_FN(ProfiledType::affine_grid_generator));
  m.impl_UNBOXED("arange.out", &ProfiledType::arange_out_out);
  m.impl_UNBOXED("arange.start_out", &ProfiledType::arange_out_start_out);
  m.impl_UNBOXED("asin.out", &ProfiledType::asin_out_out);
  m.impl("asinh", TORCH_FN(ProfiledType::asinh));
  m.impl_UNBOXED("asinh_", &ProfiledType::asinh_);
  m.impl("avg_pool2d_backward", TORCH_FN(ProfiledType::avg_pool2d_backward));
  m.impl("avg_pool3d", TORCH_FN(ProfiledType::avg_pool3d));
  m.impl_UNBOXED("avg_pool3d_backward.grad_input", &ProfiledType::avg_pool3d_backward_out_grad_input);
  m.impl_UNBOXED("backward", &ProfiledType::backward);
  m.impl("baddbmm", TORCH_FN(ProfiledType::baddbmm));
  m.impl_UNBOXED("baddbmm_", &ProfiledType::baddbmm_);
  m.impl_UNBOXED("batch_norm_backward_reduce", &ProfiledType::batch_norm_backward_reduce);
  m.impl_UNBOXED("bernoulli.out", &ProfiledType::bernoulli_out_out);
  m.impl_UNBOXED("binary_cross_entropy_with_logits", &ProfiledType::binary_cross_entropy_with_logits);
  m.impl_UNBOXED("bincount", &ProfiledType::bincount);
  m.impl("bitwise_and.Scalar", TORCH_FN(ProfiledType::bitwise_and_Scalar));
  m.impl("bitwise_and.Tensor", TORCH_FN(ProfiledType::bitwise_and_Tensor));
  m.impl_UNBOXED("bitwise_and_.Scalar", &ProfiledType::bitwise_and__Scalar);
  m.impl_UNBOXED("bitwise_and_.Tensor", &ProfiledType::bitwise_and__Tensor);
  m.impl("bitwise_not", TORCH_FN(ProfiledType::bitwise_not));
  m.impl_UNBOXED("bitwise_not_", &ProfiledType::bitwise_not_);
  m.impl("bmm", TORCH_FN(ProfiledType::bmm));
  m.impl("bucketize.Tensor", TORCH_FN(ProfiledType::bucketize_Tensor));
  m.impl("bucketize.Scalar", TORCH_FN(ProfiledType::bucketize_Scalar));
  m.impl("cartesian_prod", TORCH_FN(ProfiledType::cartesian_prod));
  m.impl_UNBOXED("cat.out", &ProfiledType::cat_out_out);
  m.impl_UNBOXED("cat.names_out", &ProfiledType::cat_out_names_out);
  m.impl("chain_matmul", TORCH_FN(ProfiledType::chain_matmul));
  m.impl("cholesky", TORCH_FN(ProfiledType::cholesky));
  m.impl("clamp_max", TORCH_FN(ProfiledType::clamp_max));
  m.impl_UNBOXED("clamp_max_", &ProfiledType::clamp_max_);
  m.impl("coalesce", TORCH_FN(ProfiledType::coalesce));
  m.impl_UNBOXED("col2im.out", &ProfiledType::col2im_out_out);
  m.impl("combinations", TORCH_FN(ProfiledType::combinations));
  m.impl("conj", TORCH_FN(ProfiledType::conj));
  m.impl("conv_tbc", TORCH_FN(ProfiledType::conv_tbc));
  m.impl_UNBOXED("convolution", &ProfiledType::convolution);
  m.impl("cosine_similarity", TORCH_FN(ProfiledType::cosine_similarity));
  m.impl("cudnn_convolution_backward_weight", TORCH_FN(ProfiledType::cudnn_convolution_backward_weight));
  m.impl("cummax", TORCH_FN(ProfiledType::cummax));
  m.impl_UNBOXED("cummax.dimname", &ProfiledType::cummax_dimname);
  m.impl_UNBOXED("cumprod.out", &ProfiledType::cumprod_out_out);
  m.impl_UNBOXED("cumprod.dimname_out", &ProfiledType::cumprod_out_dimname_out);
  m.impl_UNBOXED("cumsum", &ProfiledType::cumsum);
  m.impl_UNBOXED("cumsum.dimname", &ProfiledType::cumsum_dimname);
  m.impl("dense_dim", TORCH_FN(ProfiledType::dense_dim));
  m.impl("diagonal", TORCH_FN(ProfiledType::diagonal));
  m.impl_UNBOXED("diagonal.Dimname", &ProfiledType::diagonal_Dimname);
  m.impl("dist", TORCH_FN(ProfiledType::dist));
  m.impl_UNBOXED("dot.out", &ProfiledType::dot_out_out);
  m.impl("dropout", TORCH_FN(ProfiledType::dropout));
  m.impl_UNBOXED("dropout_", &ProfiledType::dropout_);
  m.impl("elu", TORCH_FN(ProfiledType::elu));
  m.impl_UNBOXED("elu_", &ProfiledType::elu_);
  m.impl_UNBOXED("elu_backward.grad_input", &ProfiledType::elu_backward_out_grad_input);
  m.impl_UNBOXED("embedding_renorm_", &ProfiledType::embedding_renorm_);
  m.impl("equal", TORCH_FN(ProfiledType::equal));
  m.impl_UNBOXED("erf.out", &ProfiledType::erf_out_out);
  m.impl("erfc", TORCH_FN(ProfiledType::erfc));
  m.impl_UNBOXED("erfc_", &ProfiledType::erfc_);
  m.impl("expm1", TORCH_FN(ProfiledType::expm1));
  m.impl_UNBOXED("expm1_", &ProfiledType::expm1_);
  m.impl_UNBOXED("exponential_", &ProfiledType::exponential_);
  m.impl("fake_quantize_per_channel_affine", TORCH_FN(ProfiledType::fake_quantize_per_channel_affine));
  m.impl("fbgemm_linear_fp16_weight_fp32_activation", TORCH_FN(ProfiledType::fbgemm_linear_fp16_weight_fp32_activation));
  m.impl("fbgemm_linear_int8_weight_fp32_activation", TORCH_FN(ProfiledType::fbgemm_linear_int8_weight_fp32_activation));
  m.impl("fbgemm_linear_quantize_weight", TORCH_FN(ProfiledType::fbgemm_linear_quantize_weight));
  m.impl("floor", TORCH_FN(ProfiledType::floor));
  m.impl_UNBOXED("floor_", &ProfiledType::floor_);
  m.impl_UNBOXED("fmod.Scalar_out", &ProfiledType::fmod_out_Scalar_out);
  m.impl_UNBOXED("fmod.Tensor_out", &ProfiledType::fmod_out_Tensor_out);
  m.impl_UNBOXED("frac.out", &ProfiledType::frac_out_out);
  m.impl_UNBOXED("fractional_max_pool2d.output", &ProfiledType::fractional_max_pool2d_out_output);
  m.impl_UNBOXED("frobenius_norm.out", &ProfiledType::frobenius_norm_out_out);
  m.impl_UNBOXED("full_like", &ProfiledType::full_like);
  m.impl_UNBOXED("group_norm", &ProfiledType::group_norm);
  m.impl_UNBOXED("hamming_window", &ProfiledType::hamming_window);
  m.impl_UNBOXED("hamming_window.periodic", &ProfiledType::hamming_window_periodic);
  m.impl_UNBOXED("hamming_window.periodic_alpha", &ProfiledType::hamming_window_periodic_alpha);
  m.impl_UNBOXED("hamming_window.periodic_alpha_beta", &ProfiledType::hamming_window_periodic_alpha_beta);
  m.impl("hardshrink_backward", TORCH_FN(ProfiledType::hardshrink_backward));
  m.impl_UNBOXED("hardtanh.out", &ProfiledType::hardtanh_out_out);
  m.impl_UNBOXED("im2col.out", &ProfiledType::im2col_out_out);
  m.impl_UNBOXED("index.Tensor", &ProfiledType::index_Tensor);
  m.impl_UNBOXED("index_put", &ProfiledType::index_put);
  m.impl_UNBOXED("index_put_", &ProfiledType::index_put_);
  m.impl("index_select", TORCH_FN(ProfiledType::index_select));
  m.impl_UNBOXED("index_select.dimname", &ProfiledType::index_select_dimname);
  m.impl("is_coalesced", TORCH_FN(ProfiledType::is_coalesced));
  m.impl("is_floating_point", TORCH_FN(ProfiledType::is_floating_point));
  m.impl("is_vulkan_available", TORCH_FN(ProfiledType::is_vulkan_available));
  m.impl("item", TORCH_FN(ProfiledType::item));
  m.impl("l1_loss", TORCH_FN(ProfiledType::l1_loss));
  m.impl_UNBOXED("l1_loss_backward.grad_input", &ProfiledType::l1_loss_backward_out_grad_input);
  m.impl_UNBOXED("linspace.out", &ProfiledType::linspace_out_out);
  m.impl_UNBOXED("log2.out", &ProfiledType::log2_out_out);
  m.impl_UNBOXED("log_normal_", &ProfiledType::log_normal_);
  m.impl_UNBOXED("log.out", &ProfiledType::log_out_out);
  m.impl("log_sigmoid_backward", TORCH_FN(ProfiledType::log_sigmoid_backward));
  m.impl_UNBOXED("log_sigmoid_forward.output", &ProfiledType::log_sigmoid_forward_out_output);
  m.impl_UNBOXED("logaddexp2.out", &ProfiledType::logaddexp2_out_out);
  m.impl_UNBOXED("logaddexp.out", &ProfiledType::logaddexp_out_out);
  m.impl_UNBOXED("logcumsumexp", &ProfiledType::logcumsumexp);
  m.impl_UNBOXED("logcumsumexp.dimname", &ProfiledType::logcumsumexp_dimname);
  m.impl("logical_or", TORCH_FN(ProfiledType::logical_or));
  m.impl_UNBOXED("logical_or_", &ProfiledType::logical_or_);
  m.impl("logical_xor", TORCH_FN(ProfiledType::logical_xor));
  m.impl_UNBOXED("logical_xor_", &ProfiledType::logical_xor_);
  m.impl_UNBOXED("logspace", &ProfiledType::logspace);
  m.impl("logsumexp", TORCH_FN(ProfiledType::logsumexp));
  m.impl_UNBOXED("logsumexp.names", &ProfiledType::logsumexp_names);
  m.impl_UNBOXED("lstsq.X", &ProfiledType::lstsq_out_X);
  m.impl("matmul", TORCH_FN(ProfiledType::matmul));
  m.impl("max.dim", TORCH_FN(ProfiledType::max_dim));
  m.impl_UNBOXED("max.names_dim", &ProfiledType::max_names_dim);
  m.impl("max.other", TORCH_FN(ProfiledType::max_other));
  m.impl("max", TORCH_FN(ProfiledType::max));
  m.impl("max_pool1d_with_indices", TORCH_FN(ProfiledType::max_pool1d_with_indices));
  m.impl_UNBOXED("max_pool2d_with_indices.out", &ProfiledType::max_pool2d_with_indices_out_out);
  m.impl("max_unpool2d_backward", TORCH_FN(ProfiledType::max_unpool2d_backward));
  m.impl("max_unpool3d", TORCH_FN(ProfiledType::max_unpool3d));
  m.impl_UNBOXED("max_unpool3d_backward.grad_input", &ProfiledType::max_unpool3d_backward_out_grad_input);
  m.impl_UNBOXED("mean.out", &ProfiledType::mean_out_out);
  m.impl_UNBOXED("mean.names_out", &ProfiledType::mean_out_names_out);
  m.impl_UNBOXED("median.dim_values", &ProfiledType::median_out_dim_values);
  m.impl_UNBOXED("median.names_dim_values", &ProfiledType::median_out_names_dim_values);
  m.impl("meshgrid", TORCH_FN(ProfiledType::meshgrid));
  m.impl_UNBOXED("miopen_batch_norm", &ProfiledType::miopen_batch_norm);
  m.impl("miopen_convolution_transpose_backward", TORCH_FN(ProfiledType::miopen_convolution_transpose_backward));
  m.impl("miopen_convolution_transpose_backward_input", TORCH_FN(ProfiledType::miopen_convolution_transpose_backward_input));
  m.impl("miopen_depthwise_convolution_backward_weight", TORCH_FN(ProfiledType::miopen_depthwise_convolution_backward_weight));
  m.impl("mkldnn_convolution_backward", TORCH_FN(ProfiledType::mkldnn_convolution_backward));
  m.impl("mkldnn_convolution_backward_input", TORCH_FN(ProfiledType::mkldnn_convolution_backward_input));
  m.impl("mode", TORCH_FN(ProfiledType::mode));
  m.impl_UNBOXED("mode.dimname", &ProfiledType::mode_dimname);
  m.impl_UNBOXED("multi_margin_loss.out", &ProfiledType::multi_margin_loss_out_out);
  m.impl("multilabel_margin_loss_forward", TORCH_FN(ProfiledType::multilabel_margin_loss_forward));
  m.impl_UNBOXED("mv.out", &ProfiledType::mv_out_out);
  m.impl_UNBOXED("native_batch_norm_backward", &ProfiledType::native_batch_norm_backward);
  m.impl("native_norm", TORCH_FN(ProfiledType::native_norm));
  m.impl("ne.Scalar", TORCH_FN(ProfiledType::ne_Scalar));
  m.impl("ne.Tensor", TORCH_FN(ProfiledType::ne_Tensor));
  m.impl_UNBOXED("ne_.Scalar", &ProfiledType::ne__Scalar);
  m.impl_UNBOXED("ne_.Tensor", &ProfiledType::ne__Tensor);
  m.impl_UNBOXED("nll_loss2d_backward", &ProfiledType::nll_loss2d_backward);
  m.impl_UNBOXED("nll_loss2d_forward.output", &ProfiledType::nll_loss2d_forward_out_output);
  m.impl_UNBOXED("nll_loss_backward", &ProfiledType::nll_loss_backward);
  m.impl_UNBOXED("nll_loss_forward.output", &ProfiledType::nll_loss_forward_out_output);
  m.impl_UNBOXED("ones.out", &ProfiledType::ones_out_out);
  m.impl("ormqr", TORCH_FN(ProfiledType::ormqr));
  m.impl("pairwise_distance", TORCH_FN(ProfiledType::pairwise_distance));
  m.impl("pinverse", TORCH_FN(ProfiledType::pinverse));
  m.impl_UNBOXED("polygamma.out", &ProfiledType::polygamma_out_out);
  m.impl_UNBOXED("pow.Tensor_Scalar_out", &ProfiledType::pow_out_Tensor_Scalar_out);
  m.impl_UNBOXED("pow.Tensor_Tensor_out", &ProfiledType::pow_out_Tensor_Tensor_out);
  m.impl_UNBOXED("pow.Scalar_out", &ProfiledType::pow_out_Scalar_out);
  m.impl_UNBOXED("prod.int_out", &ProfiledType::prod_out_int_out);
  m.impl_UNBOXED("prod.Dimname_out", &ProfiledType::prod_out_Dimname_out);
  m.impl("q_per_channel_axis", TORCH_FN(ProfiledType::q_per_channel_axis));
  m.impl("q_per_channel_zero_points", TORCH_FN(ProfiledType::q_per_channel_zero_points));
  m.impl_UNBOXED("qr.Q", &ProfiledType::qr_out_Q);
  m.impl("quantized_gru_cell", TORCH_FN(ProfiledType::quantized_gru_cell));
  m.impl("quantized_rnn_relu_cell", TORCH_FN(ProfiledType::quantized_rnn_relu_cell));
  m.impl_UNBOXED("rand.names", &ProfiledType::rand_names);
  m.impl_UNBOXED("rand.generator_with_names", &ProfiledType::rand_generator_with_names);
  m.impl_UNBOXED("rand", &ProfiledType::rand);
  m.impl_UNBOXED("rand.generator", &ProfiledType::rand_generator);
  m.impl_UNBOXED("randint.out", &ProfiledType::randint_out_out);
  m.impl_UNBOXED("randint.generator_out", &ProfiledType::randint_out_generator_out);
  m.impl_UNBOXED("randint.low_out", &ProfiledType::randint_out_low_out);
  m.impl_UNBOXED("randint.low_generator_out", &ProfiledType::randint_out_low_generator_out);
  m.impl_UNBOXED("randn", &ProfiledType::randn);
  m.impl_UNBOXED("randn.generator", &ProfiledType::randn_generator);
  m.impl_UNBOXED("randn.names", &ProfiledType::randn_names);
  m.impl_UNBOXED("randn.generator_with_names", &ProfiledType::randn_generator_with_names);
  m.impl_UNBOXED("random_.from", &ProfiledType::random__from);
  m.impl_UNBOXED("random_.to", &ProfiledType::random__to);
  m.impl_UNBOXED("random_", &ProfiledType::random_);
  m.impl_UNBOXED("randperm.out", &ProfiledType::randperm_out_out);
  m.impl_UNBOXED("randperm.generator_out", &ProfiledType::randperm_out_generator_out);
  m.impl("reflection_pad1d_backward", TORCH_FN(ProfiledType::reflection_pad1d_backward));
  m.impl("reflection_pad2d", TORCH_FN(ProfiledType::reflection_pad2d));
  m.impl_UNBOXED("reflection_pad2d_backward.grad_input", &ProfiledType::reflection_pad2d_backward_out_grad_input);
  m.impl("remainder.Scalar", TORCH_FN(ProfiledType::remainder_Scalar));
  m.impl("remainder.Tensor", TORCH_FN(ProfiledType::remainder_Tensor));
  m.impl_UNBOXED("remainder_.Scalar", &ProfiledType::remainder__Scalar);
  m.impl_UNBOXED("remainder_.Tensor", &ProfiledType::remainder__Tensor);
  m.impl("repeat", TORCH_FN(ProfiledType::repeat));
  m.impl("replication_pad1d", TORCH_FN(ProfiledType::replication_pad1d));
  m.impl_UNBOXED("replication_pad1d_backward.grad_input", &ProfiledType::replication_pad1d_backward_out_grad_input);
  m.impl_UNBOXED("replication_pad2d.out", &ProfiledType::replication_pad2d_out_out);
  m.impl("reshape_as", TORCH_FN(ProfiledType::reshape_as));
  m.impl_UNBOXED("result_type.Tensor", &ProfiledType::result_type_Tensor);
  m.impl_UNBOXED("result_type.Scalar", &ProfiledType::result_type_Scalar);
  m.impl_UNBOXED("result_type.Scalar_Tensor", &ProfiledType::result_type_Scalar_Tensor);
  m.impl_UNBOXED("result_type.Scalar_Scalar", &ProfiledType::result_type_Scalar_Scalar);
  m.impl("rnn_tanh.input", TORCH_FN(ProfiledType::rnn_tanh_input));
  m.impl("rnn_tanh.data", TORCH_FN(ProfiledType::rnn_tanh_data));
  m.impl("roll", TORCH_FN(ProfiledType::roll));
  m.impl("rot90", TORCH_FN(ProfiledType::rot90));
  m.impl_UNBOXED("round.out", &ProfiledType::round_out_out);
  m.impl("rrelu_with_noise_backward", TORCH_FN(ProfiledType::rrelu_with_noise_backward));
  m.impl_UNBOXED("rsqrt.out", &ProfiledType::rsqrt_out_out);
  m.impl_UNBOXED("searchsorted.Tensor_out", &ProfiledType::searchsorted_out_Tensor_out);
  m.impl("selu", TORCH_FN(ProfiledType::selu));
  m.impl_UNBOXED("selu_", &ProfiledType::selu_);
  m.impl_UNBOXED("slow_conv3d.out", &ProfiledType::slow_conv3d_out_out);
  m.impl_UNBOXED("slow_conv_transpose2d.out", &ProfiledType::slow_conv_transpose2d_out_out);
  m.impl_UNBOXED("smooth_l1_loss.out", &ProfiledType::smooth_l1_loss_out_out);
  m.impl("soft_margin_loss", TORCH_FN(ProfiledType::soft_margin_loss));
  m.impl_UNBOXED("soft_margin_loss_backward.grad_input", &ProfiledType::soft_margin_loss_backward_out_grad_input);
  m.impl("softplus", TORCH_FN(ProfiledType::softplus));
  m.impl_UNBOXED("softplus_backward.grad_input", &ProfiledType::softplus_backward_out_grad_input);
  m.impl("sort", TORCH_FN(ProfiledType::sort));
  m.impl_UNBOXED("sort.dimname", &ProfiledType::sort_dimname);
  m.impl("split.Tensor", TORCH_FN(ProfiledType::split_Tensor));
  m.impl_UNBOXED("sspaddmm.out", &ProfiledType::sspaddmm_out_out);
  m.impl("std", TORCH_FN(ProfiledType::std));
  m.impl("std.dim", TORCH_FN(ProfiledType::std_dim));
  m.impl_UNBOXED("std.names_dim", &ProfiledType::std_names_dim);
  m.impl("stride.int", TORCH_FN(ProfiledType::stride_int));
  m.impl_UNBOXED("stride.Dimname", &ProfiledType::stride_Dimname);
  m.impl_UNBOXED("sum", &ProfiledType::sum);
  m.impl_UNBOXED("sum.dim_IntList", &ProfiledType::sum_dim_IntList);
  m.impl_UNBOXED("sum.dim_DimnameList", &ProfiledType::sum_dim_DimnameList);
  m.impl("sum_to_size", TORCH_FN(ProfiledType::sum_to_size));
  m.impl("t", TORCH_FN(ProfiledType::t));
  m.impl_UNBOXED("t_", &ProfiledType::t_);
  m.impl("take", TORCH_FN(ProfiledType::take));
  m.impl_UNBOXED("tanh.out", &ProfiledType::tanh_out_out);
  m.impl_UNBOXED("thnn_conv_depthwise2d", &ProfiledType::thnn_conv_depthwise2d);
  m.impl_UNBOXED("thnn_conv_depthwise2d_backward.grad_input", &ProfiledType::thnn_conv_depthwise2d_backward_out_grad_input);
  m.impl("to_mkldnn", TORCH_FN(ProfiledType::to_mkldnn));
  m.impl("to_sparse.sparse_dim", TORCH_FN(ProfiledType::to_sparse_sparse_dim));
  m.impl("to_sparse", TORCH_FN(ProfiledType::to_sparse));
  m.impl("topk", TORCH_FN(ProfiledType::topk));
  m.impl("transpose.int", TORCH_FN(ProfiledType::transpose_int));
  m.impl_UNBOXED("transpose.Dimname", &ProfiledType::transpose_Dimname);
  m.impl_UNBOXED("transpose_", &ProfiledType::transpose_);
  m.impl("trapz.x", TORCH_FN(ProfiledType::trapz_x));
  m.impl("trapz.dx", TORCH_FN(ProfiledType::trapz_dx));
  m.impl_UNBOXED("triu.out", &ProfiledType::triu_out_out);
  m.impl_UNBOXED("true_divide.out", &ProfiledType::true_divide_out_out);
  m.impl("trunc", TORCH_FN(ProfiledType::trunc));
  m.impl_UNBOXED("trunc_", &ProfiledType::trunc_);
  m.impl_UNBOXED("upsample_bicubic2d.out", &ProfiledType::upsample_bicubic2d_out_out);
  m.impl("upsample_linear1d_backward", TORCH_FN(ProfiledType::upsample_linear1d_backward));
  m.impl("upsample_nearest2d_backward", TORCH_FN(ProfiledType::upsample_nearest2d_backward));
  m.impl("upsample_nearest3d", TORCH_FN(ProfiledType::upsample_nearest3d));
  m.impl_UNBOXED("upsample_nearest3d_backward.grad_input", &ProfiledType::upsample_nearest3d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_trilinear3d.out", &ProfiledType::upsample_trilinear3d_out_out);
  m.impl("values", TORCH_FN(ProfiledType::values));
  m.impl("var_mean", TORCH_FN(ProfiledType::var_mean));
  m.impl("var_mean.dim", TORCH_FN(ProfiledType::var_mean_dim));
  m.impl_UNBOXED("var_mean.names_dim", &ProfiledType::var_mean_names_dim);;
}

}  // namespace

} // namespace torch
