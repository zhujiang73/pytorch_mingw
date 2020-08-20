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
Tensor __and___Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__and__", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("__and__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __and___Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__and__", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__and__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __iand___Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__iand__", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("__iand__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __iand___Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__iand__", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__iand__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_adaptive_avg_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_adaptive_avg_pool2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
void _amp_non_finite_check_and_unscale_(Tensor & self, Tensor & found_inf, const Tensor & inv_scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_amp_non_finite_check_and_unscale_", "")
      .typed<void (Tensor &, Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_amp_non_finite_check_and_unscale_", std::vector<c10::IValue>({self, found_inf, inv_scale}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, found_inf, inv_scale);
}
Tensor _cast_Long(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Long", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Long", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cat(TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cat", "")
      .typed<Tensor (TensorList, int64_t)>();
  RECORD_FUNCTION("_cat", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t>(op, c10::DispatchKey::Profiler, tensors, dim);
}
Tensor _cdist_forward(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cdist_forward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>)>();
  RECORD_FUNCTION("_cdist_forward", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, x1, x2, p, compute_mode);
}
std::tuple<double,int64_t> _choose_qparams_per_tensor(const Tensor & self, bool reduce_range) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_choose_qparams_per_tensor", "")
      .typed<std::tuple<double,int64_t> (const Tensor &, bool)>();
  RECORD_FUNCTION("_choose_qparams_per_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<double,int64_t>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, reduce_range);
}
std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_convolution_double_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>)>();
  RECORD_FUNCTION("_convolution_double_backward", std::vector<c10::IValue>({ggI, ggW, ggb, gO, weight, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, bool, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Profiler, ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
}
void _cufft_clear_plan_cache(int64_t device_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_clear_plan_cache", "")
      .typed<void (int64_t)>();
  RECORD_FUNCTION("_cufft_clear_plan_cache", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, int64_t>(op, c10::DispatchKey::Profiler, device_index);
}
int64_t _cufft_get_plan_cache_max_size(int64_t device_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_get_plan_cache_max_size", "")
      .typed<int64_t (int64_t)>();
  RECORD_FUNCTION("_cufft_get_plan_cache_max_size", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, int64_t>(op, c10::DispatchKey::Profiler, device_index);
}
Tensor _cumprod(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumprod", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("_cumprod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
int64_t _debug_has_internal_overlap(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_debug_has_internal_overlap", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_debug_has_internal_overlap", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
int64_t _dimI(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dimI", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_dimI", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_empty_affine_quantized", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("_empty_affine_quantized", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &, double, int64_t, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, options, scale, zero_point, memory_format);
}
Tensor _fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_fft_with_size", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool, bool, IntArrayRef, bool, bool, IntArrayRef)>();
  RECORD_FUNCTION("_fft_with_size", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool, bool, IntArrayRef, bool, bool, IntArrayRef>(op, c10::DispatchKey::Profiler, self, signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
}
bool _has_compatible_shallow_copy_type(const Tensor & self, const Tensor & from) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_has_compatible_shallow_copy_type", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_has_compatible_shallow_copy_type", std::vector<c10::IValue>({self, from}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, from);
}
Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_log_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, half_to_float);
}
std::tuple<Tensor,Tensor,Tensor> _lu_with_info(const Tensor & self, bool pivot, bool check_errors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_lu_with_info", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_lu_with_info", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, pivot, check_errors);
}
Tensor _multinomial_alias_draw(const Tensor & J, const Tensor & q, int64_t num_samples, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_multinomial_alias_draw", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, c10::optional<Generator>)>();
  RECORD_FUNCTION("_multinomial_alias_draw", std::vector<c10::IValue>({J, q}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, J, q, num_samples, generator);
}
std::tuple<Tensor,Tensor,Tensor> _nnpack_spatial_convolution_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, std::array<bool,3>)>();
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward", std::vector<c10::IValue>({input, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, std::array<bool,3>>(op, c10::DispatchKey::Profiler, input, grad_output, weight, padding, output_mask);
}
Tensor _nnpack_spatial_convolution_backward_input(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution_backward_input", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward_input", std::vector<c10::IValue>({input, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, input, grad_output, weight, padding);
}
int64_t _nnz(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnz", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_nnz", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> _pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pack_padded_sequence", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_pack_padded_sequence", std::vector<c10::IValue>({input, lengths}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, input, lengths, batch_first);
}
std::tuple<Tensor,Tensor> _pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pad_packed_sequence", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, Scalar, int64_t)>();
  RECORD_FUNCTION("_pad_packed_sequence", std::vector<c10::IValue>({data, batch_sizes, padding_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, Scalar, int64_t>(op, c10::DispatchKey::Profiler, data, batch_sizes, batch_first, padding_value, total_length);
}
std::tuple<Tensor,Tensor> _qr_helper(const Tensor & self, bool some) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_qr_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("_qr_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, some);
}
Tensor _reshape_from_tensor(const Tensor & self, const Tensor & shape) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_reshape_from_tensor", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_reshape_from_tensor", std::vector<c10::IValue>({self, shape}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, shape);
}
Tensor & _sobol_engine_ff_(Tensor & self, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sobol_engine_ff_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("_sobol_engine_ff_", std::vector<c10::IValue>({self, sobolstate}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, n, sobolstate, dimension, num_generated);
}
Tensor & _sobol_engine_initialize_state_(Tensor & self, int64_t dimension) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sobol_engine_initialize_state_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("_sobol_engine_initialize_state_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dimension);
}
Tensor _sparse_log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_log_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_sparse_log_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output, dim, self);
}
Tensor _sparse_mm(const Tensor & sparse, const Tensor & dense) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_mm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_sparse_mm", std::vector<c10::IValue>({sparse, dense}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, sparse, dense);
}
Tensor _test_serialization_subcmul(const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_test_serialization_subcmul", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("_test_serialization_subcmul", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
bool _use_cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_use_cudnn_ctc_loss", "")
      .typed<bool (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("_use_cudnn_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank);
}
Tensor _weight_norm(const Tensor & v, const Tensor & g, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm", std::vector<c10::IValue>({v, g}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, v, g, dim);
}
Tensor & absolute_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::absolute", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("absolute_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor acos(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acos", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("acos", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & acos_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acos_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("acos_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor & adaptive_avg_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_avg_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self);
}
Tensor add_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("add", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor add_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add", "Scalar")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("add", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor & add__Tensor(Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("add_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor & add__Scalar(Tensor & self, Scalar other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::add_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("add_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor & addbmm_out_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addbmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addbmm_out", std::vector<c10::IValue>({out, self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, batch1, batch2, beta, alpha);
}
Tensor affine_grid_generator_backward(const Tensor & grad, IntArrayRef size, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::affine_grid_generator_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("affine_grid_generator_backward", std::vector<c10::IValue>({grad}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, grad, size, align_corners);
}
Tensor alias(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::alias", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("alias", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & all_out_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::all", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("all_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor & all_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::all", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("all_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
bool allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::allclose", "")
      .typed<bool (const Tensor &, const Tensor &, double, double, bool)>();
  RECORD_FUNCTION("allclose", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &, double, double, bool>(op, c10::DispatchKey::Profiler, self, other, rtol, atol, equal_nan);
}
Tensor & any_out_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::any", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("any_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor & any_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::any", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("any_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor arange(Scalar end, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "")
      .typed<Tensor (Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("arange", std::vector<c10::IValue>({end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, end, options);
}
Tensor arange_start(Scalar start, Scalar end, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "start")
      .typed<Tensor (Scalar, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("arange", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, options);
}
Tensor arange_start_step(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::arange", "start_step")
      .typed<Tensor (Scalar, Scalar, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("arange", std::vector<c10::IValue>({start, end, step}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, step, options);
}
Tensor asin(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asin", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("asin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & asin_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asin_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("asin_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & batch_norm_elemt_out_out(Tensor & out, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_elemt", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_elemt_out", std::vector<c10::IValue>({out, input, weight, bias, mean, invstd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, out, input, weight, bias, mean, invstd, eps);
}
std::tuple<Tensor,Tensor> batch_norm_gather_stats(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, int64_t count) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_gather_stats", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  RECORD_FUNCTION("batch_norm_gather_stats", std::vector<c10::IValue>({input, mean, invstd, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Profiler, input, mean, invstd, running_mean, running_var, momentum, eps, count);
}
std::tuple<Tensor,Tensor> batch_norm_stats(const Tensor & input, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_stats", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_stats", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, eps);
}
Tensor bernoulli(const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("bernoulli", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, generator);
}
Tensor bernoulli_p(const Tensor & self, double p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli", "p")
      .typed<Tensor (const Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("bernoulli", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
Tensor & bernoulli__Tensor(Tensor & self, const Tensor & p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("bernoulli_", std::vector<c10::IValue>({self, p}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
Tensor & bernoulli__float(Tensor & self, double p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bernoulli_", "float")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("bernoulli_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
Tensor binary_cross_entropy_with_logits_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_with_logits_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_with_logits_backward", std::vector<c10::IValue>({grad_output, self, target, weight, pos_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, pos_weight, reduction);
}
Tensor cat(TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cat", "")
      .typed<Tensor (TensorList, int64_t)>();
  RECORD_FUNCTION("cat", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t>(op, c10::DispatchKey::Profiler, tensors, dim);
}
Tensor cat_names(TensorList tensors, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cat", "names")
      .typed<Tensor (TensorList, Dimname)>();
  RECORD_FUNCTION("cat", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, Dimname>(op, c10::DispatchKey::Profiler, tensors, dim);
}
Tensor & cauchy_(Tensor & self, double median, double sigma, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cauchy_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("cauchy_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, median, sigma, generator);
}
Tensor & ceil_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ceil", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ceil_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::chunk", "")
      .typed<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("chunk", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, chunks, dim);
}
Tensor col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size, kernel_size, dilation, padding, stride);
}
Tensor & col2im_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, kernel_size, dilation, padding, stride);
}
Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv1d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("conv1d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, groups);
}
std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_tbc_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("conv_tbc_backward", std::vector<c10::IValue>({self, input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, input, weight, bias, pad);
}
Tensor cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosine_embedding_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("cosine_embedding_loss", std::vector<c10::IValue>({input1, input2, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, input1, input2, target, margin, reduction);
}
Tensor cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_affine_grid_generator", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("cudnn_affine_grid_generator", std::vector<c10::IValue>({theta}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, theta, N, C, H, W);
}
Tensor cudnn_convolution_deprecated(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution", "deprecated")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution", std::vector<c10::IValue>({self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cudnn_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_transpose_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cumprod(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumprod", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumprod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor cumprod_dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumprod", "dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumprod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor dequantize_self(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dequantize", "self")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("dequantize", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::vector<Tensor> dequantize_tensors(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dequantize", "tensors")
      .typed<std::vector<Tensor> (TensorList)>();
  RECORD_FUNCTION("dequantize", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
Tensor det(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::det", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("det", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor detach(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::detach", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("detach", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & detach_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::detach_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & div_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::div", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("div_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor dot(const Tensor & self, const Tensor & tensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::dot", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("dot", std::vector<c10::IValue>({self, tensor}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, tensor);
}
Tensor einsum(std::string equation, TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::einsum", "")
      .typed<Tensor (std::string, TensorList)>();
  RECORD_FUNCTION("einsum", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, std::string, TensorList>(op, c10::DispatchKey::Profiler, equation, tensors);
}
Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu_backward", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("elu_backward", std::vector<c10::IValue>({grad_output, alpha, scale, input_scale, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, alpha, scale, input_scale, output);
}
Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, bool, bool)>();
  RECORD_FUNCTION("embedding", std::vector<c10::IValue>({weight, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight, indices, padding_idx, scale_grad_by_freq, sparse);
}
Tensor embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_sparse_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("embedding_sparse_backward", std::vector<c10::IValue>({grad, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor empty_meta(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_meta", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty_meta", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, options, memory_format);
}
Tensor erf(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erf", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("erf", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & erf_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erf_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("erf_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & exp_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exp", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("exp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor expand_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expand_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("expand_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & eye_out_out(Tensor & out, int64_t n) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eye", "out")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("eye_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, n);
}
Tensor & eye_out_m_out(Tensor & out, int64_t n, int64_t m) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eye", "m_out")
      .typed<Tensor & (Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("eye_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, out, n, m);
}
Tensor fake_quantize_per_channel_affine_backward(const Tensor & grad, const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_channel_affine_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_channel_affine_backward", std::vector<c10::IValue>({grad, self, scale, zero_point}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad, self, scale, zero_point, axis, quant_min, quant_max);
}
Tensor fbgemm_linear_fp16_weight(const Tensor & input, const Tensor & packed_weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_fp16_weight", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_fp16_weight", std::vector<c10::IValue>({input, packed_weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, packed_weight, bias);
}
Tensor fbgemm_linear_int8_weight(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_linear_int8_weight", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("fbgemm_linear_int8_weight", std::vector<c10::IValue>({input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}
Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_pack_gemm_matrix_fp16", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("fbgemm_pack_gemm_matrix_fp16", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, input);
}
Tensor feature_alpha_dropout(const Tensor & input, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::feature_alpha_dropout", "")
      .typed<Tensor (const Tensor &, double, bool)>();
  RECORD_FUNCTION("feature_alpha_dropout", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, p, train);
}
Tensor & feature_alpha_dropout_(Tensor & self, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::feature_alpha_dropout_", "")
      .typed<Tensor & (Tensor &, double, bool)>();
  RECORD_FUNCTION("feature_alpha_dropout_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, bool>(op, c10::DispatchKey::Profiler, self, p, train);
}
Tensor flip(const Tensor & self, IntArrayRef dims) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flip", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("flip", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, dims);
}
Tensor fmod_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("fmod", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor fmod_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fmod", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & fmod__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("fmod_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & fmod__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fmod_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fmod_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor frac(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frac", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("frac", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & frac_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frac_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("frac_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d", std::vector<c10::IValue>({self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, self, kernel_size, output_size, random_samples);
}
Tensor & fractional_max_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, output_size, indices);
}
std::tuple<Tensor &,Tensor &> fractional_max_pool3d_out_output(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d_out", std::vector<c10::IValue>({output, indices, self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, output, indices, self, kernel_size, output_size, random_samples);
}
Tensor frobenius_norm(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frobenius_norm", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("frobenius_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor frobenius_norm_dim(const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::frobenius_norm", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("frobenius_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::from_file", "")
      .typed<Tensor (std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &)>();
  RECORD_FUNCTION("from_file", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, std::string, c10::optional<bool>, c10::optional<int64_t>, const TensorOptions &>(op, c10::DispatchKey::Profiler, filename, shared, size, options);
}
std::tuple<Tensor &,Tensor &> geqrf_out_a(Tensor & a, Tensor & tau, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geqrf", "a")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &)>();
  RECORD_FUNCTION("geqrf_out", std::vector<c10::IValue>({a, tau, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, a, tau, self);
}
Tensor & ger_out_out(Tensor & out, const Tensor & self, const Tensor & vec2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ger", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ger_out", std::vector<c10::IValue>({out, self, vec2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, vec2);
}
Tensor & glu_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
}
Tensor & hardsigmoid_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardsigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & hardswish_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardswish", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardswish_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh", std::vector<c10::IValue>({self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, min_val, max_val);
}
Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh_", "")
      .typed<Tensor & (Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_", std::vector<c10::IValue>({self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, min_val, max_val);
}
Tensor & hardtanh_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, min_val, max_val);
}
Tensor im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, kernel_size, dilation, padding, stride);
}
Tensor & im2col_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, input_size, kernel_size, dilation, padding, stride);
}
Tensor instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::instance_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>();
  RECORD_FUNCTION("instance_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}
Tensor int_repr(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::int_repr", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("int_repr", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_distributed(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_distributed", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_distributed", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor isinf(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::isinf", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("isinf", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor isnan(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::isnan", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("isnan", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor &,Tensor &> kthvalue_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kthvalue", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("kthvalue_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, values, indices, self, k, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> kthvalue_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kthvalue", "dimname_out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, Dimname, bool)>();
  RECORD_FUNCTION("kthvalue_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, Dimname, bool>(op, c10::DispatchKey::Profiler, values, indices, self, k, dim, keepdim);
}
Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor & lgamma_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lgamma", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lgamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linspace", "")
      .typed<Tensor (Scalar, Scalar, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("linspace", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, steps, options);
}
Tensor log(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log1p_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log1p", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log1p_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor log2(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log2", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log2", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log2_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log2_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("log2_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("log_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_forward", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor log_softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_softmax", "int")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor log_softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_softmax", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("log_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor logaddexp(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logaddexp", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logaddexp", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor logaddexp2(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logaddexp2", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logaddexp2", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & logical_and_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_and", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_and_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & logical_not_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_not", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_not_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::tuple<Tensor,Tensor> lstsq(const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstsq", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lstsq", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, A);
}
Tensor & lu_solve_out_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lu_solve", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lu_solve_out", std::vector<c10::IValue>({out, self, LU_data, LU_pivots}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, LU_data, LU_pivots);
}
Tensor masked_fill_Scalar(const Tensor & self, const Tensor & mask, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_fill", "Scalar")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("masked_fill", std::vector<c10::IValue>({self, mask, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, mask, value);
}
Tensor masked_fill_Tensor(const Tensor & self, const Tensor & mask, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_fill", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_fill", std::vector<c10::IValue>({self, mask, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask, value);
}
Tensor & masked_fill__Scalar(Tensor & self, const Tensor & mask, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_fill_", "Scalar")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("masked_fill_", std::vector<c10::IValue>({self, mask, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, mask, value);
}
Tensor & masked_fill__Tensor(Tensor & self, const Tensor & mask, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_fill_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_fill_", std::vector<c10::IValue>({self, mask, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask, value);
}
Tensor masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_scatter", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_scatter", std::vector<c10::IValue>({self, mask, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask, source);
}
Tensor & masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_scatter_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_scatter_", std::vector<c10::IValue>({self, mask, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask, source);
}
Tensor max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool2d_with_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & max_pool2d_with_indices_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool2d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool3d_with_indices_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, indices, output_size, stride, padding);
}
Tensor max_values(const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_values", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor max_values_names(const Tensor & self, DimnameList dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_values", "names")
      .typed<Tensor (const Tensor &, DimnameList, bool)>();
  RECORD_FUNCTION("max_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "")
      .typed<Tensor (const Tensor &, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dtype);
}
Tensor mean_dim(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, keepdim, dtype);
}
Tensor mean_names_dim(const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mean", "names_dim")
      .typed<Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, keepdim, dtype);
}
std::tuple<Tensor,Tensor> median_dim(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("median", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> median_names_dim(const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("median", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor median(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::median", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("median", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("miopen_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}
Tensor miopen_depthwise_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_depthwise_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> miopen_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_rnn", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("miopen_rnn", std::vector<c10::IValue>({input, hx, cx, dropout_state}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
Tensor mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("mkldnn_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & mm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, mat2);
}
Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss", std::vector<c10::IValue>({self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, p, margin, weight, reduction);
}
Tensor & multi_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, p, margin, weight, reduction);
}
Tensor mv(const Tensor & self, const Tensor & vec) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mv", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mv", std::vector<c10::IValue>({self, vec}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, vec);
}
std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_layer_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double)>();
  RECORD_FUNCTION("native_layer_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, double>(op, c10::DispatchKey::Profiler, input, weight, bias, M, N, eps);
}
std::tuple<Tensor,Tensor> nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d_forward", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction, ignore_index);
}
std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_forward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss_forward", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction, ignore_index);
}
Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm_except_dim", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("norm_except_dim", std::vector<c10::IValue>({v}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, v, pow, dim);
}
Tensor ones_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("ones", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, names, options);
}
Tensor ones(IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("ones", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, options);
}
Tensor pin_memory(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pin_memory", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("pin_memory", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor polygamma(int64_t n, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::polygamma", "")
      .typed<Tensor (int64_t, const Tensor &)>();
  RECORD_FUNCTION("polygamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, n, self);
}
Tensor & polygamma_(Tensor & self, int64_t n) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::polygamma_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("polygamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, n);
}
Tensor pow_Tensor_Scalar(const Tensor & self, Scalar exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Tensor_Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("pow", std::vector<c10::IValue>({self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, exponent);
}
Tensor pow_Tensor_Tensor(const Tensor & self, const Tensor & exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Tensor_Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("pow", std::vector<c10::IValue>({self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, exponent);
}
Tensor pow_Scalar(Scalar self, const Tensor & exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow", "Scalar")
      .typed<Tensor (Scalar, const Tensor &)>();
  RECORD_FUNCTION("pow", std::vector<c10::IValue>({self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, self, exponent);
}
Tensor & pow__Scalar(Tensor & self, Scalar exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("pow_", std::vector<c10::IValue>({self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, exponent);
}
Tensor & pow__Tensor(Tensor & self, const Tensor & exponent) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pow_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("pow_", std::vector<c10::IValue>({self, exponent}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, exponent);
}
Tensor prelu(const Tensor & self, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prelu", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("prelu", std::vector<c10::IValue>({self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, weight);
}
Tensor prod(const Tensor & self, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "")
      .typed<Tensor (const Tensor &, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("prod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dtype);
}
Tensor prod_dim_int(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "dim_int")
      .typed<Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("prod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, keepdim, dtype);
}
Tensor prod_dim_Dimname(const Tensor & self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prod", "dim_Dimname")
      .typed<Tensor (const Tensor &, Dimname, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("prod", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, keepdim, dtype);
}
std::tuple<Tensor,Tensor> qr(const Tensor & self, bool some) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::qr", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("qr", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, some);
}
QScheme qscheme(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::qscheme", "")
      .typed<QScheme (const Tensor &)>();
  RECORD_FUNCTION("qscheme", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<QScheme, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor quantize_per_channel(const Tensor & self, const Tensor & scales, const Tensor & zero_points, int64_t axis, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantize_per_channel", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, ScalarType)>();
  RECORD_FUNCTION("quantize_per_channel", std::vector<c10::IValue>({self, scales, zero_points}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, ScalarType>(op, c10::DispatchKey::Profiler, self, scales, zero_points, axis, dtype);
}
Tensor rand_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("rand_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
Tensor randint(int64_t high, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "")
      .typed<Tensor (int64_t, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, high, size, options);
}
Tensor randint_generator(int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "generator")
      .typed<Tensor (int64_t, IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Profiler, high, size, generator, options);
}
Tensor randint_low(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "low")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, low, high, size, options);
}
Tensor randint_low_generator(int64_t low, int64_t high, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint", "low_generator")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  RECORD_FUNCTION("randint", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Profiler, low, high, size, generator, options);
}
Tensor randn_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("randn_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
Tensor randperm(int64_t n, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randperm", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("randperm", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, n, options);
}
Tensor randperm_generator(int64_t n, c10::optional<Generator> generator, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randperm", "generator")
      .typed<Tensor (int64_t, c10::optional<Generator>, const TensorOptions &)>();
  RECORD_FUNCTION("randperm", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Profiler, n, generator, options);
}
Tensor reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor rename(const Tensor & self, c10::optional<DimnameList> names) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rename", "")
      .typed<Tensor (const Tensor &, c10::optional<DimnameList>)>();
  RECORD_FUNCTION("rename", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<DimnameList>>(op, c10::DispatchKey::Profiler, self, names);
}
Tensor & rename_(Tensor & self, c10::optional<DimnameList> names) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rename_", "")
      .typed<Tensor & (Tensor &, c10::optional<DimnameList>)>();
  RECORD_FUNCTION("rename_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, c10::optional<DimnameList>>(op, c10::DispatchKey::Profiler, self, names);
}
Tensor repeat_interleave_Tensor(const Tensor & repeats) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::repeat_interleave", "Tensor")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("repeat_interleave", std::vector<c10::IValue>({repeats}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, repeats);
}
Tensor repeat_interleave_self_Tensor(const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::repeat_interleave", "self_Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>();
  RECORD_FUNCTION("repeat_interleave", std::vector<c10::IValue>({self, repeats}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, repeats, dim);
}
Tensor repeat_interleave_self_int(const Tensor & self, int64_t repeats, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::repeat_interleave", "self_int")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>)>();
  RECORD_FUNCTION("repeat_interleave", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, repeats, dim);
}
Tensor replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor replication_pad2d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor & replication_pad2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & replication_pad3d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
void retain_grad(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::retain_grad", "")
      .typed<void (const Tensor &)>();
  RECORD_FUNCTION("retain_grad", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_tanh_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rnn_tanh_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor round(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::round", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("round", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & round_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::round_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("round_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor rsqrt(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsqrt", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("rsqrt", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & rsqrt_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsqrt_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("rsqrt_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor scalar_tensor(Scalar s, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scalar_tensor", "")
      .typed<Tensor (Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("scalar_tensor", std::vector<c10::IValue>({s}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, s, options);
}
Tensor searchsorted_Tensor(const Tensor & sorted_sequence, const Tensor & self, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::searchsorted", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("searchsorted", std::vector<c10::IValue>({sorted_sequence, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, sorted_sequence, self, out_int32, right);
}
Tensor searchsorted_Scalar(const Tensor & sorted_sequence, Scalar self, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::searchsorted", "Scalar")
      .typed<Tensor (const Tensor &, Scalar, bool, bool)>();
  RECORD_FUNCTION("searchsorted", std::vector<c10::IValue>({sorted_sequence, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, bool, bool>(op, c10::DispatchKey::Profiler, sorted_sequence, self, out_int32, right);
}
Tensor & set__source_Storage(Tensor & self, Storage source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::set_", "source_Storage")
      .typed<Tensor & (Tensor &, Storage)>();
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Storage>(op, c10::DispatchKey::Profiler, self, source);
}
Tensor & set__source_Storage_storage_offset(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::set_", "source_Storage_storage_offset")
      .typed<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, source, storage_offset, size, stride);
}
Tensor & set__source_Tensor(Tensor & self, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::set_", "source_Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, source);
}
Tensor & set_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::set_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("set_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
void set_data(const Tensor & self, const Tensor & new_data) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::set_data", "")
      .typed<void (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("set_data", std::vector<c10::IValue>({self, new_data}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, new_data);
}
Tensor & set_quantizer_(Tensor & self, ConstQuantizerPtr quantizer) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::set_quantizer_", "")
      .typed<Tensor & (Tensor &, ConstQuantizerPtr)>();
  RECORD_FUNCTION("set_quantizer_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, ConstQuantizerPtr>(op, c10::DispatchKey::Profiler, self, quantizer);
}
Tensor & sigmoid_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & sign_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sign", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sign_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor slow_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("slow_conv3d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
Tensor slow_conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose2d_backward_out_grad_output(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d_backward", "grad_output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("slow_conv_transpose2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, columns, ones}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}
Tensor & slow_conv_transpose3d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose3d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor & smooth_l1_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
}
Tensor soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("soft_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &)>();
  RECORD_FUNCTION("softplus_backward", std::vector<c10::IValue>({grad_output, self, beta, threshold, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, beta, threshold, output);
}
std::tuple<Tensor &,Tensor &> solve_out_solution(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::solve", "solution")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("solve_out", std::vector<c10::IValue>({solution, lu, self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, solution, lu, self, A);
}
Tensor sparse_coo_tensor_size(IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_coo_tensor", "size")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("sparse_coo_tensor", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, options);
}
Tensor sparse_coo_tensor_indices(const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_coo_tensor", "indices")
      .typed<Tensor (const Tensor &, const Tensor &, const TensorOptions &)>();
  RECORD_FUNCTION("sparse_coo_tensor", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const TensorOptions &>(op, c10::DispatchKey::Profiler, indices, values, options);
}
Tensor sparse_coo_tensor_indices_size(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_coo_tensor", "indices_size")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("sparse_coo_tensor", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, indices, values, size, options);
}
Tensor sparse_mask(const Tensor & self, const Tensor & mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_mask", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sparse_mask", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask);
}
Tensor & sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_resize_", "")
      .typed<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>();
  RECORD_FUNCTION("sparse_resize_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, size, sparse_dim, dense_dim);
}
Tensor & sqrt_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sqrt", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sqrt_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor square(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::square", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("square", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & square_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::square_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("square_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sspaddmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("sspaddmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, mat1, mat2, beta, alpha);
}
std::tuple<Tensor &,Tensor &,Tensor &> svd_out_U(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::svd", "U")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("svd_out", std::vector<c10::IValue>({U, S, V, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, U, S, V, self, some, compute_uv);
}
Tensor & tan_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tan", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tan_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor tanh(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("tanh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & tanh_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("tanh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & tanh_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tanh_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output);
}
Tensor & thnn_conv2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,2>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
Tensor & thnn_conv_depthwise2d_forward_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_forward", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_forward_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor & threshold_out_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::threshold", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("threshold_out", std::vector<c10::IValue>({out, self, threshold, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, threshold, value);
}
Tensor to_dtype_layout(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to", "dtype_layout")
      .typed<Tensor (const Tensor &, const TensorOptions &, bool, bool, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, bool, bool, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, non_blocking, copy, memory_format);
}
Tensor to_device(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to", "device")
      .typed<Tensor (const Tensor &, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Device, ScalarType, bool, bool, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, device, dtype, non_blocking, copy, memory_format);
}
Tensor to_dtype(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to", "dtype")
      .typed<Tensor (const Tensor &, ScalarType, bool, bool, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, ScalarType, bool, bool, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, dtype, non_blocking, copy, memory_format);
}
Tensor to_other(const Tensor & self, const Tensor & other, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to", "other")
      .typed<Tensor (const Tensor &, const Tensor &, bool, bool, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("to", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool, bool, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, other, non_blocking, copy, memory_format);
}
Tensor to_mkldnn_backward(const Tensor & grad, const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_mkldnn_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("to_mkldnn_backward", std::vector<c10::IValue>({grad, input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad, input);
}
Tensor trace(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trace", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("trace", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & tril_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tril", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("tril_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, diagonal);
}
Tensor triu(const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triu", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("triu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, diagonal);
}
Tensor & triu_(Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triu_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("triu_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, diagonal);
}
Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triu_indices", "")
      .typed<Tensor (int64_t, int64_t, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("triu_indices", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, row, col, offset, options);
}
Tensor true_divide_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::true_divide", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("true_divide", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor true_divide_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::true_divide", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("true_divide", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & true_divide__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::true_divide_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("true_divide_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & true_divide__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::true_divide_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("true_divide_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor type_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::type_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("type_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor unflatten_Dimname(const Tensor & self, Dimname dim, IntArrayRef sizes, DimnameList names) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unflatten", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, IntArrayRef, DimnameList)>();
  RECORD_FUNCTION("unflatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, IntArrayRef, DimnameList>(op, c10::DispatchKey::Profiler, self, dim, sizes, names);
}
Tensor unflatten_int(const Tensor & self, int64_t dim, IntArrayRef sizes, DimnameList names) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unflatten", "int")
      .typed<Tensor (const Tensor &, int64_t, IntArrayRef, DimnameList)>();
  RECORD_FUNCTION("unflatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, IntArrayRef, DimnameList>(op, c10::DispatchKey::Profiler, self, dim, sizes, names);
}
std::tuple<Tensor,Tensor,Tensor> unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_dim", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, int64_t, bool, bool, bool)>();
  RECORD_FUNCTION("unique_dim", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, int64_t, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, sorted, return_inverse, return_counts);
}
Tensor unsqueeze(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unsqueeze", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("unsqueeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & unsqueeze_(Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unsqueeze_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("unsqueeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor upsample_bicubic2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bicubic2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}
Tensor upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales_d, scales_h, scales_w);
}
Tensor & upsample_trilinear3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}
Tensor & zeros_out_out(Tensor & out, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::zeros", "out")
      .typed<Tensor & (Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("zeros_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, size);
}
}  // namespace
}  // namespace ProfiledType

namespace {

TORCH_LIBRARY_IMPL(aten, Profiler, m) {
  m.impl("__and__.Scalar", TORCH_FN(ProfiledType::__and___Scalar));
  m.impl("__and__.Tensor", TORCH_FN(ProfiledType::__and___Tensor));
  m.impl_UNBOXED("__iand__.Scalar", &ProfiledType::__iand___Scalar);
  m.impl_UNBOXED("__iand__.Tensor", &ProfiledType::__iand___Tensor);
  m.impl("_adaptive_avg_pool2d_backward", TORCH_FN(ProfiledType::_adaptive_avg_pool2d_backward));
  m.impl_UNBOXED("_amp_non_finite_check_and_unscale_", &ProfiledType::_amp_non_finite_check_and_unscale_);
  m.impl("_cast_Long", TORCH_FN(ProfiledType::_cast_Long));
  m.impl("_cat", TORCH_FN(ProfiledType::_cat));
  m.impl("_cdist_forward", TORCH_FN(ProfiledType::_cdist_forward));
  m.impl("_choose_qparams_per_tensor", TORCH_FN(ProfiledType::_choose_qparams_per_tensor));
  m.impl_UNBOXED("_convolution_double_backward", &ProfiledType::_convolution_double_backward);
  m.impl("_cufft_clear_plan_cache", TORCH_FN(ProfiledType::_cufft_clear_plan_cache));
  m.impl("_cufft_get_plan_cache_max_size", TORCH_FN(ProfiledType::_cufft_get_plan_cache_max_size));
  m.impl("_cumprod", TORCH_FN(ProfiledType::_cumprod));
  m.impl("_debug_has_internal_overlap", TORCH_FN(ProfiledType::_debug_has_internal_overlap));
  m.impl("_dimI", TORCH_FN(ProfiledType::_dimI));
  m.impl_UNBOXED("_empty_affine_quantized", &ProfiledType::_empty_affine_quantized);
  m.impl("_fft_with_size", TORCH_FN(ProfiledType::_fft_with_size));
  m.impl("_has_compatible_shallow_copy_type", TORCH_FN(ProfiledType::_has_compatible_shallow_copy_type));
  m.impl("_log_softmax", TORCH_FN(ProfiledType::_log_softmax));
  m.impl("_lu_with_info", TORCH_FN(ProfiledType::_lu_with_info));
  m.impl_UNBOXED("_multinomial_alias_draw", &ProfiledType::_multinomial_alias_draw);
  m.impl("_nnpack_spatial_convolution_backward", TORCH_FN(ProfiledType::_nnpack_spatial_convolution_backward));
  m.impl("_nnpack_spatial_convolution_backward_input", TORCH_FN(ProfiledType::_nnpack_spatial_convolution_backward_input));
  m.impl("_nnz", TORCH_FN(ProfiledType::_nnz));
  m.impl("_pack_padded_sequence", TORCH_FN(ProfiledType::_pack_padded_sequence));
  m.impl("_pad_packed_sequence", TORCH_FN(ProfiledType::_pad_packed_sequence));
  m.impl("_qr_helper", TORCH_FN(ProfiledType::_qr_helper));
  m.impl("_reshape_from_tensor", TORCH_FN(ProfiledType::_reshape_from_tensor));
  m.impl_UNBOXED("_sobol_engine_ff_", &ProfiledType::_sobol_engine_ff_);
  m.impl_UNBOXED("_sobol_engine_initialize_state_", &ProfiledType::_sobol_engine_initialize_state_);
  m.impl_UNBOXED("_sparse_log_softmax_backward_data", &ProfiledType::_sparse_log_softmax_backward_data);
  m.impl("_sparse_mm", TORCH_FN(ProfiledType::_sparse_mm));
  m.impl("_test_serialization_subcmul", TORCH_FN(ProfiledType::_test_serialization_subcmul));
  m.impl("_use_cudnn_ctc_loss", TORCH_FN(ProfiledType::_use_cudnn_ctc_loss));
  m.impl("_weight_norm", TORCH_FN(ProfiledType::_weight_norm));
  m.impl_UNBOXED("absolute.out", &ProfiledType::absolute_out_out);
  m.impl("acos", TORCH_FN(ProfiledType::acos));
  m.impl_UNBOXED("acos_", &ProfiledType::acos_);
  m.impl("adaptive_avg_pool3d", TORCH_FN(ProfiledType::adaptive_avg_pool3d));
  m.impl_UNBOXED("adaptive_avg_pool3d_backward.grad_input", &ProfiledType::adaptive_avg_pool3d_backward_out_grad_input);
  m.impl("add.Tensor", TORCH_FN(ProfiledType::add_Tensor));
  m.impl("add.Scalar", TORCH_FN(ProfiledType::add_Scalar));
  m.impl_UNBOXED("add_.Tensor", &ProfiledType::add__Tensor);
  m.impl_UNBOXED("add_.Scalar", &ProfiledType::add__Scalar);
  m.impl_UNBOXED("addbmm.out", &ProfiledType::addbmm_out_out);
  m.impl("affine_grid_generator_backward", TORCH_FN(ProfiledType::affine_grid_generator_backward));
  m.impl("alias", TORCH_FN(ProfiledType::alias));
  m.impl_UNBOXED("all.out", &ProfiledType::all_out_out);
  m.impl_UNBOXED("all.dimname_out", &ProfiledType::all_out_dimname_out);
  m.impl("allclose", TORCH_FN(ProfiledType::allclose));
  m.impl_UNBOXED("any.out", &ProfiledType::any_out_out);
  m.impl_UNBOXED("any.dimname_out", &ProfiledType::any_out_dimname_out);
  m.impl_UNBOXED("arange", &ProfiledType::arange);
  m.impl_UNBOXED("arange.start", &ProfiledType::arange_start);
  m.impl_UNBOXED("arange.start_step", &ProfiledType::arange_start_step);
  m.impl("asin", TORCH_FN(ProfiledType::asin));
  m.impl_UNBOXED("asin_", &ProfiledType::asin_);
  m.impl("avg_pool3d_backward", TORCH_FN(ProfiledType::avg_pool3d_backward));
  m.impl_UNBOXED("batch_norm_elemt.out", &ProfiledType::batch_norm_elemt_out_out);
  m.impl_UNBOXED("batch_norm_gather_stats", &ProfiledType::batch_norm_gather_stats);
  m.impl("batch_norm_stats", TORCH_FN(ProfiledType::batch_norm_stats));
  m.impl_UNBOXED("bernoulli", &ProfiledType::bernoulli);
  m.impl_UNBOXED("bernoulli.p", &ProfiledType::bernoulli_p);
  m.impl_UNBOXED("bernoulli_.Tensor", &ProfiledType::bernoulli__Tensor);
  m.impl_UNBOXED("bernoulli_.float", &ProfiledType::bernoulli__float);
  m.impl_UNBOXED("binary_cross_entropy_with_logits_backward", &ProfiledType::binary_cross_entropy_with_logits_backward);
  m.impl("cat", TORCH_FN(ProfiledType::cat));
  m.impl_UNBOXED("cat.names", &ProfiledType::cat_names);
  m.impl_UNBOXED("cauchy_", &ProfiledType::cauchy_);
  m.impl_UNBOXED("ceil.out", &ProfiledType::ceil_out_out);
  m.impl("chunk", TORCH_FN(ProfiledType::chunk));
  m.impl("col2im", TORCH_FN(ProfiledType::col2im));
  m.impl_UNBOXED("col2im_backward.grad_input", &ProfiledType::col2im_backward_out_grad_input);
  m.impl_UNBOXED("conv1d", &ProfiledType::conv1d);
  m.impl("conv_tbc_backward", TORCH_FN(ProfiledType::conv_tbc_backward));
  m.impl("cosine_embedding_loss", TORCH_FN(ProfiledType::cosine_embedding_loss));
  m.impl("cudnn_affine_grid_generator", TORCH_FN(ProfiledType::cudnn_affine_grid_generator));
  m.impl_UNBOXED("cudnn_convolution.deprecated", &ProfiledType::cudnn_convolution_deprecated);
  m.impl("cudnn_convolution", TORCH_FN(ProfiledType::cudnn_convolution));
  m.impl("cudnn_convolution_transpose_backward_weight", TORCH_FN(ProfiledType::cudnn_convolution_transpose_backward_weight));
  m.impl_UNBOXED("cumprod", &ProfiledType::cumprod);
  m.impl_UNBOXED("cumprod.dimname", &ProfiledType::cumprod_dimname);
  m.impl("dequantize.self", TORCH_FN(ProfiledType::dequantize_self));
  m.impl("dequantize.tensors", TORCH_FN(ProfiledType::dequantize_tensors));
  m.impl("det", TORCH_FN(ProfiledType::det));
  m.impl("detach", TORCH_FN(ProfiledType::detach));
  m.impl_UNBOXED("detach_", &ProfiledType::detach_);
  m.impl_UNBOXED("div.out", &ProfiledType::div_out_out);
  m.impl("dot", TORCH_FN(ProfiledType::dot));
  m.impl("einsum", TORCH_FN(ProfiledType::einsum));
  m.impl("elu_backward", TORCH_FN(ProfiledType::elu_backward));
  m.impl("embedding", TORCH_FN(ProfiledType::embedding));
  m.impl("embedding_sparse_backward", TORCH_FN(ProfiledType::embedding_sparse_backward));
  m.impl_UNBOXED("empty_meta", &ProfiledType::empty_meta);
  m.impl("erf", TORCH_FN(ProfiledType::erf));
  m.impl_UNBOXED("erf_", &ProfiledType::erf_);
  m.impl_UNBOXED("exp.out", &ProfiledType::exp_out_out);
  m.impl("expand_as", TORCH_FN(ProfiledType::expand_as));
  m.impl_UNBOXED("eye.out", &ProfiledType::eye_out_out);
  m.impl_UNBOXED("eye.m_out", &ProfiledType::eye_out_m_out);
  m.impl("fake_quantize_per_channel_affine_backward", TORCH_FN(ProfiledType::fake_quantize_per_channel_affine_backward));
  m.impl("fbgemm_linear_fp16_weight", TORCH_FN(ProfiledType::fbgemm_linear_fp16_weight));
  m.impl("fbgemm_linear_int8_weight", TORCH_FN(ProfiledType::fbgemm_linear_int8_weight));
  m.impl("fbgemm_pack_gemm_matrix_fp16", TORCH_FN(ProfiledType::fbgemm_pack_gemm_matrix_fp16));
  m.impl("feature_alpha_dropout", TORCH_FN(ProfiledType::feature_alpha_dropout));
  m.impl_UNBOXED("feature_alpha_dropout_", &ProfiledType::feature_alpha_dropout_);
  m.impl("flip", TORCH_FN(ProfiledType::flip));
  m.impl("fmod.Scalar", TORCH_FN(ProfiledType::fmod_Scalar));
  m.impl("fmod.Tensor", TORCH_FN(ProfiledType::fmod_Tensor));
  m.impl_UNBOXED("fmod_.Scalar", &ProfiledType::fmod__Scalar);
  m.impl_UNBOXED("fmod_.Tensor", &ProfiledType::fmod__Tensor);
  m.impl("frac", TORCH_FN(ProfiledType::frac));
  m.impl_UNBOXED("frac_", &ProfiledType::frac_);
  m.impl("fractional_max_pool2d", TORCH_FN(ProfiledType::fractional_max_pool2d));
  m.impl_UNBOXED("fractional_max_pool2d_backward.grad_input", &ProfiledType::fractional_max_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("fractional_max_pool3d.output", &ProfiledType::fractional_max_pool3d_out_output);
  m.impl("frobenius_norm", TORCH_FN(ProfiledType::frobenius_norm));
  m.impl("frobenius_norm.dim", TORCH_FN(ProfiledType::frobenius_norm_dim));
  m.impl_UNBOXED("from_file", &ProfiledType::from_file);
  m.impl_UNBOXED("geqrf.a", &ProfiledType::geqrf_out_a);
  m.impl_UNBOXED("ger.out", &ProfiledType::ger_out_out);
  m.impl_UNBOXED("glu.out", &ProfiledType::glu_out_out);
  m.impl_UNBOXED("hardsigmoid.out", &ProfiledType::hardsigmoid_out_out);
  m.impl_UNBOXED("hardswish.out", &ProfiledType::hardswish_out_out);
  m.impl("hardtanh", TORCH_FN(ProfiledType::hardtanh));
  m.impl_UNBOXED("hardtanh_", &ProfiledType::hardtanh_);
  m.impl_UNBOXED("hardtanh_backward.grad_input", &ProfiledType::hardtanh_backward_out_grad_input);
  m.impl("im2col", TORCH_FN(ProfiledType::im2col));
  m.impl_UNBOXED("im2col_backward.grad_input", &ProfiledType::im2col_backward_out_grad_input);
  m.impl_UNBOXED("instance_norm", &ProfiledType::instance_norm);
  m.impl("int_repr", TORCH_FN(ProfiledType::int_repr));
  m.impl("is_distributed", TORCH_FN(ProfiledType::is_distributed));
  m.impl("isinf", TORCH_FN(ProfiledType::isinf));
  m.impl("isnan", TORCH_FN(ProfiledType::isnan));
  m.impl_UNBOXED("kthvalue.values", &ProfiledType::kthvalue_out_values);
  m.impl_UNBOXED("kthvalue.dimname_out", &ProfiledType::kthvalue_out_dimname_out);
  m.impl("l1_loss_backward", TORCH_FN(ProfiledType::l1_loss_backward));
  m.impl_UNBOXED("lgamma.out", &ProfiledType::lgamma_out_out);
  m.impl_UNBOXED("linspace", &ProfiledType::linspace);
  m.impl("log", TORCH_FN(ProfiledType::log));
  m.impl_UNBOXED("log1p.out", &ProfiledType::log1p_out_out);
  m.impl("log2", TORCH_FN(ProfiledType::log2));
  m.impl_UNBOXED("log2_", &ProfiledType::log2_);
  m.impl_UNBOXED("log_", &ProfiledType::log_);
  m.impl("log_sigmoid_forward", TORCH_FN(ProfiledType::log_sigmoid_forward));
  m.impl_UNBOXED("log_softmax.int", &ProfiledType::log_softmax_int);
  m.impl_UNBOXED("log_softmax.Dimname", &ProfiledType::log_softmax_Dimname);
  m.impl("logaddexp", TORCH_FN(ProfiledType::logaddexp));
  m.impl("logaddexp2", TORCH_FN(ProfiledType::logaddexp2));
  m.impl_UNBOXED("logical_and.out", &ProfiledType::logical_and_out_out);
  m.impl_UNBOXED("logical_not.out", &ProfiledType::logical_not_out_out);
  m.impl("lstsq", TORCH_FN(ProfiledType::lstsq));
  m.impl_UNBOXED("lu_solve.out", &ProfiledType::lu_solve_out_out);
  m.impl("masked_fill.Scalar", TORCH_FN(ProfiledType::masked_fill_Scalar));
  m.impl("masked_fill.Tensor", TORCH_FN(ProfiledType::masked_fill_Tensor));
  m.impl_UNBOXED("masked_fill_.Scalar", &ProfiledType::masked_fill__Scalar);
  m.impl_UNBOXED("masked_fill_.Tensor", &ProfiledType::masked_fill__Tensor);
  m.impl("masked_scatter", TORCH_FN(ProfiledType::masked_scatter));
  m.impl_UNBOXED("masked_scatter_", &ProfiledType::masked_scatter_);
  m.impl("max_pool1d", TORCH_FN(ProfiledType::max_pool1d));
  m.impl("max_pool2d_with_indices", TORCH_FN(ProfiledType::max_pool2d_with_indices));
  m.impl_UNBOXED("max_pool2d_with_indices_backward.grad_input", &ProfiledType::max_pool2d_with_indices_backward_out_grad_input);
  m.impl_UNBOXED("max_pool3d_with_indices.out", &ProfiledType::max_pool3d_with_indices_out_out);
  m.impl("max_unpool3d_backward", TORCH_FN(ProfiledType::max_unpool3d_backward));
  m.impl("max_values", TORCH_FN(ProfiledType::max_values));
  m.impl_UNBOXED("max_values.names", &ProfiledType::max_values_names);
  m.impl_UNBOXED("mean", &ProfiledType::mean);
  m.impl_UNBOXED("mean.dim", &ProfiledType::mean_dim);
  m.impl_UNBOXED("mean.names_dim", &ProfiledType::mean_names_dim);
  m.impl("median.dim", TORCH_FN(ProfiledType::median_dim));
  m.impl_UNBOXED("median.names_dim", &ProfiledType::median_names_dim);
  m.impl("median", TORCH_FN(ProfiledType::median));
  m.impl_UNBOXED("miopen_batch_norm_backward", &ProfiledType::miopen_batch_norm_backward);
  m.impl_UNBOXED("miopen_depthwise_convolution", &ProfiledType::miopen_depthwise_convolution);
  m.impl_UNBOXED("miopen_rnn", &ProfiledType::miopen_rnn);
  m.impl("mkldnn_max_pool2d", TORCH_FN(ProfiledType::mkldnn_max_pool2d));
  m.impl_UNBOXED("mm.out", &ProfiledType::mm_out_out);
  m.impl_UNBOXED("multi_margin_loss", &ProfiledType::multi_margin_loss);
  m.impl_UNBOXED("multi_margin_loss_backward.grad_input", &ProfiledType::multi_margin_loss_backward_out_grad_input);
  m.impl("mv", TORCH_FN(ProfiledType::mv));
  m.impl_UNBOXED("native_layer_norm", &ProfiledType::native_layer_norm);
  m.impl_UNBOXED("nll_loss2d_forward", &ProfiledType::nll_loss2d_forward);
  m.impl_UNBOXED("nll_loss_forward", &ProfiledType::nll_loss_forward);
  m.impl("norm_except_dim", TORCH_FN(ProfiledType::norm_except_dim));
  m.impl_UNBOXED("ones.names", &ProfiledType::ones_names);
  m.impl_UNBOXED("ones", &ProfiledType::ones);
  m.impl("pin_memory", TORCH_FN(ProfiledType::pin_memory));
  m.impl("polygamma", TORCH_FN(ProfiledType::polygamma));
  m.impl_UNBOXED("polygamma_", &ProfiledType::polygamma_);
  m.impl("pow.Tensor_Scalar", TORCH_FN(ProfiledType::pow_Tensor_Scalar));
  m.impl("pow.Tensor_Tensor", TORCH_FN(ProfiledType::pow_Tensor_Tensor));
  m.impl("pow.Scalar", TORCH_FN(ProfiledType::pow_Scalar));
  m.impl_UNBOXED("pow_.Scalar", &ProfiledType::pow__Scalar);
  m.impl_UNBOXED("pow_.Tensor", &ProfiledType::pow__Tensor);
  m.impl("prelu", TORCH_FN(ProfiledType::prelu));
  m.impl_UNBOXED("prod", &ProfiledType::prod);
  m.impl_UNBOXED("prod.dim_int", &ProfiledType::prod_dim_int);
  m.impl_UNBOXED("prod.dim_Dimname", &ProfiledType::prod_dim_Dimname);
  m.impl("qr", TORCH_FN(ProfiledType::qr));
  m.impl("qscheme", TORCH_FN(ProfiledType::qscheme));
  m.impl_UNBOXED("quantize_per_channel", &ProfiledType::quantize_per_channel);
  m.impl_UNBOXED("rand_like", &ProfiledType::rand_like);
  m.impl_UNBOXED("randint", &ProfiledType::randint);
  m.impl_UNBOXED("randint.generator", &ProfiledType::randint_generator);
  m.impl_UNBOXED("randint.low", &ProfiledType::randint_low);
  m.impl_UNBOXED("randint.low_generator", &ProfiledType::randint_low_generator);
  m.impl_UNBOXED("randn_like", &ProfiledType::randn_like);
  m.impl_UNBOXED("randperm", &ProfiledType::randperm);
  m.impl_UNBOXED("randperm.generator", &ProfiledType::randperm_generator);
  m.impl("reflection_pad2d_backward", TORCH_FN(ProfiledType::reflection_pad2d_backward));
  m.impl_UNBOXED("rename", &ProfiledType::rename);
  m.impl_UNBOXED("rename_", &ProfiledType::rename_);
  m.impl("repeat_interleave.Tensor", TORCH_FN(ProfiledType::repeat_interleave_Tensor));
  m.impl("repeat_interleave.self_Tensor", TORCH_FN(ProfiledType::repeat_interleave_self_Tensor));
  m.impl("repeat_interleave.self_int", TORCH_FN(ProfiledType::repeat_interleave_self_int));
  m.impl("replication_pad1d_backward", TORCH_FN(ProfiledType::replication_pad1d_backward));
  m.impl("replication_pad2d", TORCH_FN(ProfiledType::replication_pad2d));
  m.impl_UNBOXED("replication_pad2d_backward.grad_input", &ProfiledType::replication_pad2d_backward_out_grad_input);
  m.impl_UNBOXED("replication_pad3d.out", &ProfiledType::replication_pad3d_out_out);
  m.impl("retain_grad", TORCH_FN(ProfiledType::retain_grad));
  m.impl_UNBOXED("rnn_tanh_cell", &ProfiledType::rnn_tanh_cell);
  m.impl("round", TORCH_FN(ProfiledType::round));
  m.impl_UNBOXED("round_", &ProfiledType::round_);
  m.impl("rsqrt", TORCH_FN(ProfiledType::rsqrt));
  m.impl_UNBOXED("rsqrt_", &ProfiledType::rsqrt_);
  m.impl_UNBOXED("scalar_tensor", &ProfiledType::scalar_tensor);
  m.impl("searchsorted.Tensor", TORCH_FN(ProfiledType::searchsorted_Tensor));
  m.impl("searchsorted.Scalar", TORCH_FN(ProfiledType::searchsorted_Scalar));
  m.impl_UNBOXED("set_.source_Storage", &ProfiledType::set__source_Storage);
  m.impl_UNBOXED("set_.source_Storage_storage_offset", &ProfiledType::set__source_Storage_storage_offset);
  m.impl_UNBOXED("set_.source_Tensor", &ProfiledType::set__source_Tensor);
  m.impl_UNBOXED("set_", &ProfiledType::set_);
  m.impl("set_data", TORCH_FN(ProfiledType::set_data));
  m.impl_UNBOXED("set_quantizer_", &ProfiledType::set_quantizer_);
  m.impl_UNBOXED("sigmoid.out", &ProfiledType::sigmoid_out_out);
  m.impl_UNBOXED("sign.out", &ProfiledType::sign_out_out);
  m.impl_UNBOXED("slow_conv3d", &ProfiledType::slow_conv3d);
  m.impl_UNBOXED("slow_conv3d_backward.grad_input", &ProfiledType::slow_conv3d_backward_out_grad_input);
  m.impl_UNBOXED("slow_conv_transpose2d", &ProfiledType::slow_conv_transpose2d);
  m.impl_UNBOXED("slow_conv_transpose2d_backward.grad_output", &ProfiledType::slow_conv_transpose2d_backward_out_grad_output);
  m.impl_UNBOXED("slow_conv_transpose3d.out", &ProfiledType::slow_conv_transpose3d_out_out);
  m.impl("smooth_l1_loss", TORCH_FN(ProfiledType::smooth_l1_loss));
  m.impl_UNBOXED("smooth_l1_loss_backward.grad_input", &ProfiledType::smooth_l1_loss_backward_out_grad_input);
  m.impl("soft_margin_loss_backward", TORCH_FN(ProfiledType::soft_margin_loss_backward));
  m.impl("softplus_backward", TORCH_FN(ProfiledType::softplus_backward));
  m.impl_UNBOXED("solve.solution", &ProfiledType::solve_out_solution);
  m.impl_UNBOXED("sparse_coo_tensor.size", &ProfiledType::sparse_coo_tensor_size);
  m.impl_UNBOXED("sparse_coo_tensor.indices", &ProfiledType::sparse_coo_tensor_indices);
  m.impl_UNBOXED("sparse_coo_tensor.indices_size", &ProfiledType::sparse_coo_tensor_indices_size);
  m.impl("sparse_mask", TORCH_FN(ProfiledType::sparse_mask));
  m.impl_UNBOXED("sparse_resize_", &ProfiledType::sparse_resize_);
  m.impl_UNBOXED("sqrt.out", &ProfiledType::sqrt_out_out);
  m.impl("square", TORCH_FN(ProfiledType::square));
  m.impl_UNBOXED("square_", &ProfiledType::square_);
  m.impl("sspaddmm", TORCH_FN(ProfiledType::sspaddmm));
  m.impl_UNBOXED("svd.U", &ProfiledType::svd_out_U);
  m.impl_UNBOXED("tan.out", &ProfiledType::tan_out_out);
  m.impl("tanh", TORCH_FN(ProfiledType::tanh));
  m.impl_UNBOXED("tanh_", &ProfiledType::tanh_);
  m.impl_UNBOXED("tanh_backward.grad_input", &ProfiledType::tanh_backward_out_grad_input);
  m.impl_UNBOXED("thnn_conv2d.out", &ProfiledType::thnn_conv2d_out_out);
  m.impl("thnn_conv_depthwise2d_backward.output_mask", TORCH_FN(ProfiledType::thnn_conv_depthwise2d_backward_output_mask));
  m.impl_UNBOXED("thnn_conv_depthwise2d_forward.out", &ProfiledType::thnn_conv_depthwise2d_forward_out_out);
  m.impl_UNBOXED("threshold.out", &ProfiledType::threshold_out_out);
  m.impl_UNBOXED("to.dtype_layout", &ProfiledType::to_dtype_layout);
  m.impl_UNBOXED("to.device", &ProfiledType::to_device);
  m.impl_UNBOXED("to.dtype", &ProfiledType::to_dtype);
  m.impl_UNBOXED("to.other", &ProfiledType::to_other);
  m.impl("to_mkldnn_backward", TORCH_FN(ProfiledType::to_mkldnn_backward));
  m.impl("trace", TORCH_FN(ProfiledType::trace));
  m.impl_UNBOXED("tril.out", &ProfiledType::tril_out_out);
  m.impl("triu", TORCH_FN(ProfiledType::triu));
  m.impl_UNBOXED("triu_", &ProfiledType::triu_);
  m.impl_UNBOXED("triu_indices", &ProfiledType::triu_indices);
  m.impl("true_divide.Tensor", TORCH_FN(ProfiledType::true_divide_Tensor));
  m.impl("true_divide.Scalar", TORCH_FN(ProfiledType::true_divide_Scalar));
  m.impl_UNBOXED("true_divide_.Tensor", &ProfiledType::true_divide__Tensor);
  m.impl_UNBOXED("true_divide_.Scalar", &ProfiledType::true_divide__Scalar);
  m.impl("type_as", TORCH_FN(ProfiledType::type_as));
  m.impl_UNBOXED("unflatten.Dimname", &ProfiledType::unflatten_Dimname);
  m.impl_UNBOXED("unflatten.int", &ProfiledType::unflatten_int);
  m.impl("unique_dim", TORCH_FN(ProfiledType::unique_dim));
  m.impl("unsqueeze", TORCH_FN(ProfiledType::unsqueeze));
  m.impl_UNBOXED("unsqueeze_", &ProfiledType::unsqueeze_);
  m.impl("upsample_bicubic2d", TORCH_FN(ProfiledType::upsample_bicubic2d));
  m.impl_UNBOXED("upsample_bicubic2d_backward.grad_input", &ProfiledType::upsample_bicubic2d_backward_out_grad_input);
  m.impl("upsample_nearest3d_backward", TORCH_FN(ProfiledType::upsample_nearest3d_backward));
  m.impl("upsample_trilinear3d", TORCH_FN(ProfiledType::upsample_trilinear3d));
  m.impl_UNBOXED("upsample_trilinear3d_backward.grad_input", &ProfiledType::upsample_trilinear3d_backward_out_grad_input);
  m.impl_UNBOXED("zeros.out", &ProfiledType::zeros_out_out);;
}

}  // namespace

} // namespace torch
