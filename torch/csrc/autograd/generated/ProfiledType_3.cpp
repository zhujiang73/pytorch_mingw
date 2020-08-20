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
Tensor & _addmv_impl_(Tensor & self, const Tensor & self2, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addmv_impl_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_addmv_impl_", std::vector<c10::IValue>({self, self2, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, self2, mat, vec, beta, alpha);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> _batch_norm_impl_index(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_batch_norm_impl_index", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>();
  RECORD_FUNCTION("_batch_norm_impl_index", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
std::tuple<Tensor,Tensor> _ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_ctc_loss", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool)>();
  RECORD_FUNCTION("_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}
Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_init_dropout_state", "")
      .typed<Tensor (double, bool, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("_cudnn_init_dropout_state", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, double, bool, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, dropout, train, dropout_seed, options);
}
Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn_flatten_weight", "")
      .typed<Tensor (TensorList, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("_cudnn_rnn_flatten_weight", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
}
Tensor _embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_dense_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_embedding_bag_dense_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, maximum_indices, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights);
}
Tensor _embedding_bag_per_sample_weights_backward(const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_per_sample_weights_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_embedding_bag_per_sample_weights_backward", std::vector<c10::IValue>({grad, weight, indices, offsets, offset2bag}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad, weight, indices, offsets, offset2bag, mode);
}
Tensor _empty_per_channel_affine_quantized(IntArrayRef size, const Tensor & scales, const Tensor & zero_points, int64_t axis, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_empty_per_channel_affine_quantized", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("_empty_per_channel_affine_quantized", std::vector<c10::IValue>({scales, zero_points}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, scales, zero_points, axis, options, memory_format);
}
Tensor _log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_log_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_log_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output, dim, self);
}
Tensor _lu_solve_helper(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_lu_solve_helper", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_lu_solve_helper", std::vector<c10::IValue>({self, LU_data, LU_pivots}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, LU_data, LU_pivots);
}
Tensor _make_per_channel_quantized_tensor(const Tensor & self, const Tensor & scale, const Tensor & zero_point, int64_t axis) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_make_per_channel_quantized_tensor", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_make_per_channel_quantized_tensor", std::vector<c10::IValue>({self, scale, zero_point}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point, axis);
}
Tensor _mkldnn_reshape(const Tensor & self, IntArrayRef shape) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mkldnn_reshape", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_mkldnn_reshape", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, shape);
}
std::tuple<Tensor,Tensor> _multinomial_alias_setup(const Tensor & probs) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_multinomial_alias_setup", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("_multinomial_alias_setup", std::vector<c10::IValue>({probs}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, probs);
}
Tensor _pack_padded_sequence_backward(const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pack_padded_sequence_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, bool)>();
  RECORD_FUNCTION("_pack_padded_sequence_backward", std::vector<c10::IValue>({grad, batch_sizes}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, bool>(op, c10::DispatchKey::Profiler, grad, input_size, batch_sizes, batch_first);
}
Tensor _shape_as_tensor(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_shape_as_tensor", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("_shape_as_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> _sobol_engine_draw(const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sobol_engine_draw", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, const Tensor &, int64_t, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("_sobol_engine_draw", std::vector<c10::IValue>({quasi, sobolstate}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, const Tensor &, int64_t, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, quasi, n, sobolstate, dimension, num_generated, dtype);
}
std::tuple<Tensor,Tensor> _solve_helper(const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_solve_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, A);
}
Tensor _standard_gamma_grad(const Tensor & self, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_standard_gamma_grad", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_standard_gamma_grad", std::vector<c10::IValue>({self, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, output);
}
std::tuple<Tensor,Tensor,Tensor> _svd_helper(const Tensor & self, bool some, bool compute_uv) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_svd_helper", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_svd_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, some, compute_uv);
}
std::tuple<Tensor,Tensor> _unique(const Tensor & self, bool sorted, bool return_inverse) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_unique", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_unique", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, sorted, return_inverse);
}
std::tuple<Tensor,Tensor,Tensor> _unique2(const Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_unique2", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("_unique2", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, sorted, return_inverse, return_counts);
}
std::tuple<Tensor,Tensor> _weight_norm_differentiable_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_differentiable_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm_differentiable_backward", std::vector<c10::IValue>({grad_w, saved_v, saved_g, saved_norms}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_w, saved_v, saved_g, saved_norms, dim);
}
Tensor absolute(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::absolute", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("absolute", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & absolute_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::absolute_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("absolute_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_avg_pool3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool1d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool2d_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, indices, self, output_size);
}
Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addbmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addbmm", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, batch1, batch2, beta, alpha);
}
Tensor & addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addbmm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addbmm_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, batch1, batch2, beta, alpha);
}
Tensor & addcdiv_out_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcdiv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcdiv_out", std::vector<c10::IValue>({out, self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, tensor1, tensor2, value);
}
Tensor & addmv_out_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmv", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmv_out", std::vector<c10::IValue>({out, self, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, mat, vec, beta, alpha);
}
Tensor all_dim(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::all", "dim")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("all", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor all_dimname(const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::all", "dimname")
      .typed<Tensor (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("all", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor all(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::all", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("all", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & angle_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::angle", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("angle_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor any_dim(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::any", "dim")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("any", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor any_dimname(const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::any", "dimname")
      .typed<Tensor (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("any", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor any(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::any", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("any", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argmin", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  RECORD_FUNCTION("argmin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::as_strided", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>();
  RECORD_FUNCTION("as_strided", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, size, stride, storage_offset);
}
Tensor & as_strided_(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::as_strided_", "")
      .typed<Tensor & (Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>();
  RECORD_FUNCTION("as_strided_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, size, stride, storage_offset);
}
Tensor & atanh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atanh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atanh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool)>();
  RECORD_FUNCTION("batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double, bool>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
Tensor batch_norm_elemt(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_elemt", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_elemt", std::vector<c10::IValue>({input, weight, bias, mean, invstd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, weight, bias, mean, invstd, eps);
}
Tensor bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bilinear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bilinear", std::vector<c10::IValue>({input1, input2, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input1, input2, weight, bias);
}
Tensor & binary_cross_entropy_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, weight, reduction);
}
Tensor binomial(const Tensor & count, const Tensor & prob, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binomial", "")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("binomial", std::vector<c10::IValue>({count, prob}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, count, prob, generator);
}
Tensor & bitwise_or_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_or_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & bitwise_or_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_or_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & bitwise_xor_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_xor_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & bitwise_xor_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_xor_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor block_diag(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::block_diag", "")
      .typed<Tensor (TensorList)>();
  RECORD_FUNCTION("block_diag", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
bool can_cast(ScalarType from, ScalarType to) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::can_cast", "")
      .typed<bool (ScalarType, ScalarType)>();
  RECORD_FUNCTION("can_cast", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, ScalarType, ScalarType>(op, c10::DispatchKey::Profiler, from, to);
}
Tensor ceil(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ceil", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("ceil", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & ceil_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ceil_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("ceil_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor channel_shuffle(const Tensor & self, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::channel_shuffle", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("channel_shuffle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, groups);
}
Tensor & cholesky_solve_out_out(Tensor & out, const Tensor & self, const Tensor & input2, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_solve", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_solve_out", std::vector<c10::IValue>({out, self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, input2, upper);
}
Tensor & clamp_min_out_out(Tensor & out, const Tensor & self, Scalar min) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_min", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_min_out", std::vector<c10::IValue>({out, self, min}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, min);
}
Tensor col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::col2im_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("col2im_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, kernel_size, dilation, padding, stride);
}
Tensor constant_pad_nd(const Tensor & self, IntArrayRef pad, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::constant_pad_nd", "")
      .typed<Tensor (const Tensor &, IntArrayRef, Scalar)>();
  RECORD_FUNCTION("constant_pad_nd", std::vector<c10::IValue>({self, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, Scalar>(op, c10::DispatchKey::Profiler, self, pad, value);
}
Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("conv2d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, groups);
}
Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose1d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("conv_transpose1d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::copy_", "")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("copy_", std::vector<c10::IValue>({self, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, src, non_blocking);
}
Tensor & copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::copy_sparse_to_sparse_", "")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("copy_sparse_to_sparse_", std::vector<c10::IValue>({self, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, src, non_blocking);
}
Tensor & cosh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cosh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & cross_out_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cross", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, c10::optional<int64_t>)>();
  RECORD_FUNCTION("cross_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, out, self, other, dim);
}
Tensor ctc_loss_IntList(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ctc_loss", "IntList")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}
Tensor ctc_loss_Tensor(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ctc_loss", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("ctc_loss", std::vector<c10::IValue>({log_probs, targets, input_lengths, target_lengths}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}
Tensor cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_affine_grid_generator_backward", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("cudnn_affine_grid_generator_backward", std::vector<c10::IValue>({grad}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad, N, C, H, W);
}
std::tuple<Tensor,Tensor> cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,2> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,2>)>();
  RECORD_FUNCTION("cudnn_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,2>>(op, c10::DispatchKey::Profiler, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor cudnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_backward_input", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cudnn_convolution_transpose_deprecated(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose", "deprecated")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_transpose", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_transpose", std::vector<c10::IValue>({self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor &,Tensor &> cummin_out_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummin", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("cummin_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, values, indices, self, dim);
}
std::tuple<Tensor &,Tensor &> cummin_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummin", "dimname_out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname)>();
  RECORD_FUNCTION("cummin_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, values, indices, self, dim);
}
Tensor diagflat(const Tensor & self, int64_t offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diagflat", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("diagflat", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, offset);
}
Tensor div_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::div", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("div", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor div_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::div", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("div", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & div__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::div_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("div_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & div__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::div_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("div_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
std::tuple<Tensor &,Tensor &> eig_out_e(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eig", "e")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("eig_out", std::vector<c10::IValue>({e, v, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, e, v, self, eigenvectors);
}
Tensor embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("embedding_backward", std::vector<c10::IValue>({grad, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}
Tensor & empty_out_out(Tensor & out, IntArrayRef size, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty", "out")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, out, size, memory_format);
}
Tensor & eq_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("eq_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & eq_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("eq_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor exp(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exp", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("exp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & exp_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::exp_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("exp_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor eye(int64_t n, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eye", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("eye", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, n, options);
}
Tensor eye_m(int64_t n, int64_t m, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eye", "m")
      .typed<Tensor (int64_t, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("eye", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, n, m, options);
}
Tensor & fill__Scalar(Tensor & self, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fill_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("fill_", std::vector<c10::IValue>({self, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, value);
}
Tensor & fill__Tensor(Tensor & self, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fill_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("fill_", std::vector<c10::IValue>({self, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, value);
}
Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, output_size, indices);
}
std::tuple<Tensor,Tensor> fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d", std::vector<c10::IValue>({self, random_samples}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, self, kernel_size, output_size, random_samples);
}
Tensor & fractional_max_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, output_size, indices);
}
Tensor & ge_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("ge_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & ge_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ge_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
std::tuple<Tensor,Tensor> geqrf(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geqrf", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("geqrf", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor ger(const Tensor & self, const Tensor & vec2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ger", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ger", std::vector<c10::IValue>({self, vec2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, vec2);
}
Tensor glu(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & glu_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, dim);
}
Tensor grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("grid_sampler", std::vector<c10::IValue>({input, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, input, grid, interpolation_mode, padding_mode, align_corners);
}
Tensor grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("grid_sampler_2d", std::vector<c10::IValue>({input, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, input, grid, interpolation_mode, padding_mode, align_corners);
}
Tensor & gt_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("gt_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & gt_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gt_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor hardsigmoid(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("hardsigmoid", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & hardsigmoid_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("hardsigmoid_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor hardswish(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardswish", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("hardswish", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & hardswish_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardswish_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("hardswish_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardtanh_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("hardtanh_backward", std::vector<c10::IValue>({grad_output, self, min_val, max_val}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, grad_output, self, min_val, max_val);
}
Tensor & histc_out_out(Tensor & out, const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::histc", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, Scalar, Scalar)>();
  RECORD_FUNCTION("histc_out", std::vector<c10::IValue>({out, self, min, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, bins, min, max);
}
Tensor & hspmm_out_out(Tensor & out, const Tensor & mat1, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hspmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hspmm_out", std::vector<c10::IValue>({out, mat1, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, mat1, mat2);
}
Tensor im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::im2col_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("im2col_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, input_size, kernel_size, dilation, padding, stride);
}
Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_add", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_add", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor index_add_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_add", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_add", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor & index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_add_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_add_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor & inverse_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::inverse", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("inverse_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
bool is_leaf(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_leaf", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_leaf", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_pinned(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_pinned", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_pinned", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor kl_div(const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kl_div", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("kl_div", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, target, reduction, log_target);
}
std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kthvalue", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("kthvalue", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, self, k, dim, keepdim);
}
std::tuple<Tensor,Tensor> kthvalue_dimname(const Tensor & self, int64_t k, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kthvalue", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, Dimname, bool)>();
  RECORD_FUNCTION("kthvalue", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, Dimname, bool>(op, c10::DispatchKey::Profiler, self, k, dim, keepdim);
}
Tensor & le_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("le_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & le_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("le_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & leaky_relu_out_out(Tensor & out, const Tensor & self, Scalar negative_slope) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("leaky_relu_out", std::vector<c10::IValue>({out, self, negative_slope}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, negative_slope);
}
Tensor lgamma(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lgamma", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("lgamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & lgamma_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lgamma_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("lgamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log10_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log10", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log10_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor log1p(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log1p", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log1p", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log1p_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log1p_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("log1p_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor logical_and(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_and", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_and", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & logical_and_(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_and_", "")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_and_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor logical_not(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_not", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("logical_not", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & logical_not_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_not_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("logical_not_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor,Tensor> lstm_input(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstm", "input")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool, bool)>();
  RECORD_FUNCTION("lstm", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool, bool>(op, c10::DispatchKey::Profiler, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor,Tensor> lstm_data(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstm", "data")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool)>();
  RECORD_FUNCTION("lstm", std::vector<c10::IValue>({data, batch_sizes}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, TensorList, TensorList, bool, int64_t, double, bool, bool>(op, c10::DispatchKey::Profiler, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
Tensor & lt_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("lt_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & lt_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lt_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lu_solve", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lu_solve", std::vector<c10::IValue>({self, LU_data, LU_pivots}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, LU_data, LU_pivots);
}
Tensor margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::margin_ranking_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("margin_ranking_loss", std::vector<c10::IValue>({input1, input2, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, input1, input2, target, margin, reduction);
}
Tensor & masked_select_out_out(Tensor & out, const Tensor & self, const Tensor & mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_select", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_select_out", std::vector<c10::IValue>({out, self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, mask);
}
Tensor matrix_power(const Tensor & self, int64_t n) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matrix_power", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("matrix_power", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, n);
}
Tensor max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool2d_with_indices_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool2d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool3d_with_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & max_pool3d_with_indices_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool3d_with_indices_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
std::tuple<Tensor &,Tensor &> min_out_dim_min(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "dim_min")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("min_out", std::vector<c10::IValue>({min, min_indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, min, min_indices, self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> min_out_names_dim_min(Tensor & min, Tensor & min_indices, const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "names_dim_min")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("min_out", std::vector<c10::IValue>({min, min_indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, min, min_indices, self, dim, keepdim);
}
Tensor & min_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("min_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor miopen_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor> miopen_depthwise_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>();
  RECORD_FUNCTION("miopen_depthwise_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Profiler, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor miopen_depthwise_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_depthwise_convolution_backward_input", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_depthwise_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> miopen_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_rnn_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>)>();
  RECORD_FUNCTION("miopen_rnn_backward", std::vector<c10::IValue>({input, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, dropout_state, reserve}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,4>>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}
Tensor mm(const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mat2);
}
Tensor & mse_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
}
Tensor & mul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mul_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multi_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multi_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, p, margin, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, p, margin, weight, reduction);
}
Tensor & multilabel_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
}
Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::narrow_copy", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("narrow_copy", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim, start, length);
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_group_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, double)>();
  RECORD_FUNCTION("native_group_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, double>(op, c10::DispatchKey::Profiler, input, weight, bias, N, C, HxW, group, eps);
}
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t M, int64_t N, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_layer_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("native_layer_norm_backward", std::vector<c10::IValue>({grad_out, input, mean, rstd, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_out, input, mean, rstd, weight, M, N, output_mask);
}
Tensor & neg_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::neg", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("neg_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor new_zeros(const Tensor & self, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_zeros", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("new_zeros", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, self, size, options);
}
std::vector<Tensor> nonzero_numpy(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nonzero_numpy", "")
      .typed<std::vector<Tensor> (const Tensor &)>();
  RECORD_FUNCTION("nonzero_numpy", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & nonzero_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nonzero", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("nonzero_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & norm_out_dtype_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "dtype_out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>();
  RECORD_FUNCTION("norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(op, c10::DispatchKey::Profiler, out, self, p, dim, keepdim, dtype);
}
Tensor & norm_out_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>();
  RECORD_FUNCTION("norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, self, p, dim, keepdim);
}
Tensor & norm_out_names_dtype_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "names_dtype_out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType)>();
  RECORD_FUNCTION("norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType>(op, c10::DispatchKey::Profiler, out, self, p, dim, keepdim, dtype);
}
Tensor & norm_out_names_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Scalar>, DimnameList, bool)>();
  RECORD_FUNCTION("norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, DimnameList, bool>(op, c10::DispatchKey::Profiler, out, self, p, dim, keepdim);
}
Tensor & normal_out_Tensor_float_out(Tensor & out, const Tensor & mean, double std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "Tensor_float_out")
      .typed<Tensor & (Tensor &, const Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out, mean}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, mean, std, generator);
}
Tensor & normal_out_float_Tensor_out(Tensor & out, double mean, const Tensor & std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "float_Tensor_out")
      .typed<Tensor & (Tensor &, double, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out, std}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, mean, std, generator);
}
Tensor & normal_out_Tensor_Tensor_out(Tensor & out, const Tensor & mean, const Tensor & std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "Tensor_Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out, mean, std}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, mean, std, generator);
}
Tensor & normal_out_float_float_out(Tensor & out, double mean, double std, IntArrayRef size, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "float_float_out")
      .typed<Tensor & (Tensor &, double, double, IntArrayRef, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, mean, std, size, generator);
}
Tensor numpy_T(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::numpy_T", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("numpy_T", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor one_hot(const Tensor & self, int64_t num_classes) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::one_hot", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("one_hot", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, num_classes);
}
Tensor ones_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ones_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("ones_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
int64_t output_nr(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::output_nr", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("output_nr", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor pdist(const Tensor & self, double p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pdist", "")
      .typed<Tensor (const Tensor &, double)>();
  RECORD_FUNCTION("pdist", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, p);
}
std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::prelu_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("prelu_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, weight);
}
double q_scale(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_scale", "")
      .typed<double (const Tensor &)>();
  RECORD_FUNCTION("q_scale", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<double, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor quantized_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_rnn_tanh_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_rnn_tanh_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor randint_like(const Tensor & self, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint_like", "")
      .typed<Tensor (const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("randint_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, high, options, memory_format);
}
Tensor randint_like_low_dtype(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randint_like", "low_dtype")
      .typed<Tensor (const Tensor &, int64_t, int64_t, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("randint_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, low, high, options, memory_format);
}
Tensor & renorm_out_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::renorm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, int64_t, Scalar)>();
  RECORD_FUNCTION("renorm_out", std::vector<c10::IValue>({out, self, p, maxnorm}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, int64_t, Scalar>(op, c10::DispatchKey::Profiler, out, self, p, dim, maxnorm);
}
Tensor replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad2d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor replication_pad3d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor & replication_pad3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::resize_", "")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("resize_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, size, memory_format);
}
Tensor scatter_src(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter", "src")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("scatter", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, src);
}
Tensor scatter_value(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter", "value")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>();
  RECORD_FUNCTION("scatter", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor scatter_dimname_src(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter", "dimname_src")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("scatter", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, src);
}
Tensor scatter_dimname_value(const Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter", "dimname_value")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, Scalar)>();
  RECORD_FUNCTION("scatter", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor & scatter__src(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_", "src")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("scatter_", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, src);
}
Tensor & scatter__value(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_", "value")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>();
  RECORD_FUNCTION("scatter_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor sigmoid(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("sigmoid", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sigmoid_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("sigmoid_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sigmoid_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output);
}
Tensor sign(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sign", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("sign", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sign_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sign_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("sign_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sinh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sinh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sinh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
int64_t size_int(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::size", "int")
      .typed<int64_t (const Tensor &, int64_t)>();
  RECORD_FUNCTION("size", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
int64_t size_Dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::size", "Dimname")
      .typed<int64_t (const Tensor &, Dimname)>();
  RECORD_FUNCTION("size", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor slice_Tensor(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slice", "Tensor")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("slice", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim, start, end, step);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv3d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv3d_forward_out_output(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d_forward_out", std::vector<c10::IValue>({output, finput, fgrad_input, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
Tensor slow_conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_dilated2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_transpose2d_backward", std::vector<c10::IValue>({grad_output, self, weight, columns, ones}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}
Tensor slow_conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_transpose3d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
std::tuple<Tensor &,Tensor &,Tensor &> slow_conv_transpose3d_backward_out_grad_output(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d_backward", "grad_output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("slow_conv_transpose3d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}
Tensor smm(const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("smm", std::vector<c10::IValue>({self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mat2);
}
Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::smooth_l1_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("smooth_l1_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor & softshrink_out_out(Tensor & out, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink_out", std::vector<c10::IValue>({out, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, lambd);
}
std::tuple<Tensor,Tensor> solve(const Tensor & self, const Tensor & A) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::solve", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("solve", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, A);
}
int64_t sparse_dim(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_dim", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("sparse_dim", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sparse_resize_and_clear_", "")
      .typed<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>();
  RECORD_FUNCTION("sparse_resize_and_clear_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, size, sparse_dim, dense_dim);
}
std::vector<Tensor> split_with_sizes(const Tensor & self, IntArrayRef split_sizes, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::split_with_sizes", "")
      .typed<std::vector<Tensor> (const Tensor &, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("split_with_sizes", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, self, split_sizes, dim);
}
Tensor sqrt(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sqrt", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("sqrt", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sqrt_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sqrt_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("sqrt_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & stack_out_out(Tensor & out, TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stack", "out")
      .typed<Tensor & (Tensor &, TensorList, int64_t)>();
  RECORD_FUNCTION("stack_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, int64_t>(op, c10::DispatchKey::Profiler, out, tensors, dim);
}
std::tuple<Tensor,Tensor> std_mean(const Tensor & self, bool unbiased) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std_mean", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("std_mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, unbiased);
}
std::tuple<Tensor,Tensor> std_mean_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std_mean", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("std_mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
std::tuple<Tensor,Tensor> std_mean_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std_mean", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, DimnameList, bool, bool)>();
  RECORD_FUNCTION("std_mean", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::svd", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("svd", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, some, compute_uv);
}
std::tuple<Tensor &,Tensor &> symeig_out_e(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::symeig", "e")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("symeig_out", std::vector<c10::IValue>({e, V, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, e, V, self, eigenvectors, upper);
}
Tensor tan(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tan", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("tan", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & tan_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tan_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("tan_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor tanh_backward(const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tanh_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("tanh_backward", std::vector<c10::IValue>({grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output);
}
Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out_grad_input(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_backward", "grad_input")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("thnn_conv2d_backward_out", std::vector<c10::IValue>({grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
Tensor thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d_forward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor threshold(const Tensor & self, Scalar threshold, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::threshold", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("threshold", std::vector<c10::IValue>({self, threshold, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, threshold, value);
}
Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::threshold_", "")
      .typed<Tensor & (Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("threshold_", std::vector<c10::IValue>({self, threshold, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, threshold, value);
}
std::tuple<Tensor &,Tensor &> triangular_solve_out_X(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triangular_solve", "X")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("triangular_solve_out", std::vector<c10::IValue>({X, M, self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, X, M, self, A, upper, transpose, unitriangular);
}
Tensor tril(const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tril", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("tril", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, diagonal);
}
Tensor & tril_(Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tril_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("tril_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, diagonal);
}
Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tril_indices", "")
      .typed<Tensor (int64_t, int64_t, int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("tril_indices", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, row, col, offset, options);
}
Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unfold", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("unfold", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dimension, size, step);
}
Tensor & uniform_(Tensor & self, double from, double to, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::uniform_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("uniform_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, from, to, generator);
}
Tensor upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bicubic2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bicubic2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bilinear2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_nearest1d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, scales);
}
Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_trilinear3d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_trilinear3d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}
Tensor & var_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("var_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, out, self, dim, unbiased, keepdim);
}
Tensor & var_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool, bool)>();
  RECORD_FUNCTION("var_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Profiler, out, self, dim, unbiased, keepdim);
}
Tensor view(const Tensor & self, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("view", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, size);
}
Tensor & zero_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::zero_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("zero_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor zeros_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::zeros", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("zeros", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, names, options);
}
Tensor zeros(IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::zeros", "")
      .typed<Tensor (IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("zeros", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, options);
}
}  // namespace
}  // namespace ProfiledType

namespace {

TORCH_LIBRARY_IMPL(aten, Profiler, m) {
  m.impl_UNBOXED("_addmv_impl_", &ProfiledType::_addmv_impl_);
  m.impl_UNBOXED("_batch_norm_impl_index", &ProfiledType::_batch_norm_impl_index);
  m.impl("_ctc_loss", TORCH_FN(ProfiledType::_ctc_loss));
  m.impl_UNBOXED("_cudnn_init_dropout_state", &ProfiledType::_cudnn_init_dropout_state);
  m.impl("_cudnn_rnn_flatten_weight", TORCH_FN(ProfiledType::_cudnn_rnn_flatten_weight));
  m.impl_UNBOXED("_embedding_bag_dense_backward", &ProfiledType::_embedding_bag_dense_backward);
  m.impl("_embedding_bag_per_sample_weights_backward", TORCH_FN(ProfiledType::_embedding_bag_per_sample_weights_backward));
  m.impl_UNBOXED("_empty_per_channel_affine_quantized", &ProfiledType::_empty_per_channel_affine_quantized);
  m.impl("_log_softmax_backward_data", TORCH_FN(ProfiledType::_log_softmax_backward_data));
  m.impl("_lu_solve_helper", TORCH_FN(ProfiledType::_lu_solve_helper));
  m.impl("_make_per_channel_quantized_tensor", TORCH_FN(ProfiledType::_make_per_channel_quantized_tensor));
  m.impl("_mkldnn_reshape", TORCH_FN(ProfiledType::_mkldnn_reshape));
  m.impl("_multinomial_alias_setup", TORCH_FN(ProfiledType::_multinomial_alias_setup));
  m.impl("_pack_padded_sequence_backward", TORCH_FN(ProfiledType::_pack_padded_sequence_backward));
  m.impl("_shape_as_tensor", TORCH_FN(ProfiledType::_shape_as_tensor));
  m.impl_UNBOXED("_sobol_engine_draw", &ProfiledType::_sobol_engine_draw);
  m.impl("_solve_helper", TORCH_FN(ProfiledType::_solve_helper));
  m.impl("_standard_gamma_grad", TORCH_FN(ProfiledType::_standard_gamma_grad));
  m.impl("_svd_helper", TORCH_FN(ProfiledType::_svd_helper));
  m.impl("_unique", TORCH_FN(ProfiledType::_unique));
  m.impl("_unique2", TORCH_FN(ProfiledType::_unique2));
  m.impl("_weight_norm_differentiable_backward", TORCH_FN(ProfiledType::_weight_norm_differentiable_backward));
  m.impl("absolute", TORCH_FN(ProfiledType::absolute));
  m.impl_UNBOXED("absolute_", &ProfiledType::absolute_);
  m.impl("adaptive_avg_pool3d_backward", TORCH_FN(ProfiledType::adaptive_avg_pool3d_backward));
  m.impl("adaptive_max_pool1d", TORCH_FN(ProfiledType::adaptive_max_pool1d));
  m.impl_UNBOXED("adaptive_max_pool2d.out", &ProfiledType::adaptive_max_pool2d_out_out);
  m.impl("addbmm", TORCH_FN(ProfiledType::addbmm));
  m.impl_UNBOXED("addbmm_", &ProfiledType::addbmm_);
  m.impl_UNBOXED("addcdiv.out", &ProfiledType::addcdiv_out_out);
  m.impl_UNBOXED("addmv.out", &ProfiledType::addmv_out_out);
  m.impl("all.dim", TORCH_FN(ProfiledType::all_dim));
  m.impl_UNBOXED("all.dimname", &ProfiledType::all_dimname);
  m.impl("all", TORCH_FN(ProfiledType::all));
  m.impl_UNBOXED("angle.out", &ProfiledType::angle_out_out);
  m.impl("any.dim", TORCH_FN(ProfiledType::any_dim));
  m.impl_UNBOXED("any.dimname", &ProfiledType::any_dimname);
  m.impl("any", TORCH_FN(ProfiledType::any));
  m.impl("argmin", TORCH_FN(ProfiledType::argmin));
  m.impl("as_strided", TORCH_FN(ProfiledType::as_strided));
  m.impl_UNBOXED("as_strided_", &ProfiledType::as_strided_);
  m.impl_UNBOXED("atanh.out", &ProfiledType::atanh_out_out);
  m.impl_UNBOXED("batch_norm", &ProfiledType::batch_norm);
  m.impl_UNBOXED("batch_norm_elemt", &ProfiledType::batch_norm_elemt);
  m.impl_UNBOXED("bilinear", &ProfiledType::bilinear);
  m.impl_UNBOXED("binary_cross_entropy.out", &ProfiledType::binary_cross_entropy_out_out);
  m.impl_UNBOXED("binomial", &ProfiledType::binomial);
  m.impl_UNBOXED("bitwise_or.Tensor_out", &ProfiledType::bitwise_or_out_Tensor_out);
  m.impl_UNBOXED("bitwise_or.Scalar_out", &ProfiledType::bitwise_or_out_Scalar_out);
  m.impl_UNBOXED("bitwise_xor.Tensor_out", &ProfiledType::bitwise_xor_out_Tensor_out);
  m.impl_UNBOXED("bitwise_xor.Scalar_out", &ProfiledType::bitwise_xor_out_Scalar_out);
  m.impl("block_diag", TORCH_FN(ProfiledType::block_diag));
  m.impl_UNBOXED("can_cast", &ProfiledType::can_cast);
  m.impl("ceil", TORCH_FN(ProfiledType::ceil));
  m.impl_UNBOXED("ceil_", &ProfiledType::ceil_);
  m.impl("channel_shuffle", TORCH_FN(ProfiledType::channel_shuffle));
  m.impl_UNBOXED("cholesky_solve.out", &ProfiledType::cholesky_solve_out_out);
  m.impl_UNBOXED("clamp_min.out", &ProfiledType::clamp_min_out_out);
  m.impl("col2im_backward", TORCH_FN(ProfiledType::col2im_backward));
  m.impl("constant_pad_nd", TORCH_FN(ProfiledType::constant_pad_nd));
  m.impl_UNBOXED("conv2d", &ProfiledType::conv2d);
  m.impl_UNBOXED("conv_transpose1d", &ProfiledType::conv_transpose1d);
  m.impl_UNBOXED("copy_", &ProfiledType::copy_);
  m.impl_UNBOXED("copy_sparse_to_sparse_", &ProfiledType::copy_sparse_to_sparse_);
  m.impl_UNBOXED("cosh.out", &ProfiledType::cosh_out_out);
  m.impl_UNBOXED("cross.out", &ProfiledType::cross_out_out);
  m.impl("ctc_loss.IntList", TORCH_FN(ProfiledType::ctc_loss_IntList));
  m.impl("ctc_loss.Tensor", TORCH_FN(ProfiledType::ctc_loss_Tensor));
  m.impl("cudnn_affine_grid_generator_backward", TORCH_FN(ProfiledType::cudnn_affine_grid_generator_backward));
  m.impl("cudnn_convolution_backward", TORCH_FN(ProfiledType::cudnn_convolution_backward));
  m.impl("cudnn_convolution_backward_input", TORCH_FN(ProfiledType::cudnn_convolution_backward_input));
  m.impl_UNBOXED("cudnn_convolution_transpose.deprecated", &ProfiledType::cudnn_convolution_transpose_deprecated);
  m.impl("cudnn_convolution_transpose", TORCH_FN(ProfiledType::cudnn_convolution_transpose));
  m.impl_UNBOXED("cummin.out", &ProfiledType::cummin_out_out);
  m.impl_UNBOXED("cummin.dimname_out", &ProfiledType::cummin_out_dimname_out);
  m.impl("diagflat", TORCH_FN(ProfiledType::diagflat));
  m.impl("div.Tensor", TORCH_FN(ProfiledType::div_Tensor));
  m.impl("div.Scalar", TORCH_FN(ProfiledType::div_Scalar));
  m.impl_UNBOXED("div_.Tensor", &ProfiledType::div__Tensor);
  m.impl_UNBOXED("div_.Scalar", &ProfiledType::div__Scalar);
  m.impl_UNBOXED("eig.e", &ProfiledType::eig_out_e);
  m.impl("embedding_backward", TORCH_FN(ProfiledType::embedding_backward));
  m.impl_UNBOXED("empty.out", &ProfiledType::empty_out_out);
  m.impl_UNBOXED("eq.Scalar_out", &ProfiledType::eq_out_Scalar_out);
  m.impl_UNBOXED("eq.Tensor_out", &ProfiledType::eq_out_Tensor_out);
  m.impl("exp", TORCH_FN(ProfiledType::exp));
  m.impl_UNBOXED("exp_", &ProfiledType::exp_);
  m.impl_UNBOXED("eye", &ProfiledType::eye);
  m.impl_UNBOXED("eye.m", &ProfiledType::eye_m);
  m.impl_UNBOXED("fill_.Scalar", &ProfiledType::fill__Scalar);
  m.impl_UNBOXED("fill_.Tensor", &ProfiledType::fill__Tensor);
  m.impl("fractional_max_pool2d_backward", TORCH_FN(ProfiledType::fractional_max_pool2d_backward));
  m.impl("fractional_max_pool3d", TORCH_FN(ProfiledType::fractional_max_pool3d));
  m.impl_UNBOXED("fractional_max_pool3d_backward.grad_input", &ProfiledType::fractional_max_pool3d_backward_out_grad_input);
  m.impl_UNBOXED("ge.Scalar_out", &ProfiledType::ge_out_Scalar_out);
  m.impl_UNBOXED("ge.Tensor_out", &ProfiledType::ge_out_Tensor_out);
  m.impl("geqrf", TORCH_FN(ProfiledType::geqrf));
  m.impl("ger", TORCH_FN(ProfiledType::ger));
  m.impl("glu", TORCH_FN(ProfiledType::glu));
  m.impl_UNBOXED("glu_backward.grad_input", &ProfiledType::glu_backward_out_grad_input);
  m.impl("grid_sampler", TORCH_FN(ProfiledType::grid_sampler));
  m.impl("grid_sampler_2d", TORCH_FN(ProfiledType::grid_sampler_2d));
  m.impl_UNBOXED("gt.Scalar_out", &ProfiledType::gt_out_Scalar_out);
  m.impl_UNBOXED("gt.Tensor_out", &ProfiledType::gt_out_Tensor_out);
  m.impl("hardsigmoid", TORCH_FN(ProfiledType::hardsigmoid));
  m.impl_UNBOXED("hardsigmoid_", &ProfiledType::hardsigmoid_);
  m.impl("hardswish", TORCH_FN(ProfiledType::hardswish));
  m.impl_UNBOXED("hardswish_", &ProfiledType::hardswish_);
  m.impl("hardtanh_backward", TORCH_FN(ProfiledType::hardtanh_backward));
  m.impl_UNBOXED("histc.out", &ProfiledType::histc_out_out);
  m.impl_UNBOXED("hspmm.out", &ProfiledType::hspmm_out_out);
  m.impl("im2col_backward", TORCH_FN(ProfiledType::im2col_backward));
  m.impl("index_add", TORCH_FN(ProfiledType::index_add));
  m.impl_UNBOXED("index_add.dimname", &ProfiledType::index_add_dimname);
  m.impl_UNBOXED("index_add_", &ProfiledType::index_add_);
  m.impl_UNBOXED("inverse.out", &ProfiledType::inverse_out_out);
  m.impl("is_leaf", TORCH_FN(ProfiledType::is_leaf));
  m.impl("is_pinned", TORCH_FN(ProfiledType::is_pinned));
  m.impl("kl_div", TORCH_FN(ProfiledType::kl_div));
  m.impl("kthvalue", TORCH_FN(ProfiledType::kthvalue));
  m.impl_UNBOXED("kthvalue.dimname", &ProfiledType::kthvalue_dimname);
  m.impl_UNBOXED("le.Scalar_out", &ProfiledType::le_out_Scalar_out);
  m.impl_UNBOXED("le.Tensor_out", &ProfiledType::le_out_Tensor_out);
  m.impl_UNBOXED("leaky_relu.out", &ProfiledType::leaky_relu_out_out);
  m.impl("lgamma", TORCH_FN(ProfiledType::lgamma));
  m.impl_UNBOXED("lgamma_", &ProfiledType::lgamma_);
  m.impl_UNBOXED("log10.out", &ProfiledType::log10_out_out);
  m.impl("log1p", TORCH_FN(ProfiledType::log1p));
  m.impl_UNBOXED("log1p_", &ProfiledType::log1p_);
  m.impl("logical_and", TORCH_FN(ProfiledType::logical_and));
  m.impl_UNBOXED("logical_and_", &ProfiledType::logical_and_);
  m.impl("logical_not", TORCH_FN(ProfiledType::logical_not));
  m.impl_UNBOXED("logical_not_", &ProfiledType::logical_not_);
  m.impl("lstm.input", TORCH_FN(ProfiledType::lstm_input));
  m.impl("lstm.data", TORCH_FN(ProfiledType::lstm_data));
  m.impl_UNBOXED("lt.Scalar_out", &ProfiledType::lt_out_Scalar_out);
  m.impl_UNBOXED("lt.Tensor_out", &ProfiledType::lt_out_Tensor_out);
  m.impl("lu_solve", TORCH_FN(ProfiledType::lu_solve));
  m.impl("margin_ranking_loss", TORCH_FN(ProfiledType::margin_ranking_loss));
  m.impl_UNBOXED("masked_select.out", &ProfiledType::masked_select_out_out);
  m.impl("matrix_power", TORCH_FN(ProfiledType::matrix_power));
  m.impl("max_pool2d", TORCH_FN(ProfiledType::max_pool2d));
  m.impl("max_pool2d_with_indices_backward", TORCH_FN(ProfiledType::max_pool2d_with_indices_backward));
  m.impl("max_pool3d_with_indices", TORCH_FN(ProfiledType::max_pool3d_with_indices));
  m.impl_UNBOXED("max_pool3d_with_indices_backward.grad_input", &ProfiledType::max_pool3d_with_indices_backward_out_grad_input);
  m.impl_UNBOXED("min.dim_min", &ProfiledType::min_out_dim_min);
  m.impl_UNBOXED("min.names_dim_min", &ProfiledType::min_out_names_dim_min);
  m.impl_UNBOXED("min.out", &ProfiledType::min_out_out);
  m.impl("miopen_convolution_backward_weight", TORCH_FN(ProfiledType::miopen_convolution_backward_weight));
  m.impl("miopen_depthwise_convolution_backward", TORCH_FN(ProfiledType::miopen_depthwise_convolution_backward));
  m.impl("miopen_depthwise_convolution_backward_input", TORCH_FN(ProfiledType::miopen_depthwise_convolution_backward_input));
  m.impl_UNBOXED("miopen_rnn_backward", &ProfiledType::miopen_rnn_backward);
  m.impl("mm", TORCH_FN(ProfiledType::mm));
  m.impl_UNBOXED("mse_loss.out", &ProfiledType::mse_loss_out_out);
  m.impl_UNBOXED("mul.out", &ProfiledType::mul_out_out);
  m.impl_UNBOXED("multi_margin_loss_backward", &ProfiledType::multi_margin_loss_backward);
  m.impl_UNBOXED("multilabel_margin_loss.out", &ProfiledType::multilabel_margin_loss_out_out);
  m.impl("narrow_copy", TORCH_FN(ProfiledType::narrow_copy));
  m.impl_UNBOXED("native_group_norm", &ProfiledType::native_group_norm);
  m.impl_UNBOXED("native_layer_norm_backward", &ProfiledType::native_layer_norm_backward);
  m.impl_UNBOXED("neg.out", &ProfiledType::neg_out_out);
  m.impl_UNBOXED("new_zeros", &ProfiledType::new_zeros);
  m.impl("nonzero_numpy", TORCH_FN(ProfiledType::nonzero_numpy));
  m.impl_UNBOXED("nonzero.out", &ProfiledType::nonzero_out_out);
  m.impl_UNBOXED("norm.dtype_out", &ProfiledType::norm_out_dtype_out);
  m.impl_UNBOXED("norm.out", &ProfiledType::norm_out_out);
  m.impl_UNBOXED("norm.names_dtype_out", &ProfiledType::norm_out_names_dtype_out);
  m.impl_UNBOXED("norm.names_out", &ProfiledType::norm_out_names_out);
  m.impl_UNBOXED("normal.Tensor_float_out", &ProfiledType::normal_out_Tensor_float_out);
  m.impl_UNBOXED("normal.float_Tensor_out", &ProfiledType::normal_out_float_Tensor_out);
  m.impl_UNBOXED("normal.Tensor_Tensor_out", &ProfiledType::normal_out_Tensor_Tensor_out);
  m.impl_UNBOXED("normal.float_float_out", &ProfiledType::normal_out_float_float_out);
  m.impl("numpy_T", TORCH_FN(ProfiledType::numpy_T));
  m.impl("one_hot", TORCH_FN(ProfiledType::one_hot));
  m.impl_UNBOXED("ones_like", &ProfiledType::ones_like);
  m.impl("output_nr", TORCH_FN(ProfiledType::output_nr));
  m.impl("pdist", TORCH_FN(ProfiledType::pdist));
  m.impl("prelu_backward", TORCH_FN(ProfiledType::prelu_backward));
  m.impl("q_scale", TORCH_FN(ProfiledType::q_scale));
  m.impl("quantized_rnn_tanh_cell", TORCH_FN(ProfiledType::quantized_rnn_tanh_cell));
  m.impl_UNBOXED("randint_like", &ProfiledType::randint_like);
  m.impl_UNBOXED("randint_like.low_dtype", &ProfiledType::randint_like_low_dtype);
  m.impl_UNBOXED("renorm.out", &ProfiledType::renorm_out_out);
  m.impl("replication_pad2d_backward", TORCH_FN(ProfiledType::replication_pad2d_backward));
  m.impl("replication_pad3d", TORCH_FN(ProfiledType::replication_pad3d));
  m.impl_UNBOXED("replication_pad3d_backward.grad_input", &ProfiledType::replication_pad3d_backward_out_grad_input);
  m.impl_UNBOXED("resize_", &ProfiledType::resize_);
  m.impl("scatter.src", TORCH_FN(ProfiledType::scatter_src));
  m.impl("scatter.value", TORCH_FN(ProfiledType::scatter_value));
  m.impl_UNBOXED("scatter.dimname_src", &ProfiledType::scatter_dimname_src);
  m.impl_UNBOXED("scatter.dimname_value", &ProfiledType::scatter_dimname_value);
  m.impl_UNBOXED("scatter_.src", &ProfiledType::scatter__src);
  m.impl_UNBOXED("scatter_.value", &ProfiledType::scatter__value);
  m.impl("sigmoid", TORCH_FN(ProfiledType::sigmoid));
  m.impl_UNBOXED("sigmoid_", &ProfiledType::sigmoid_);
  m.impl_UNBOXED("sigmoid_backward.grad_input", &ProfiledType::sigmoid_backward_out_grad_input);
  m.impl("sign", TORCH_FN(ProfiledType::sign));
  m.impl_UNBOXED("sign_", &ProfiledType::sign_);
  m.impl_UNBOXED("sinh.out", &ProfiledType::sinh_out_out);
  m.impl("size.int", TORCH_FN(ProfiledType::size_int));
  m.impl_UNBOXED("size.Dimname", &ProfiledType::size_Dimname);
  m.impl("slice.Tensor", TORCH_FN(ProfiledType::slice_Tensor));
  m.impl("slow_conv3d_backward.output_mask", TORCH_FN(ProfiledType::slow_conv3d_backward_output_mask));
  m.impl_UNBOXED("slow_conv3d_forward.output", &ProfiledType::slow_conv3d_forward_out_output);
  m.impl_UNBOXED("slow_conv_dilated2d", &ProfiledType::slow_conv_dilated2d);
  m.impl("slow_conv_transpose2d_backward.output_mask", TORCH_FN(ProfiledType::slow_conv_transpose2d_backward_output_mask));
  m.impl_UNBOXED("slow_conv_transpose3d", &ProfiledType::slow_conv_transpose3d);
  m.impl_UNBOXED("slow_conv_transpose3d_backward.grad_output", &ProfiledType::slow_conv_transpose3d_backward_out_grad_output);
  m.impl("smm", TORCH_FN(ProfiledType::smm));
  m.impl("smooth_l1_loss_backward", TORCH_FN(ProfiledType::smooth_l1_loss_backward));
  m.impl_UNBOXED("softshrink.out", &ProfiledType::softshrink_out_out);
  m.impl("solve", TORCH_FN(ProfiledType::solve));
  m.impl("sparse_dim", TORCH_FN(ProfiledType::sparse_dim));
  m.impl_UNBOXED("sparse_resize_and_clear_", &ProfiledType::sparse_resize_and_clear_);
  m.impl("split_with_sizes", TORCH_FN(ProfiledType::split_with_sizes));
  m.impl("sqrt", TORCH_FN(ProfiledType::sqrt));
  m.impl_UNBOXED("sqrt_", &ProfiledType::sqrt_);
  m.impl_UNBOXED("stack.out", &ProfiledType::stack_out_out);
  m.impl("std_mean", TORCH_FN(ProfiledType::std_mean));
  m.impl("std_mean.dim", TORCH_FN(ProfiledType::std_mean_dim));
  m.impl_UNBOXED("std_mean.names_dim", &ProfiledType::std_mean_names_dim);
  m.impl("svd", TORCH_FN(ProfiledType::svd));
  m.impl_UNBOXED("symeig.e", &ProfiledType::symeig_out_e);
  m.impl("tan", TORCH_FN(ProfiledType::tan));
  m.impl_UNBOXED("tan_", &ProfiledType::tan_);
  m.impl("tanh_backward", TORCH_FN(ProfiledType::tanh_backward));
  m.impl_UNBOXED("thnn_conv2d", &ProfiledType::thnn_conv2d);
  m.impl_UNBOXED("thnn_conv2d_backward.grad_input", &ProfiledType::thnn_conv2d_backward_out_grad_input);
  m.impl_UNBOXED("thnn_conv_depthwise2d_forward", &ProfiledType::thnn_conv_depthwise2d_forward);
  m.impl("threshold", TORCH_FN(ProfiledType::threshold));
  m.impl_UNBOXED("threshold_", &ProfiledType::threshold_);
  m.impl_UNBOXED("triangular_solve.X", &ProfiledType::triangular_solve_out_X);
  m.impl("tril", TORCH_FN(ProfiledType::tril));
  m.impl_UNBOXED("tril_", &ProfiledType::tril_);
  m.impl_UNBOXED("tril_indices", &ProfiledType::tril_indices);
  m.impl("unfold", TORCH_FN(ProfiledType::unfold));
  m.impl_UNBOXED("uniform_", &ProfiledType::uniform_);
  m.impl("upsample_bicubic2d_backward", TORCH_FN(ProfiledType::upsample_bicubic2d_backward));
  m.impl_UNBOXED("upsample_bilinear2d.out", &ProfiledType::upsample_bilinear2d_out_out);
  m.impl_UNBOXED("upsample_nearest1d.out", &ProfiledType::upsample_nearest1d_out_out);
  m.impl("upsample_trilinear3d_backward", TORCH_FN(ProfiledType::upsample_trilinear3d_backward));
  m.impl_UNBOXED("var.out", &ProfiledType::var_out_out);
  m.impl_UNBOXED("var.names_out", &ProfiledType::var_out_names_out);
  m.impl("view", TORCH_FN(ProfiledType::view));
  m.impl_UNBOXED("zero_", &ProfiledType::zero_);
  m.impl_UNBOXED("zeros.names", &ProfiledType::zeros_names);
  m.impl_UNBOXED("zeros", &ProfiledType::zeros);;
}

}  // namespace

} // namespace torch
