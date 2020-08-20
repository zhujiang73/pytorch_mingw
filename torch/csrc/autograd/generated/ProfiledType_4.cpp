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
std::tuple<Tensor,Tensor,Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var_transform, bool train, double eps, std::array<bool,3> output_mask, const Tensor & reservedSpace) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_batch_norm_impl_index_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &)>();
  RECORD_FUNCTION("_batch_norm_impl_index_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, reservedSpace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, int64_t, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, std::array<bool,3>, const Tensor &>(op, c10::DispatchKey::Profiler, impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}
Tensor _cast_Char(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Char", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Char", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Float(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Float", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Float", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cholesky_solve_helper", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_cholesky_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, A, upper);
}
Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_convolution_nogroup", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef)>();
  RECORD_FUNCTION("_convolution_nogroup", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, transposed, output_padding);
}
Tensor _copy_from(const Tensor & self, const Tensor & dst, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_copy_from", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_copy_from", std::vector<c10::IValue>({self, dst}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, dst, non_blocking);
}
Tensor _ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_ctc_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_ctc_loss_backward", std::vector<c10::IValue>({grad, log_probs, targets, neg_log_likelihood, log_alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
}
std::tuple<Tensor,Tensor> _cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_ctc_loss", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("_cudnn_ctc_loss", std::vector<c10::IValue>({log_probs, targets}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
}
void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cufft_set_plan_cache_max_size", "")
      .typed<void (int64_t, int64_t)>();
  RECORD_FUNCTION("_cufft_set_plan_cache_max_size", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, int64_t, int64_t>(op, c10::DispatchKey::Profiler, device_index, max_size);
}
void _cummin_helper(const Tensor & self, Tensor & values, Tensor & indices, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cummin_helper", "")
      .typed<void (const Tensor &, Tensor &, Tensor &, int64_t)>();
  RECORD_FUNCTION("_cummin_helper", std::vector<c10::IValue>({self, values, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<void, const Tensor &, Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, values, indices, dim);
}
Tensor _euclidean_dist(const Tensor & x1, const Tensor & x2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_euclidean_dist", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_euclidean_dist", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, x1, x2);
}
Tensor & _index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_index_copy_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor _inverse_helper(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_inverse_helper", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("_inverse_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor _masked_scale(const Tensor & self, const Tensor & mask, double scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_masked_scale", "")
      .typed<Tensor (const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("_masked_scale", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, mask, scale);
}
bool _nnpack_available() {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_available", "")
      .typed<bool ()>();
  RECORD_FUNCTION("_nnpack_available", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool>(op, c10::DispatchKey::Profiler);
}
Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pdist_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, const Tensor &)>();
  RECORD_FUNCTION("_pdist_backward", std::vector<c10::IValue>({grad, self, pdist}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Profiler, grad, self, p, pdist);
}
Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_s_where", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_s_where", std::vector<c10::IValue>({condition, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, condition, self, other);
}
Tensor _sample_dirichlet(const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sample_dirichlet", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("_sample_dirichlet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, generator);
}
Tensor & _sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sobol_engine_scramble_", "")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_sobol_engine_scramble_", std::vector<c10::IValue>({self, ltm}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, ltm, dimension);
}
Tensor _sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_addmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_sparse_addmm", std::vector<c10::IValue>({self, sparse, dense, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, sparse, dense, beta, alpha);
}
Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_unsafe", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("_sparse_coo_tensor_unsafe", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, indices, values, size, options);
}
Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims_and_tensors", "")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &)>();
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims_and_tensors", std::vector<c10::IValue>({indices, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const Tensor &, const Tensor &, const TensorOptions &>(op, c10::DispatchKey::Profiler, sparse_dim, dense_dim, size, indices, values, options);
}
Tensor _sparse_softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_softmax", "int")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("_sparse_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor _sparse_softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_softmax", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("_sparse_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor _sparse_softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_sparse_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, half_to_float);
}
Tensor _standard_gamma(const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_standard_gamma", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("_standard_gamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, generator);
}
std::tuple<Tensor,Tensor> _symeig_helper(const Tensor & self, bool eigenvectors, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_symeig_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_symeig_helper", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, eigenvectors, upper);
}
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_thnn_fused_lstm_cell", std::vector<c10::IValue>({input_gates, hidden_gates, cx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input_gates, hidden_gates, cx, input_bias, hidden_bias);
}
std::tuple<Tensor,Tensor> _triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_triangular_solve_helper", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("_triangular_solve_helper", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, A, upper, transpose, unitriangular);
}
int64_t _version(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_version", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_version", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & abs_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("abs_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor & adaptive_max_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices);
}
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool3d_out", std::vector<c10::IValue>({out, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, indices, self, output_size);
}
Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcdiv", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcdiv", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, tensor1, tensor2, value);
}
Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcdiv_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcdiv_", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, tensor1, tensor2, value);
}
Tensor & addcmul_out_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcmul_out", std::vector<c10::IValue>({out, self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, tensor1, tensor2, value);
}
Tensor & addmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmm_out", std::vector<c10::IValue>({out, self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, mat1, mat2, beta, alpha);
}
Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmv", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmv", std::vector<c10::IValue>({self, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, mat, vec, beta, alpha);
}
Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmv_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmv_", std::vector<c10::IValue>({self, mat, vec, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, mat, vec, beta, alpha);
}
Tensor alpha_dropout(const Tensor & input, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::alpha_dropout", "")
      .typed<Tensor (const Tensor &, double, bool)>();
  RECORD_FUNCTION("alpha_dropout", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, p, train);
}
Tensor & alpha_dropout_(Tensor & self, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::alpha_dropout_", "")
      .typed<Tensor & (Tensor &, double, bool)>();
  RECORD_FUNCTION("alpha_dropout_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, bool>(op, c10::DispatchKey::Profiler, self, p, train);
}
Tensor angle(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::angle", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("angle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & atan2_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atan2_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & atan_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atan_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor atanh(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atanh", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("atanh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & atanh_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atanh_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("atanh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("avg_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
Tensor & avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor batch_norm_backward_elemt(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_backward_elemt", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("batch_norm_backward_elemt", std::vector<c10::IValue>({grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
}
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, const Tensor & counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_gather_stats_with_counts", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &)>();
  RECORD_FUNCTION("batch_norm_gather_stats_with_counts", std::vector<c10::IValue>({input, mean, invstd, running_mean, running_var, counts}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, const Tensor &>(op, c10::DispatchKey::Profiler, input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction);
}
Tensor & binary_cross_entropy_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, weight, reduction);
}
Tensor bitwise_or_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_or", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor bitwise_or_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_or", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & bitwise_or__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_or_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & bitwise_or__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_or_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_or_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor bitwise_xor_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_xor", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor bitwise_xor_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_xor", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & bitwise_xor__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_xor_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & bitwise_xor__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_xor_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_xor_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor blackman_window(int64_t window_length, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::blackman_window", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("blackman_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, options);
}
Tensor blackman_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::blackman_window", "periodic")
      .typed<Tensor (int64_t, bool, const TensorOptions &)>();
  RECORD_FUNCTION("blackman_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, periodic, options);
}
std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::broadcast_tensors", "")
      .typed<std::vector<Tensor> (TensorList)>();
  RECORD_FUNCTION("broadcast_tensors", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
Tensor & cholesky_inverse_out_out(Tensor & out, const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_inverse", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_inverse_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, upper);
}
Tensor cholesky_solve(const Tensor & self, const Tensor & input2, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_solve", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_solve", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, input2, upper);
}
Tensor clamp_min(const Tensor & self, Scalar min) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_min", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_min", std::vector<c10::IValue>({self, min}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, min);
}
Tensor & clamp_min_(Tensor & self, Scalar min) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_min_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_min_", std::vector<c10::IValue>({self, min}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, min);
}
Tensor & clamp_out_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>();
  RECORD_FUNCTION("clamp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, c10::DispatchKey::Profiler, out, self, min, max);
}
Tensor clone(const Tensor & self, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clone", "")
      .typed<Tensor (const Tensor &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("clone", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, memory_format);
}
Tensor contiguous(const Tensor & self, MemoryFormat memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::contiguous", "")
      .typed<Tensor (const Tensor &, MemoryFormat)>();
  RECORD_FUNCTION("contiguous", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, MemoryFormat>(op, c10::DispatchKey::Profiler, self, memory_format);
}
Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("conv3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, groups);
}
Tensor conv_transpose2d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose2d", "input")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("conv_transpose2d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, output_padding, groups, dilation);
}
Tensor convolution_overrideable(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution_overrideable", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("convolution_overrideable", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
Tensor & cos_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cos", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cos_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor cosh(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosh", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("cosh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & cosh_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cosh_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("cosh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cross", "")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>();
  RECORD_FUNCTION("cross", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, other, dim);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("cudnn_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
std::tuple<Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,2> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,2>)>();
  RECORD_FUNCTION("cudnn_convolution_transpose_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,2>>(op, c10::DispatchKey::Profiler, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_convolution_transpose_backward_input", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("cudnn_convolution_transpose_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_grid_sampler", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cudnn_grid_sampler", std::vector<c10::IValue>({self, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, grid);
}
bool cudnn_is_acceptable(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_is_acceptable", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("cudnn_is_acceptable", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> cummin(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummin", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t)>();
  RECORD_FUNCTION("cummin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
std::tuple<Tensor,Tensor> cummin_dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummin", "dimname")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname)>();
  RECORD_FUNCTION("cummin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & deg2rad_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("deg2rad_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diag_embed", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("diag_embed", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, offset, dim1, dim2);
}
Tensor & diag_out_out(Tensor & out, const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diag", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("diag_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, diagonal);
}
Tensor & digamma_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("digamma_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eig", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool)>();
  RECORD_FUNCTION("eig", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, eigenvectors);
}
Tensor empty_names(IntArrayRef size, c10::optional<DimnameList> names, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty", "names")
      .typed<Tensor (IntArrayRef, c10::optional<DimnameList>, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, c10::optional<DimnameList>, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, names, options, memory_format);
}
Tensor empty_memory_format(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty", "memory_format")
      .typed<Tensor (IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, size, options, memory_format);
}
Tensor eq_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("eq", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor eq_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("eq", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & eq__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("eq_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & eq__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::eq_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("eq_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & erfinv_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("erfinv_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_tensor_affine", "")
      .typed<Tensor (const Tensor &, double, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_tensor_affine", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, scale, zero_point, quant_min, quant_max);
}
Tensor fbgemm_pack_quantized_matrix(const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_pack_quantized_matrix", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("fbgemm_pack_quantized_matrix", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, input);
}
Tensor fbgemm_pack_quantized_matrix_KN(const Tensor & input, int64_t K, int64_t N) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fbgemm_pack_quantized_matrix", "KN")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("fbgemm_pack_quantized_matrix", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, input, K, N);
}
Tensor feature_dropout(const Tensor & input, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::feature_dropout", "")
      .typed<Tensor (const Tensor &, double, bool)>();
  RECORD_FUNCTION("feature_dropout", std::vector<c10::IValue>({input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, p, train);
}
Tensor & feature_dropout_(Tensor & self, double p, bool train) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::feature_dropout_", "")
      .typed<Tensor & (Tensor &, double, bool)>();
  RECORD_FUNCTION("feature_dropout_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, bool>(op, c10::DispatchKey::Profiler, self, p, train);
}
Tensor & fill_diagonal_(Tensor & self, Scalar fill_value, bool wrap) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fill_diagonal_", "")
      .typed<Tensor & (Tensor &, Scalar, bool)>();
  RECORD_FUNCTION("fill_diagonal_", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, bool>(op, c10::DispatchKey::Profiler, self, fill_value, wrap);
}
Tensor fliplr(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fliplr", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("fliplr", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor flipud(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flipud", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("flipud", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & floor_divide_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("floor_divide_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fractional_max_pool3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("fractional_max_pool3d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, output_size, indices);
}
Tensor & full_out_out(Tensor & out, IntArrayRef size, Scalar fill_value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "out")
      .typed<Tensor & (Tensor &, IntArrayRef, Scalar)>();
  RECORD_FUNCTION("full_out", std::vector<c10::IValue>({out, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, Scalar>(op, c10::DispatchKey::Profiler, out, size, fill_value);
}
Tensor & gather_out_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, const Tensor &, bool)>();
  RECORD_FUNCTION("gather_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, dim, index, sparse_grad);
}
Tensor & gather_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, const Tensor &, bool)>();
  RECORD_FUNCTION("gather_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, dim, index, sparse_grad);
}
Tensor ge_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("ge", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor ge_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ge", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & ge__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("ge_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & ge__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ge_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ge_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor gelu(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gelu", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("gelu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & geometric_(Tensor & self, double p, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::geometric_", "")
      .typed<Tensor & (Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("geometric_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, p, generator);
}
Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::glu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("glu_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, dim);
}
std::tuple<Tensor,Tensor> grid_sampler_2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_2d_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("grid_sampler_2d_backward", std::vector<c10::IValue>({grad_output, input, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}
Tensor grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("grid_sampler_3d", std::vector<c10::IValue>({input, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, input, grid, interpolation_mode, padding_mode, align_corners);
}
std::tuple<Tensor,Tensor> gru_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gru", "input")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>();
  RECORD_FUNCTION("gru", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool>(op, c10::DispatchKey::Profiler, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> gru_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gru", "data")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>();
  RECORD_FUNCTION("gru", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool>(op, c10::DispatchKey::Profiler, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
Tensor gt_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("gt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor gt_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & gt__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("gt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & gt__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gt_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor hardsigmoid_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardsigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardsigmoid_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
Tensor hardswish_backward(const Tensor & grad_output, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardswish_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hardswish_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self);
}
Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hinge_embedding_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, int64_t)>();
  RECORD_FUNCTION("hinge_embedding_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, int64_t>(op, c10::DispatchKey::Profiler, self, target, margin, reduction);
}
Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::histc", "")
      .typed<Tensor (const Tensor &, int64_t, Scalar, Scalar)>();
  RECORD_FUNCTION("histc", std::vector<c10::IValue>({self, min, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, bins, min, max);
}
Tensor hspmm(const Tensor & mat1, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hspmm", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("hspmm", std::vector<c10::IValue>({mat1, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, mat1, mat2);
}
Tensor imag(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::imag", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("imag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_copy", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor index_copy_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_copy", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor & index_copy__dimname(Tensor & self, Dimname dim, const Tensor & index, const Tensor & source) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_copy_", "dimname")
      .typed<Tensor & (Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_copy_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, source);
}
Tensor index_fill_int_Scalar(const Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "int_Scalar")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>();
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor index_fill_int_Tensor(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "int_Tensor")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor index_fill_Dimname_Scalar(const Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "Dimname_Scalar")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, Scalar)>();
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor index_fill_Dimname_Tensor(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill", "Dimname_Tensor")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_fill", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor & index_fill__int_Scalar(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "int_Scalar")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>();
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor & index_fill__int_Tensor(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "int_Tensor")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor & index_fill__Dimname_Scalar(Tensor & self, Dimname dim, const Tensor & index, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "Dimname_Scalar")
      .typed<Tensor & (Tensor &, Dimname, const Tensor &, Scalar)>();
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor & index_fill__Dimname_Tensor(Tensor & self, Dimname dim, const Tensor & index, const Tensor & value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_fill_", "Dimname_Tensor")
      .typed<Tensor & (Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("index_fill_", std::vector<c10::IValue>({self, index, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, value);
}
Tensor inverse(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::inverse", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::irfft", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool, IntArrayRef)>();
  RECORD_FUNCTION("irfft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized, onesided, signal_sizes);
}
bool is_nonzero(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_nonzero", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_nonzero", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_set_to(const Tensor & self, const Tensor & tensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_set_to", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("is_set_to", std::vector<c10::IValue>({self, tensor}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, tensor);
}
bool is_signed(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_signed", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_signed", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::isclose", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, double, bool)>();
  RECORD_FUNCTION("isclose", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, double, bool>(op, c10::DispatchKey::Profiler, self, other, rtol, atol, equal_nan);
}
Tensor isfinite(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::isfinite", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("isfinite", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor istft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool center, bool normalized, bool onesided, c10::optional<int64_t> length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::istft", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("istft", std::vector<c10::IValue>({self, window}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length);
}
Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, bool log_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::kl_div_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("kl_div_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction, log_target);
}
Tensor le_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("le", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor le_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("le", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & le__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("le_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & le__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::le_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("le_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("leaky_relu", std::vector<c10::IValue>({self, negative_slope}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, negative_slope);
}
Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("leaky_relu_", std::vector<c10::IValue>({self, negative_slope}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, negative_slope);
}
Tensor & lerp_out_Scalar_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("lerp_out", std::vector<c10::IValue>({out, self, end, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, end, weight);
}
Tensor & lerp_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lerp_out", std::vector<c10::IValue>({out, self, end, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, end, weight);
}
Tensor log10(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log10", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log10", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log10_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log10_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("log10_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log_sigmoid_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor logdet(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logdet", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("logdet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor lt_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("lt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor lt_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lt", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & lt__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("lt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & lt__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lt_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lt_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor masked_select(const Tensor & self, const Tensor & mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::masked_select", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("masked_select", std::vector<c10::IValue>({self, mask}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, mask);
}
Tensor matrix_rank_tol(const Tensor & self, double tol, bool symmetric) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matrix_rank", "tol")
      .typed<Tensor (const Tensor &, double, bool)>();
  RECORD_FUNCTION("matrix_rank", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, self, tol, symmetric);
}
Tensor matrix_rank(const Tensor & self, bool symmetric) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matrix_rank", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("matrix_rank", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, symmetric);
}
Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_pool3d_with_indices_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &)>();
  RECORD_FUNCTION("max_pool3d_with_indices_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
Tensor & max_unpool2d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, indices, output_size);
}
std::tuple<Tensor,Tensor> min_dim(const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor,Tensor> min_names_dim(const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "names_dim")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor min_other(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "other")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor min(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("min", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose_backward_weight", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_transpose_backward_weight", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
std::tuple<Tensor,Tensor> mkldnn_convolution_backward_weights(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution_backward_weights", "")
      .typed<std::tuple<Tensor,Tensor> (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool)>();
  RECORD_FUNCTION("mkldnn_convolution_backward_weights", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool>(op, c10::DispatchKey::Profiler, weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}
Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_linear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mkldnn_linear", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, bias);
}
Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor & mse_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction);
}
Tensor mul_Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor mul_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("mul", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & mul__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("mul_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & mul__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mul_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("mul_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss", std::vector<c10::IValue>({self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, target, reduction);
}
Tensor & multilabel_margin_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("multilabel_margin_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, reduction, is_target);
}
Tensor & multinomial_out_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multinomial", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("multinomial_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, self, num_samples, replacement, generator);
}
std::tuple<Tensor &,Tensor &,Tensor &> native_batch_norm_out_out(Tensor & out, Tensor & save_mean, Tensor & save_invstd, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm", "out")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("native_batch_norm_out", std::vector<c10::IValue>({out, save_mean, save_invstd, input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
}
std::tuple<Tensor,Tensor,Tensor> native_group_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_group_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("native_group_norm_backward", std::vector<c10::IValue>({grad_out, input, mean, rstd, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}
Tensor neg(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::neg", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("neg", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & neg_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::neg_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("neg_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor new_empty(const Tensor & self, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_empty", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("new_empty", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, self, size, options);
}
Tensor & nll_loss2d_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, weight, reduction, ignore_index);
}
Tensor & nll_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss_out", std::vector<c10::IValue>({out, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, weight, reduction, ignore_index);
}
Tensor nonzero(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nonzero", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("nonzero", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor norm_ScalarOpt_dtype(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "ScalarOpt_dtype")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>();
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, ScalarType>(op, c10::DispatchKey::Profiler, self, p, dtype);
}
Tensor norm_Scalar(const Tensor & self, Scalar p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self, p}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, p);
}
Tensor norm_ScalarOpt_dim_dtype(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "ScalarOpt_dim_dtype")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>();
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType>(op, c10::DispatchKey::Profiler, self, p, dim, keepdim, dtype);
}
Tensor norm_ScalarOpt_dim(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "ScalarOpt_dim")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>();
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, p, dim, keepdim);
}
Tensor norm_names_ScalarOpt_dim_dtype(const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "names_ScalarOpt_dim_dtype")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType)>();
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType>(op, c10::DispatchKey::Profiler, self, p, dim, keepdim, dtype);
}
Tensor norm_names_ScalarOpt_dim(const Tensor & self, c10::optional<Scalar> p, DimnameList dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::norm", "names_ScalarOpt_dim")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool)>();
  RECORD_FUNCTION("norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, DimnameList, bool>(op, c10::DispatchKey::Profiler, self, p, dim, keepdim);
}
Tensor normal_Tensor_float(const Tensor & mean, double std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "Tensor_float")
      .typed<Tensor (const Tensor &, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({mean}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, mean, std, generator);
}
Tensor normal_float_Tensor(double mean, const Tensor & std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "float_Tensor")
      .typed<Tensor (double, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({std}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, double, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, mean, std, generator);
}
Tensor normal_Tensor_Tensor(const Tensor & mean, const Tensor & std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "Tensor_Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({mean, std}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, mean, std, generator);
}
Tensor normal_float_float(double mean, double std, IntArrayRef size, c10::optional<Generator> generator, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal", "float_float")
      .typed<Tensor (double, double, IntArrayRef, c10::optional<Generator>, const TensorOptions &)>();
  RECORD_FUNCTION("normal", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, double, double, IntArrayRef, c10::optional<Generator>, const TensorOptions &>(op, c10::DispatchKey::Profiler, mean, std, size, generator, options);
}
Tensor & normal_(Tensor & self, double mean, double std, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::normal_", "")
      .typed<Tensor & (Tensor &, double, double, c10::optional<Generator>)>();
  RECORD_FUNCTION("normal_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, double, double, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, mean, std, generator);
}
Tensor & nuclear_norm_out_out(Tensor & out, const Tensor & self, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("nuclear_norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, keepdim);
}
Tensor & nuclear_norm_out_dim_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "dim_out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("nuclear_norm_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor & orgqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::orgqr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("orgqr_out", std::vector<c10::IValue>({out, self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, input2);
}
Tensor poisson(const Tensor & self, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::poisson", "")
      .typed<Tensor (const Tensor &, c10::optional<Generator>)>();
  RECORD_FUNCTION("poisson", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, generator);
}
Tensor poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::poisson_nll_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, bool, bool, double, int64_t)>();
  RECORD_FUNCTION("poisson_nll_loss", std::vector<c10::IValue>({input, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, bool, bool, double, int64_t>(op, c10::DispatchKey::Profiler, input, target, log_input, full, eps, reduction);
}
ScalarType promote_types(ScalarType type1, ScalarType type2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::promote_types", "")
      .typed<ScalarType (ScalarType, ScalarType)>();
  RECORD_FUNCTION("promote_types", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<ScalarType, ScalarType, ScalarType>(op, c10::DispatchKey::Profiler, type1, type2);
}
Tensor q_per_channel_scales(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_per_channel_scales", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("q_per_channel_scales", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor quantized_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_batch_norm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t)>();
  RECORD_FUNCTION("quantized_batch_norm", std::vector<c10::IValue>({input, weight, bias, mean, var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, double, int64_t>(op, c10::DispatchKey::Profiler, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_max_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool)>();
  RECORD_FUNCTION("quantized_max_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, dilation, ceil_mode);
}
Tensor & rad2deg_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rad2deg_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & range_out_out(Tensor & out, Scalar start, Scalar end, Scalar step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("range_out", std::vector<c10::IValue>({out, start, end, step}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, start, end, step);
}
Tensor & reciprocal_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("reciprocal_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & reflection_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::renorm", "")
      .typed<Tensor (const Tensor &, Scalar, int64_t, Scalar)>();
  RECORD_FUNCTION("renorm", std::vector<c10::IValue>({self, p, maxnorm}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, int64_t, Scalar>(op, c10::DispatchKey::Profiler, self, p, dim, maxnorm);
}
Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::renorm_", "")
      .typed<Tensor & (Tensor &, Scalar, int64_t, Scalar)>();
  RECORD_FUNCTION("renorm_", std::vector<c10::IValue>({self, p, maxnorm}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, int64_t, Scalar>(op, c10::DispatchKey::Profiler, self, p, dim, maxnorm);
}
Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad3d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad3d_backward", std::vector<c10::IValue>({grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_output, self, padding);
}
Tensor & requires_grad_(Tensor & self, bool requires_grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::requires_grad_", "")
      .typed<Tensor & (Tensor &, bool)>();
  RECORD_FUNCTION("requires_grad_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, bool>(op, c10::DispatchKey::Profiler, self, requires_grad);
}
Tensor reshape(const Tensor & self, IntArrayRef shape) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reshape", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reshape", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, shape);
}
Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rfft", "")
      .typed<Tensor (const Tensor &, int64_t, bool, bool)>();
  RECORD_FUNCTION("rfft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized, onesided);
}
std::tuple<Tensor,Tensor> rnn_relu_input(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_relu", "input")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool)>();
  RECORD_FUNCTION("rnn_relu", std::vector<c10::IValue>({input, hx}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool, bool>(op, c10::DispatchKey::Profiler, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
std::tuple<Tensor,Tensor> rnn_relu_data(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_relu", "data")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool)>();
  RECORD_FUNCTION("rnn_relu", std::vector<c10::IValue>({data, batch_sizes, hx}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, TensorList, bool, int64_t, double, bool, bool>(op, c10::DispatchKey::Profiler, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}
Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu", "")
      .typed<Tensor (const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("rrelu", std::vector<c10::IValue>({self, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, lower, upper, training, generator);
}
Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_", "")
      .typed<Tensor & (Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("rrelu_", std::vector<c10::IValue>({self, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, lower, upper, training, generator);
}
Tensor & rrelu_with_noise_out_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("rrelu_with_noise_out", std::vector<c10::IValue>({out, self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, self, noise, lower, upper, training, generator);
}
Tensor rsub_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsub", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("rsub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor rsub_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rsub", "Scalar")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("rsub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sigmoid_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sigmoid_backward", std::vector<c10::IValue>({grad_output, output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output);
}
Tensor & sin_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sin", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("sin_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor sinh(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sinh", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("sinh", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sinh_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sinh_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("sinh_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> slogdet(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slogdet", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &)>();
  RECORD_FUNCTION("slogdet", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv3d_forward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv3d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated2d_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_dilated2d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
Tensor slow_conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated3d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("slow_conv_dilated3d", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding, dilation);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_transpose3d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_transpose3d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_transpose3d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}
Tensor softshrink(const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink", std::vector<c10::IValue>({self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, lambd);
}
Tensor & softshrink_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, lambd);
}
Tensor stack(TensorList tensors, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stack", "")
      .typed<Tensor (TensorList, int64_t)>();
  RECORD_FUNCTION("stack", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, TensorList, int64_t>(op, c10::DispatchKey::Profiler, tensors, dim);
}
Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::stft", "")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("stft", std::vector<c10::IValue>({self, window}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, n_fft, hop_length, win_length, window, normalized, onesided);
}
Tensor & sub_out_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("sub_out", std::vector<c10::IValue>({out, self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other, alpha);
}
std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::symeig", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>();
  RECORD_FUNCTION("symeig", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, eigenvectors, upper);
}
Tensor tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::tensordot", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("tensordot", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, other, dims_self, dims_other);
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward_output_mask(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_backward", "output_mask")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>)>();
  RECORD_FUNCTION("thnn_conv2d_backward", std::vector<c10::IValue>({grad_output, self, weight, finput, fgrad_input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, const Tensor &, const Tensor &, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out_output(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &,Tensor &> (Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d_forward_out", std::vector<c10::IValue>({output, finput, fgrad_input, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &,Tensor &>, Tensor &, Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::threshold_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("threshold_backward", std::vector<c10::IValue>({grad_output, self, threshold}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_output, self, threshold);
}
Tensor to_dense(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_dense", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("to_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor> triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triangular_solve", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>();
  RECORD_FUNCTION("triangular_solve", std::vector<c10::IValue>({self, A}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, c10::DispatchKey::Profiler, self, A, upper, transpose, unitriangular);
}
Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::triplet_margin_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t)>();
  RECORD_FUNCTION("triplet_margin_loss", std::vector<c10::IValue>({anchor, positive, negative}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, double, double, double, bool, int64_t>(op, c10::DispatchKey::Profiler, anchor, positive, negative, margin, p, eps, swap, reduction);
}
Tensor unfold_backward(const Tensor & grad_in, IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unfold_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("unfold_backward", std::vector<c10::IValue>({grad_in}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad_in, input_sizes, dim, size, step);
}
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_dim_consecutive", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, int64_t, bool, bool)>();
  RECORD_FUNCTION("unique_dim_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, return_inverse, return_counts);
}
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_bilinear2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor & upsample_linear1d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, align_corners, scales);
}
Tensor upsample_nearest1d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, scales);
}
Tensor & upsample_nearest1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, scales);
}
Tensor & upsample_nearest2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, scales_h, scales_w);
}
Tensor var(const Tensor & self, bool unbiased) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, unbiased);
}
Tensor var_dim(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
Tensor var_names_dim(const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::var", "names_dim")
      .typed<Tensor (const Tensor &, DimnameList, bool, bool)>();
  RECORD_FUNCTION("var", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Profiler, self, dim, unbiased, keepdim);
}
Tensor where_self(const Tensor & condition, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::where", "self")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("where", std::vector<c10::IValue>({condition, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, condition, self, other);
}
std::vector<Tensor> where(const Tensor & condition) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::where", "")
      .typed<std::vector<Tensor> (const Tensor &)>();
  RECORD_FUNCTION("where", std::vector<c10::IValue>({condition}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &>(op, c10::DispatchKey::Profiler, condition);
}
Tensor zeros_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::zeros_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("zeros_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
}  // namespace
}  // namespace ProfiledType

namespace {

TORCH_LIBRARY_IMPL(aten, Profiler, m) {
  m.impl_UNBOXED("_batch_norm_impl_index_backward", &ProfiledType::_batch_norm_impl_index_backward);
  m.impl("_cast_Char", TORCH_FN(ProfiledType::_cast_Char));
  m.impl("_cast_Float", TORCH_FN(ProfiledType::_cast_Float));
  m.impl("_cholesky_solve_helper", TORCH_FN(ProfiledType::_cholesky_solve_helper));
  m.impl_UNBOXED("_convolution_nogroup", &ProfiledType::_convolution_nogroup);
  m.impl("_copy_from", TORCH_FN(ProfiledType::_copy_from));
  m.impl("_ctc_loss_backward", TORCH_FN(ProfiledType::_ctc_loss_backward));
  m.impl("_cudnn_ctc_loss", TORCH_FN(ProfiledType::_cudnn_ctc_loss));
  m.impl("_cufft_set_plan_cache_max_size", TORCH_FN(ProfiledType::_cufft_set_plan_cache_max_size));
  m.impl_UNBOXED("_cummin_helper", &ProfiledType::_cummin_helper);
  m.impl("_euclidean_dist", TORCH_FN(ProfiledType::_euclidean_dist));
  m.impl_UNBOXED("_index_copy_", &ProfiledType::_index_copy_);
  m.impl("_inverse_helper", TORCH_FN(ProfiledType::_inverse_helper));
  m.impl("_masked_scale", TORCH_FN(ProfiledType::_masked_scale));
  m.impl("_nnpack_available", TORCH_FN(ProfiledType::_nnpack_available));
  m.impl("_pdist_backward", TORCH_FN(ProfiledType::_pdist_backward));
  m.impl("_s_where", TORCH_FN(ProfiledType::_s_where));
  m.impl_UNBOXED("_sample_dirichlet", &ProfiledType::_sample_dirichlet);
  m.impl_UNBOXED("_sobol_engine_scramble_", &ProfiledType::_sobol_engine_scramble_);
  m.impl("_sparse_addmm", TORCH_FN(ProfiledType::_sparse_addmm));
  m.impl_UNBOXED("_sparse_coo_tensor_unsafe", &ProfiledType::_sparse_coo_tensor_unsafe);
  m.impl_UNBOXED("_sparse_coo_tensor_with_dims_and_tensors", &ProfiledType::_sparse_coo_tensor_with_dims_and_tensors);
  m.impl_UNBOXED("_sparse_softmax.int", &ProfiledType::_sparse_softmax_int);
  m.impl_UNBOXED("_sparse_softmax.Dimname", &ProfiledType::_sparse_softmax_Dimname);
  m.impl("_sparse_softmax", TORCH_FN(ProfiledType::_sparse_softmax));
  m.impl_UNBOXED("_standard_gamma", &ProfiledType::_standard_gamma);
  m.impl("_symeig_helper", TORCH_FN(ProfiledType::_symeig_helper));
  m.impl_UNBOXED("_thnn_fused_lstm_cell", &ProfiledType::_thnn_fused_lstm_cell);
  m.impl("_triangular_solve_helper", TORCH_FN(ProfiledType::_triangular_solve_helper));
  m.impl("_version", TORCH_FN(ProfiledType::_version));
  m.impl_UNBOXED("abs.out", &ProfiledType::abs_out_out);
  m.impl("adaptive_max_pool2d", TORCH_FN(ProfiledType::adaptive_max_pool2d));
  m.impl_UNBOXED("adaptive_max_pool2d_backward.grad_input", &ProfiledType::adaptive_max_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("adaptive_max_pool3d.out", &ProfiledType::adaptive_max_pool3d_out_out);
  m.impl("addcdiv", TORCH_FN(ProfiledType::addcdiv));
  m.impl_UNBOXED("addcdiv_", &ProfiledType::addcdiv_);
  m.impl_UNBOXED("addcmul.out", &ProfiledType::addcmul_out_out);
  m.impl_UNBOXED("addmm.out", &ProfiledType::addmm_out_out);
  m.impl("addmv", TORCH_FN(ProfiledType::addmv));
  m.impl_UNBOXED("addmv_", &ProfiledType::addmv_);
  m.impl("alpha_dropout", TORCH_FN(ProfiledType::alpha_dropout));
  m.impl_UNBOXED("alpha_dropout_", &ProfiledType::alpha_dropout_);
  m.impl("angle", TORCH_FN(ProfiledType::angle));
  m.impl_UNBOXED("atan2.out", &ProfiledType::atan2_out_out);
  m.impl_UNBOXED("atan.out", &ProfiledType::atan_out_out);
  m.impl("atanh", TORCH_FN(ProfiledType::atanh));
  m.impl_UNBOXED("atanh_", &ProfiledType::atanh_);
  m.impl("avg_pool1d", TORCH_FN(ProfiledType::avg_pool1d));
  m.impl_UNBOXED("avg_pool2d.out", &ProfiledType::avg_pool2d_out_out);
  m.impl_UNBOXED("batch_norm_backward_elemt", &ProfiledType::batch_norm_backward_elemt);
  m.impl_UNBOXED("batch_norm_gather_stats_with_counts", &ProfiledType::batch_norm_gather_stats_with_counts);
  m.impl_UNBOXED("binary_cross_entropy", &ProfiledType::binary_cross_entropy);
  m.impl_UNBOXED("binary_cross_entropy_backward.grad_input", &ProfiledType::binary_cross_entropy_backward_out_grad_input);
  m.impl("bitwise_or.Scalar", TORCH_FN(ProfiledType::bitwise_or_Scalar));
  m.impl("bitwise_or.Tensor", TORCH_FN(ProfiledType::bitwise_or_Tensor));
  m.impl_UNBOXED("bitwise_or_.Scalar", &ProfiledType::bitwise_or__Scalar);
  m.impl_UNBOXED("bitwise_or_.Tensor", &ProfiledType::bitwise_or__Tensor);
  m.impl("bitwise_xor.Scalar", TORCH_FN(ProfiledType::bitwise_xor_Scalar));
  m.impl("bitwise_xor.Tensor", TORCH_FN(ProfiledType::bitwise_xor_Tensor));
  m.impl_UNBOXED("bitwise_xor_.Scalar", &ProfiledType::bitwise_xor__Scalar);
  m.impl_UNBOXED("bitwise_xor_.Tensor", &ProfiledType::bitwise_xor__Tensor);
  m.impl_UNBOXED("blackman_window", &ProfiledType::blackman_window);
  m.impl_UNBOXED("blackman_window.periodic", &ProfiledType::blackman_window_periodic);
  m.impl("broadcast_tensors", TORCH_FN(ProfiledType::broadcast_tensors));
  m.impl_UNBOXED("cholesky_inverse.out", &ProfiledType::cholesky_inverse_out_out);
  m.impl("cholesky_solve", TORCH_FN(ProfiledType::cholesky_solve));
  m.impl("clamp_min", TORCH_FN(ProfiledType::clamp_min));
  m.impl_UNBOXED("clamp_min_", &ProfiledType::clamp_min_);
  m.impl_UNBOXED("clamp.out", &ProfiledType::clamp_out_out);
  m.impl_UNBOXED("clone", &ProfiledType::clone);
  m.impl_UNBOXED("contiguous", &ProfiledType::contiguous);
  m.impl_UNBOXED("conv3d", &ProfiledType::conv3d);
  m.impl_UNBOXED("conv_transpose2d.input", &ProfiledType::conv_transpose2d_input);
  m.impl_UNBOXED("convolution_overrideable", &ProfiledType::convolution_overrideable);
  m.impl_UNBOXED("cos.out", &ProfiledType::cos_out_out);
  m.impl("cosh", TORCH_FN(ProfiledType::cosh));
  m.impl_UNBOXED("cosh_", &ProfiledType::cosh_);
  m.impl("cross", TORCH_FN(ProfiledType::cross));
  m.impl_UNBOXED("cudnn_batch_norm", &ProfiledType::cudnn_batch_norm);
  m.impl("cudnn_convolution_transpose_backward", TORCH_FN(ProfiledType::cudnn_convolution_transpose_backward));
  m.impl("cudnn_convolution_transpose_backward_input", TORCH_FN(ProfiledType::cudnn_convolution_transpose_backward_input));
  m.impl("cudnn_grid_sampler", TORCH_FN(ProfiledType::cudnn_grid_sampler));
  m.impl("cudnn_is_acceptable", TORCH_FN(ProfiledType::cudnn_is_acceptable));
  m.impl("cummin", TORCH_FN(ProfiledType::cummin));
  m.impl_UNBOXED("cummin.dimname", &ProfiledType::cummin_dimname);
  m.impl_UNBOXED("deg2rad.out", &ProfiledType::deg2rad_out_out);
  m.impl("diag_embed", TORCH_FN(ProfiledType::diag_embed));
  m.impl_UNBOXED("diag.out", &ProfiledType::diag_out_out);
  m.impl_UNBOXED("digamma.out", &ProfiledType::digamma_out_out);
  m.impl("eig", TORCH_FN(ProfiledType::eig));
  m.impl_UNBOXED("empty.names", &ProfiledType::empty_names);
  m.impl_UNBOXED("empty.memory_format", &ProfiledType::empty_memory_format);
  m.impl("eq.Scalar", TORCH_FN(ProfiledType::eq_Scalar));
  m.impl("eq.Tensor", TORCH_FN(ProfiledType::eq_Tensor));
  m.impl_UNBOXED("eq_.Scalar", &ProfiledType::eq__Scalar);
  m.impl_UNBOXED("eq_.Tensor", &ProfiledType::eq__Tensor);
  m.impl_UNBOXED("erfinv.out", &ProfiledType::erfinv_out_out);
  m.impl("fake_quantize_per_tensor_affine", TORCH_FN(ProfiledType::fake_quantize_per_tensor_affine));
  m.impl("fbgemm_pack_quantized_matrix", TORCH_FN(ProfiledType::fbgemm_pack_quantized_matrix));
  m.impl("fbgemm_pack_quantized_matrix.KN", TORCH_FN(ProfiledType::fbgemm_pack_quantized_matrix_KN));
  m.impl("feature_dropout", TORCH_FN(ProfiledType::feature_dropout));
  m.impl_UNBOXED("feature_dropout_", &ProfiledType::feature_dropout_);
  m.impl_UNBOXED("fill_diagonal_", &ProfiledType::fill_diagonal_);
  m.impl("fliplr", TORCH_FN(ProfiledType::fliplr));
  m.impl("flipud", TORCH_FN(ProfiledType::flipud));
  m.impl_UNBOXED("floor_divide.out", &ProfiledType::floor_divide_out_out);
  m.impl("fractional_max_pool3d_backward", TORCH_FN(ProfiledType::fractional_max_pool3d_backward));
  m.impl_UNBOXED("full.out", &ProfiledType::full_out_out);
  m.impl_UNBOXED("gather.out", &ProfiledType::gather_out_out);
  m.impl_UNBOXED("gather.dimname_out", &ProfiledType::gather_out_dimname_out);
  m.impl("ge.Scalar", TORCH_FN(ProfiledType::ge_Scalar));
  m.impl("ge.Tensor", TORCH_FN(ProfiledType::ge_Tensor));
  m.impl_UNBOXED("ge_.Scalar", &ProfiledType::ge__Scalar);
  m.impl_UNBOXED("ge_.Tensor", &ProfiledType::ge__Tensor);
  m.impl("gelu", TORCH_FN(ProfiledType::gelu));
  m.impl_UNBOXED("geometric_", &ProfiledType::geometric_);
  m.impl("glu_backward", TORCH_FN(ProfiledType::glu_backward));
  m.impl("grid_sampler_2d_backward", TORCH_FN(ProfiledType::grid_sampler_2d_backward));
  m.impl("grid_sampler_3d", TORCH_FN(ProfiledType::grid_sampler_3d));
  m.impl("gru.input", TORCH_FN(ProfiledType::gru_input));
  m.impl("gru.data", TORCH_FN(ProfiledType::gru_data));
  m.impl("gt.Scalar", TORCH_FN(ProfiledType::gt_Scalar));
  m.impl("gt.Tensor", TORCH_FN(ProfiledType::gt_Tensor));
  m.impl_UNBOXED("gt_.Scalar", &ProfiledType::gt__Scalar);
  m.impl_UNBOXED("gt_.Tensor", &ProfiledType::gt__Tensor);
  m.impl("hardsigmoid_backward", TORCH_FN(ProfiledType::hardsigmoid_backward));
  m.impl("hardswish_backward", TORCH_FN(ProfiledType::hardswish_backward));
  m.impl("hinge_embedding_loss", TORCH_FN(ProfiledType::hinge_embedding_loss));
  m.impl("histc", TORCH_FN(ProfiledType::histc));
  m.impl("hspmm", TORCH_FN(ProfiledType::hspmm));
  m.impl("imag", TORCH_FN(ProfiledType::imag));
  m.impl("index_copy", TORCH_FN(ProfiledType::index_copy));
  m.impl_UNBOXED("index_copy.dimname", &ProfiledType::index_copy_dimname);
  m.impl_UNBOXED("index_copy_", &ProfiledType::index_copy_);
  m.impl_UNBOXED("index_copy_.dimname", &ProfiledType::index_copy__dimname);
  m.impl("index_fill.int_Scalar", TORCH_FN(ProfiledType::index_fill_int_Scalar));
  m.impl("index_fill.int_Tensor", TORCH_FN(ProfiledType::index_fill_int_Tensor));
  m.impl_UNBOXED("index_fill.Dimname_Scalar", &ProfiledType::index_fill_Dimname_Scalar);
  m.impl_UNBOXED("index_fill.Dimname_Tensor", &ProfiledType::index_fill_Dimname_Tensor);
  m.impl_UNBOXED("index_fill_.int_Scalar", &ProfiledType::index_fill__int_Scalar);
  m.impl_UNBOXED("index_fill_.int_Tensor", &ProfiledType::index_fill__int_Tensor);
  m.impl_UNBOXED("index_fill_.Dimname_Scalar", &ProfiledType::index_fill__Dimname_Scalar);
  m.impl_UNBOXED("index_fill_.Dimname_Tensor", &ProfiledType::index_fill__Dimname_Tensor);
  m.impl("inverse", TORCH_FN(ProfiledType::inverse));
  m.impl("irfft", TORCH_FN(ProfiledType::irfft));
  m.impl("is_nonzero", TORCH_FN(ProfiledType::is_nonzero));
  m.impl("is_set_to", TORCH_FN(ProfiledType::is_set_to));
  m.impl("is_signed", TORCH_FN(ProfiledType::is_signed));
  m.impl("isclose", TORCH_FN(ProfiledType::isclose));
  m.impl("isfinite", TORCH_FN(ProfiledType::isfinite));
  m.impl_UNBOXED("istft", &ProfiledType::istft);
  m.impl("kl_div_backward", TORCH_FN(ProfiledType::kl_div_backward));
  m.impl("le.Scalar", TORCH_FN(ProfiledType::le_Scalar));
  m.impl("le.Tensor", TORCH_FN(ProfiledType::le_Tensor));
  m.impl_UNBOXED("le_.Scalar", &ProfiledType::le__Scalar);
  m.impl_UNBOXED("le_.Tensor", &ProfiledType::le__Tensor);
  m.impl("leaky_relu", TORCH_FN(ProfiledType::leaky_relu));
  m.impl_UNBOXED("leaky_relu_", &ProfiledType::leaky_relu_);
  m.impl_UNBOXED("lerp.Scalar_out", &ProfiledType::lerp_out_Scalar_out);
  m.impl_UNBOXED("lerp.Tensor_out", &ProfiledType::lerp_out_Tensor_out);
  m.impl("log10", TORCH_FN(ProfiledType::log10));
  m.impl_UNBOXED("log10_", &ProfiledType::log10_);
  m.impl_UNBOXED("log_sigmoid.out", &ProfiledType::log_sigmoid_out_out);
  m.impl("logdet", TORCH_FN(ProfiledType::logdet));
  m.impl_UNBOXED("lstm_cell", &ProfiledType::lstm_cell);
  m.impl("lt.Scalar", TORCH_FN(ProfiledType::lt_Scalar));
  m.impl("lt.Tensor", TORCH_FN(ProfiledType::lt_Tensor));
  m.impl_UNBOXED("lt_.Scalar", &ProfiledType::lt__Scalar);
  m.impl_UNBOXED("lt_.Tensor", &ProfiledType::lt__Tensor);
  m.impl("masked_select", TORCH_FN(ProfiledType::masked_select));
  m.impl("matrix_rank.tol", TORCH_FN(ProfiledType::matrix_rank_tol));
  m.impl("matrix_rank", TORCH_FN(ProfiledType::matrix_rank));
  m.impl("max_pool3d", TORCH_FN(ProfiledType::max_pool3d));
  m.impl("max_pool3d_with_indices_backward", TORCH_FN(ProfiledType::max_pool3d_with_indices_backward));
  m.impl_UNBOXED("max_unpool2d.out", &ProfiledType::max_unpool2d_out_out);
  m.impl("min.dim", TORCH_FN(ProfiledType::min_dim));
  m.impl_UNBOXED("min.names_dim", &ProfiledType::min_names_dim);
  m.impl("min.other", TORCH_FN(ProfiledType::min_other));
  m.impl("min", TORCH_FN(ProfiledType::min));
  m.impl_UNBOXED("miopen_convolution", &ProfiledType::miopen_convolution);
  m.impl("miopen_convolution_transpose_backward_weight", TORCH_FN(ProfiledType::miopen_convolution_transpose_backward_weight));
  m.impl("mkldnn_convolution_backward_weights", TORCH_FN(ProfiledType::mkldnn_convolution_backward_weights));
  m.impl_UNBOXED("mkldnn_linear", &ProfiledType::mkldnn_linear);
  m.impl("mse_loss", TORCH_FN(ProfiledType::mse_loss));
  m.impl_UNBOXED("mse_loss_backward.grad_input", &ProfiledType::mse_loss_backward_out_grad_input);
  m.impl("mul.Tensor", TORCH_FN(ProfiledType::mul_Tensor));
  m.impl("mul.Scalar", TORCH_FN(ProfiledType::mul_Scalar));
  m.impl_UNBOXED("mul_.Tensor", &ProfiledType::mul__Tensor);
  m.impl_UNBOXED("mul_.Scalar", &ProfiledType::mul__Scalar);
  m.impl("multilabel_margin_loss", TORCH_FN(ProfiledType::multilabel_margin_loss));
  m.impl_UNBOXED("multilabel_margin_loss_backward.grad_input", &ProfiledType::multilabel_margin_loss_backward_out_grad_input);
  m.impl_UNBOXED("multinomial.out", &ProfiledType::multinomial_out_out);
  m.impl_UNBOXED("native_batch_norm.out", &ProfiledType::native_batch_norm_out_out);
  m.impl_UNBOXED("native_group_norm_backward", &ProfiledType::native_group_norm_backward);
  m.impl("neg", TORCH_FN(ProfiledType::neg));
  m.impl_UNBOXED("neg_", &ProfiledType::neg_);
  m.impl_UNBOXED("new_empty", &ProfiledType::new_empty);
  m.impl_UNBOXED("nll_loss2d.out", &ProfiledType::nll_loss2d_out_out);
  m.impl_UNBOXED("nll_loss.out", &ProfiledType::nll_loss_out_out);
  m.impl("nonzero", TORCH_FN(ProfiledType::nonzero));
  m.impl_UNBOXED("norm.ScalarOpt_dtype", &ProfiledType::norm_ScalarOpt_dtype);
  m.impl("norm.Scalar", TORCH_FN(ProfiledType::norm_Scalar));
  m.impl_UNBOXED("norm.ScalarOpt_dim_dtype", &ProfiledType::norm_ScalarOpt_dim_dtype);
  m.impl("norm.ScalarOpt_dim", TORCH_FN(ProfiledType::norm_ScalarOpt_dim));
  m.impl_UNBOXED("norm.names_ScalarOpt_dim_dtype", &ProfiledType::norm_names_ScalarOpt_dim_dtype);
  m.impl_UNBOXED("norm.names_ScalarOpt_dim", &ProfiledType::norm_names_ScalarOpt_dim);
  m.impl_UNBOXED("normal.Tensor_float", &ProfiledType::normal_Tensor_float);
  m.impl_UNBOXED("normal.float_Tensor", &ProfiledType::normal_float_Tensor);
  m.impl_UNBOXED("normal.Tensor_Tensor", &ProfiledType::normal_Tensor_Tensor);
  m.impl_UNBOXED("normal.float_float", &ProfiledType::normal_float_float);
  m.impl_UNBOXED("normal_", &ProfiledType::normal_);
  m.impl_UNBOXED("nuclear_norm.out", &ProfiledType::nuclear_norm_out_out);
  m.impl_UNBOXED("nuclear_norm.dim_out", &ProfiledType::nuclear_norm_out_dim_out);
  m.impl_UNBOXED("orgqr.out", &ProfiledType::orgqr_out_out);
  m.impl_UNBOXED("poisson", &ProfiledType::poisson);
  m.impl("poisson_nll_loss", TORCH_FN(ProfiledType::poisson_nll_loss));
  m.impl_UNBOXED("promote_types", &ProfiledType::promote_types);
  m.impl("q_per_channel_scales", TORCH_FN(ProfiledType::q_per_channel_scales));
  m.impl_UNBOXED("quantized_batch_norm", &ProfiledType::quantized_batch_norm);
  m.impl("quantized_max_pool2d", TORCH_FN(ProfiledType::quantized_max_pool2d));
  m.impl_UNBOXED("rad2deg.out", &ProfiledType::rad2deg_out_out);
  m.impl_UNBOXED("range.out", &ProfiledType::range_out_out);
  m.impl_UNBOXED("reciprocal.out", &ProfiledType::reciprocal_out_out);
  m.impl_UNBOXED("reflection_pad1d.out", &ProfiledType::reflection_pad1d_out_out);
  m.impl("renorm", TORCH_FN(ProfiledType::renorm));
  m.impl_UNBOXED("renorm_", &ProfiledType::renorm_);
  m.impl("replication_pad3d_backward", TORCH_FN(ProfiledType::replication_pad3d_backward));
  m.impl_UNBOXED("requires_grad_", &ProfiledType::requires_grad_);
  m.impl("reshape", TORCH_FN(ProfiledType::reshape));
  m.impl("rfft", TORCH_FN(ProfiledType::rfft));
  m.impl("rnn_relu.input", TORCH_FN(ProfiledType::rnn_relu_input));
  m.impl("rnn_relu.data", TORCH_FN(ProfiledType::rnn_relu_data));
  m.impl_UNBOXED("rrelu", &ProfiledType::rrelu);
  m.impl_UNBOXED("rrelu_", &ProfiledType::rrelu_);
  m.impl_UNBOXED("rrelu_with_noise.out", &ProfiledType::rrelu_with_noise_out_out);
  m.impl("rsub.Tensor", TORCH_FN(ProfiledType::rsub_Tensor));
  m.impl("rsub.Scalar", TORCH_FN(ProfiledType::rsub_Scalar));
  m.impl("sigmoid_backward", TORCH_FN(ProfiledType::sigmoid_backward));
  m.impl_UNBOXED("sin.out", &ProfiledType::sin_out_out);
  m.impl("sinh", TORCH_FN(ProfiledType::sinh));
  m.impl_UNBOXED("sinh_", &ProfiledType::sinh_);
  m.impl("slogdet", TORCH_FN(ProfiledType::slogdet));
  m.impl_UNBOXED("slow_conv3d_forward", &ProfiledType::slow_conv3d_forward);
  m.impl("slow_conv_dilated2d_backward", TORCH_FN(ProfiledType::slow_conv_dilated2d_backward));
  m.impl_UNBOXED("slow_conv_dilated3d", &ProfiledType::slow_conv_dilated3d);
  m.impl("slow_conv_transpose3d_backward.output_mask", TORCH_FN(ProfiledType::slow_conv_transpose3d_backward_output_mask));
  m.impl("softshrink", TORCH_FN(ProfiledType::softshrink));
  m.impl_UNBOXED("softshrink_backward.grad_input", &ProfiledType::softshrink_backward_out_grad_input);
  m.impl("stack", TORCH_FN(ProfiledType::stack));
  m.impl_UNBOXED("stft", &ProfiledType::stft);
  m.impl_UNBOXED("sub.out", &ProfiledType::sub_out_out);
  m.impl("symeig", TORCH_FN(ProfiledType::symeig));
  m.impl("tensordot", TORCH_FN(ProfiledType::tensordot));
  m.impl("thnn_conv2d_backward.output_mask", TORCH_FN(ProfiledType::thnn_conv2d_backward_output_mask));
  m.impl_UNBOXED("thnn_conv2d_forward.output", &ProfiledType::thnn_conv2d_forward_out_output);
  m.impl("threshold_backward", TORCH_FN(ProfiledType::threshold_backward));
  m.impl("to_dense", TORCH_FN(ProfiledType::to_dense));
  m.impl("triangular_solve", TORCH_FN(ProfiledType::triangular_solve));
  m.impl("triplet_margin_loss", TORCH_FN(ProfiledType::triplet_margin_loss));
  m.impl_UNBOXED("unfold_backward", &ProfiledType::unfold_backward);
  m.impl("unique_dim_consecutive", TORCH_FN(ProfiledType::unique_dim_consecutive));
  m.impl("upsample_bilinear2d", TORCH_FN(ProfiledType::upsample_bilinear2d));
  m.impl_UNBOXED("upsample_bilinear2d_backward.grad_input", &ProfiledType::upsample_bilinear2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_linear1d.out", &ProfiledType::upsample_linear1d_out_out);
  m.impl("upsample_nearest1d", TORCH_FN(ProfiledType::upsample_nearest1d));
  m.impl_UNBOXED("upsample_nearest1d_backward.grad_input", &ProfiledType::upsample_nearest1d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest2d.out", &ProfiledType::upsample_nearest2d_out_out);
  m.impl("var", TORCH_FN(ProfiledType::var));
  m.impl("var.dim", TORCH_FN(ProfiledType::var_dim));
  m.impl_UNBOXED("var.names_dim", &ProfiledType::var_names_dim);
  m.impl("where.self", TORCH_FN(ProfiledType::where_self));
  m.impl("where", TORCH_FN(ProfiledType::where));
  m.impl_UNBOXED("zeros_like", &ProfiledType::zeros_like);;
}

}  // namespace

} // namespace torch
