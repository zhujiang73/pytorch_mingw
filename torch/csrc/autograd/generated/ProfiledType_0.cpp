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
Tensor & __ilshift___Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ilshift__", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("__ilshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __ilshift___Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ilshift__", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__ilshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __ior___Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ior__", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("__ior__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __ior___Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ior__", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__ior__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __ixor___Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ixor__", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("__ixor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & __ixor___Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__ixor__", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__ixor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __lshift___Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__lshift__", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("__lshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __lshift___Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__lshift__", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__lshift__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __or___Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__or__", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("__or__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __or___Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__or__", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__or__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __xor___Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__xor__", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("__xor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor __xor___Tensor(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::__xor__", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("__xor__", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & _addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_addr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, vec1, vec2, beta, alpha);
}
Tensor & _baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_baddbmm_mkl_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("_baddbmm_mkl_", std::vector<c10::IValue>({self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, batch1, batch2, beta, alpha);
}
Tensor & _bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_bmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_bmm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, mat2, deterministic);
}
Tensor _cast_Double(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Double", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Double", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
Tensor _cast_Short(const Tensor & self, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cast_Short", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("_cast_Short", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, non_blocking);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cudnn_rnn", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("_cudnn_rnn", std::vector<c10::IValue>({input, weight_buf, hx, cx, dropout_state}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, TensorList, int64_t, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t, bool, double, bool, bool, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
Tensor & _cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_cumsum", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
}
int64_t _dimV(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_dimV", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("_dimV", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool)>();
  RECORD_FUNCTION("_embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool>(op, c10::DispatchKey::Profiler, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}
Tensor _embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_embedding_bag_sparse_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_embedding_bag_sparse_backward", std::vector<c10::IValue>({grad, indices, offsets, offset2bag, bag_size, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
}
Tensor _gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_gather_sparse_backward", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_gather_sparse_backward", std::vector<c10::IValue>({self, index, grad}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, grad);
}
Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_index_put_impl_", "")
      .typed<Tensor & (Tensor &, TensorList, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("_index_put_impl_", std::vector<c10::IValue>({self, values}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, TensorList, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, self, indices, values, accumulate, unsafe);
}
Tensor _indices(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_indices", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("_indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Scalar _local_scalar_dense(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_local_scalar_dense", "")
      .typed<Scalar (const Tensor &)>();
  RECORD_FUNCTION("_local_scalar_dense", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Scalar, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & _logcumsumexp_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_logcumsumexp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_logcumsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
}
Tensor _mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mkldnn_transpose", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("_mkldnn_transpose", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim0, dim1);
}
Tensor & _mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mkldnn_transpose_", "")
      .typed<Tensor & (Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("_mkldnn_transpose_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim0, dim1);
}
std::tuple<Tensor &,Tensor &> _mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_mode", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, keepdim);
}
Tensor _nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_nnpack_spatial_convolution_backward_weight", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_nnpack_spatial_convolution_backward_weight", std::vector<c10::IValue>({input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weightsize, grad_output, padding);
}
Tensor _pdist_forward(const Tensor & self, double p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_pdist_forward", "")
      .typed<Tensor (const Tensor &, double)>();
  RECORD_FUNCTION("_pdist_forward", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double>(op, c10::DispatchKey::Profiler, self, p);
}
Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_softmax", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("_softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, half_to_float);
}
Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_coo_tensor_with_dims", "")
      .typed<Tensor (int64_t, int64_t, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("_sparse_coo_tensor_with_dims", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, int64_t, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, sparse_dim, dense_dim, size, options);
}
Tensor _sparse_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_softmax_backward_data", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("_sparse_softmax_backward_data", std::vector<c10::IValue>({grad_output, output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, output, dim, self);
}
Tensor _sparse_sum(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor _sparse_sum_dtype(const Tensor & self, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "dtype")
      .typed<Tensor (const Tensor &, ScalarType)>();
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, ScalarType>(op, c10::DispatchKey::Profiler, self, dtype);
}
Tensor _sparse_sum_dim(const Tensor & self, IntArrayRef dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor _sparse_sum_dim_dtype(const Tensor & self, IntArrayRef dim, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_sparse_sum", "dim_dtype")
      .typed<Tensor (const Tensor &, IntArrayRef, ScalarType)>();
  RECORD_FUNCTION("_sparse_sum", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, ScalarType>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_differentiable_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & input_bias, const Tensor & hidden_bias, const Tensor & cx, const Tensor & cy) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_differentiable_lstm_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_thnn_differentiable_lstm_cell_backward", std::vector<c10::IValue>({grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
}
std::tuple<Tensor,Tensor> _thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_gru_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("_thnn_fused_gru_cell", std::vector<c10::IValue>({input_gates, hidden_gates, hx, input_bias, hidden_bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_thnn_fused_lstm_cell_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("_thnn_fused_lstm_cell_backward", std::vector<c10::IValue>({grad_hy, grad_cy, cx, cy, workspace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, grad_hy, grad_cy, cx, cy, workspace, has_bias);
}
Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_trilinear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("_trilinear", std::vector<c10::IValue>({i1, i2, i3}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}
Tensor _unsafe_view(const Tensor & self, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_unsafe_view", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("_unsafe_view", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, size);
}
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_weight_norm_cuda_interface", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("_weight_norm_cuda_interface", std::vector<c10::IValue>({v, g}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, v, g, dim);
}
Tensor abs(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("abs", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & abs_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::abs_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("abs_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & acosh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::acosh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("acosh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor & adaptive_avg_pool2d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_avg_pool2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_avg_pool2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, output_size);
}
Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool2d_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool2d_backward", std::vector<c10::IValue>({grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, indices);
}
std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("adaptive_max_pool3d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor & adaptive_max_pool3d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::adaptive_max_pool3d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("adaptive_max_pool3d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices);
}
Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcmul", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, tensor1, tensor2, value);
}
Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addcmul_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("addcmul_", std::vector<c10::IValue>({self, tensor1, tensor2, value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, tensor1, tensor2, value);
}
Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmm", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmm", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, mat1, mat2, beta, alpha);
}
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addmm_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addmm_", std::vector<c10::IValue>({self, mat1, mat2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, mat1, mat2, beta, alpha);
}
Tensor & addr_out_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::addr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("addr_out", std::vector<c10::IValue>({out, self, vec1, vec2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, vec1, vec2, beta, alpha);
}
Tensor align_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("align_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
std::vector<Tensor> align_tensors(TensorList tensors) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_tensors", "")
      .typed<std::vector<Tensor> (TensorList)>();
  RECORD_FUNCTION("align_tensors", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList>(op, c10::DispatchKey::Profiler, tensors);
}
Tensor align_to(const Tensor & self, DimnameList names) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_to", "")
      .typed<Tensor (const Tensor &, DimnameList)>();
  RECORD_FUNCTION("align_to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList>(op, c10::DispatchKey::Profiler, self, names);
}
Tensor align_to_ellipsis_idx(const Tensor & self, DimnameList order, int64_t ellipsis_idx) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::align_to", "ellipsis_idx")
      .typed<Tensor (const Tensor &, DimnameList, int64_t)>();
  RECORD_FUNCTION("align_to", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, int64_t>(op, c10::DispatchKey::Profiler, self, order, ellipsis_idx);
}
Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argmax", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  RECORD_FUNCTION("argmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argsort", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("argsort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, dim, descending);
}
Tensor argsort_dimname(const Tensor & self, Dimname dim, bool descending) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::argsort", "dimname")
      .typed<Tensor (const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("argsort", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, self, dim, descending);
}
Tensor & asinh_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::asinh", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("asinh_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor atan(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("atan", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor atan2(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atan2", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & atan2_(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan2_", "")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("atan2_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & atan_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::atan_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("atan_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & avg_pool3d_out_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::avg_pool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("avg_pool3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}
Tensor & baddbmm_out_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::baddbmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("baddbmm_out", std::vector<c10::IValue>({out, self, batch1, batch2, beta, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, batch1, batch2, beta, alpha);
}
Tensor bartlett_window(int64_t window_length, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bartlett_window", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("bartlett_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, options);
}
Tensor bartlett_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bartlett_window", "periodic")
      .typed<Tensor (int64_t, bool, const TensorOptions &)>();
  RECORD_FUNCTION("bartlett_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, periodic, options);
}
std::tuple<Tensor,Tensor> batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::batch_norm_update_stats", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, double)>();
  RECORD_FUNCTION("batch_norm_update_stats", std::vector<c10::IValue>({input, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, double>(op, c10::DispatchKey::Profiler, input, running_mean, running_var, momentum);
}
Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::binary_cross_entropy_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("binary_cross_entropy_backward", std::vector<c10::IValue>({grad_output, self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, weight, reduction);
}
Tensor & bitwise_and_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_and_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & bitwise_and_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_and", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("bitwise_and_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & bitwise_not_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bitwise_not", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bitwise_not_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor & bmm_out_out(Tensor & out, const Tensor & self, const Tensor & mat2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bmm", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("bmm_out", std::vector<c10::IValue>({out, self, mat2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, mat2);
}
Tensor & bucketize_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & boundaries, bool out_int32, bool right) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::bucketize", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("bucketize_out", std::vector<c10::IValue>({out, self, boundaries}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, out, self, boundaries, out_int32, right);
}
Tensor cdist(const Tensor & x1, const Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cdist", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, c10::optional<int64_t>)>();
  RECORD_FUNCTION("cdist", std::vector<c10::IValue>({x1, x2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, x1, x2, p, compute_mode);
}
Tensor celu(const Tensor & self, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::celu", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("celu", std::vector<c10::IValue>({self, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, alpha);
}
Tensor & celu_(Tensor & self, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::celu_", "")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("celu_", std::vector<c10::IValue>({self, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, alpha);
}
Tensor cholesky_inverse(const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky_inverse", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_inverse", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, upper);
}
Tensor & cholesky_out_out(Tensor & out, const Tensor & self, bool upper) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cholesky", "out")
      .typed<Tensor & (Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("cholesky_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, out, self, upper);
}
Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp", "")
      .typed<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>();
  RECORD_FUNCTION("clamp", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, c10::DispatchKey::Profiler, self, min, max);
}
Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_", "")
      .typed<Tensor & (Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>();
  RECORD_FUNCTION("clamp_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, c10::DispatchKey::Profiler, self, min, max);
}
Tensor & clamp_max_out_out(Tensor & out, const Tensor & self, Scalar max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::clamp_max", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("clamp_max_out", std::vector<c10::IValue>({out, self, max}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, max);
}
Tensor & conj_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conj", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("conj_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor conv_transpose3d_input(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::conv_transpose3d", "input")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef)>();
  RECORD_FUNCTION("conv_transpose3d", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, IntArrayRef>(op, c10::DispatchKey::Profiler, input, weight, bias, stride, padding, output_padding, groups, dilation);
}
std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::convolution_backward_overrideable", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>)>();
  RECORD_FUNCTION("convolution_backward_overrideable", std::vector<c10::IValue>({grad_output, input, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, bool, IntArrayRef, int64_t, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
}
Tensor cos(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cos", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("cos", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & cos_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cos_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("cos_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon, const Tensor & reserveSpace) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_batch_norm_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &)>();
  RECORD_FUNCTION("cudnn_batch_norm_backward", std::vector<c10::IValue>({input, grad_output, weight, running_mean, running_var, save_mean, save_var, reserveSpace}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, double, const Tensor &>(op, c10::DispatchKey::Profiler, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
}
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cudnn_grid_sampler_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("cudnn_grid_sampler_backward", std::vector<c10::IValue>({self, grid, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, grid, grad_output);
}
std::tuple<Tensor &,Tensor &> cummax_out_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("cummax_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, values, indices, self, dim);
}
std::tuple<Tensor &,Tensor &> cummax_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cummax", "dimname_out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname)>();
  RECORD_FUNCTION("cummax_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, values, indices, self, dim);
}
Tensor & cumsum_out_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, dtype);
}
Tensor & cumsum_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::cumsum", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("cumsum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, dtype);
}
Tensor data(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::data", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("data", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor deg2rad(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("deg2rad", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & deg2rad_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::deg2rad_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("deg2rad_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor diag(const Tensor & self, int64_t diagonal) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::diag", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("diag", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, diagonal);
}
Tensor digamma(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("digamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & digamma_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::digamma_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("digamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & elu_out_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::elu", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("elu_out", std::vector<c10::IValue>({out, self, alpha, scale, input_scale}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, alpha, scale, input_scale);
}
std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights, bool include_last_offset) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_bag", "")
      .typed<std::tuple<Tensor,Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool)>();
  RECORD_FUNCTION("embedding_bag", std::vector<c10::IValue>({weight, indices, offsets, per_sample_weights}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, bool, int64_t, bool, const Tensor &, bool>(op, c10::DispatchKey::Profiler, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}
Tensor embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::embedding_dense_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("embedding_dense_backward", std::vector<c10::IValue>({grad_output, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}
Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_like", "")
      .typed<Tensor (const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("empty_like", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const TensorOptions &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, options, memory_format);
}
Tensor empty_quantized(IntArrayRef size, const Tensor & qtensor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_quantized", "")
      .typed<Tensor (IntArrayRef, const Tensor &)>();
  RECORD_FUNCTION("empty_quantized", std::vector<c10::IValue>({qtensor}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &>(op, c10::DispatchKey::Profiler, size, qtensor);
}
Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::empty_strided", "")
      .typed<Tensor (IntArrayRef, IntArrayRef, const TensorOptions &)>();
  RECORD_FUNCTION("empty_strided", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, IntArrayRef, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, stride, options);
}
Tensor & erfc_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfc", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("erfc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor erfinv(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("erfinv", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & erfinv_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::erfinv_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("erfinv_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor expand(const Tensor & self, IntArrayRef size, bool implicit) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expand", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("expand", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, size, implicit);
}
Tensor & expm1_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::expm1", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("expm1_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor fake_quantize_per_tensor_affine_backward(const Tensor & grad, const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fake_quantize_per_tensor_affine_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("fake_quantize_per_tensor_affine_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, double, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, grad, self, scale, zero_point, quant_min, quant_max);
}
Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::fft", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("fft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized);
}
Tensor flatten_using_ints(const Tensor & self, int64_t start_dim, int64_t end_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "using_ints")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, start_dim, end_dim);
}
Tensor flatten_named_out_dim(const Tensor & self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "named_out_dim")
      .typed<Tensor (const Tensor &, int64_t, int64_t, Dimname)>();
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, Dimname>(op, c10::DispatchKey::Profiler, self, start_dim, end_dim, out_dim);
}
Tensor flatten_using_names(const Tensor & self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "using_names")
      .typed<Tensor (const Tensor &, Dimname, Dimname, Dimname)>();
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, Dimname, Dimname>(op, c10::DispatchKey::Profiler, self, start_dim, end_dim, out_dim);
}
Tensor flatten_DimnameList(const Tensor & self, DimnameList dims, Dimname out_dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::flatten", "DimnameList")
      .typed<Tensor (const Tensor &, DimnameList, Dimname)>();
  RECORD_FUNCTION("flatten", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, Dimname>(op, c10::DispatchKey::Profiler, self, dims, out_dim);
}
Tensor floor_divide(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("floor_divide", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor floor_divide_Scalar(const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide", "Scalar")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("floor_divide", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & floor_divide__Tensor(Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("floor_divide_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & floor_divide__Scalar(Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor_divide_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar)>();
  RECORD_FUNCTION("floor_divide_", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & floor_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::floor", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("floor_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
Tensor full_names(IntArrayRef size, Scalar fill_value, c10::optional<DimnameList> names, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "names")
      .typed<Tensor (IntArrayRef, Scalar, c10::optional<DimnameList>, const TensorOptions &)>();
  RECORD_FUNCTION("full", std::vector<c10::IValue>({fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, Scalar, c10::optional<DimnameList>, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, fill_value, names, options);
}
Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::full", "")
      .typed<Tensor (IntArrayRef, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("full", std::vector<c10::IValue>({fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, size, fill_value, options);
}
Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, bool)>();
  RECORD_FUNCTION("gather", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, dim, index, sparse_grad);
}
Tensor gather_dimname(const Tensor & self, Dimname dim, const Tensor & index, bool sparse_grad) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gather", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, bool)>();
  RECORD_FUNCTION("gather", std::vector<c10::IValue>({self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, dim, index, sparse_grad);
}
Tensor gelu_backward(const Tensor & grad, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gelu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gelu_backward", std::vector<c10::IValue>({grad, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad, self);
}
std::tuple<Tensor,Tensor> grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::grid_sampler_3d_backward", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool)>();
  RECORD_FUNCTION("grid_sampler_3d_backward", std::vector<c10::IValue>({grad_output, input, grid}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, bool>(op, c10::DispatchKey::Profiler, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}
Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::gru_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("gru_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor hann_window(int64_t window_length, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hann_window", "")
      .typed<Tensor (int64_t, const TensorOptions &)>();
  RECORD_FUNCTION("hann_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, options);
}
Tensor hann_window_periodic(int64_t window_length, bool periodic, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hann_window", "periodic")
      .typed<Tensor (int64_t, bool, const TensorOptions &)>();
  RECORD_FUNCTION("hann_window", std::vector<c10::IValue>({}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, int64_t, bool, const TensorOptions &>(op, c10::DispatchKey::Profiler, window_length, periodic, options);
}
Tensor hardshrink(const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::hardshrink", "")
      .typed<Tensor (const Tensor &, Scalar)>();
  RECORD_FUNCTION("hardshrink", std::vector<c10::IValue>({self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, lambd);
}
Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ifft", "")
      .typed<Tensor (const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("ifft", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, self, signal_ndim, normalized);
}
Tensor & index_select_out_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("index_select_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, dim, index);
}
Tensor & index_select_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::index_select", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname, const Tensor &)>();
  RECORD_FUNCTION("index_select_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, dim, index);
}
Tensor indices(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::indices", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("indices", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_complex(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_complex", "")
      .typed<bool (const Tensor &)>();
  RECORD_FUNCTION("is_complex", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
bool is_same_size(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::is_same_size", "")
      .typed<bool (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("is_same_size", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<bool, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor & l1_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::l1_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("l1_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
}
Tensor layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::layer_norm", "")
      .typed<Tensor (const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool)>();
  RECORD_FUNCTION("layer_norm", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, const Tensor &, const Tensor &, double, bool>(op, c10::DispatchKey::Profiler, input, normalized_shape, weight, bias, eps, cudnn_enable);
}
Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope, bool self_is_result) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::leaky_relu_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, bool)>();
  RECORD_FUNCTION("leaky_relu_backward", std::vector<c10::IValue>({grad_output, self, negative_slope}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, bool>(op, c10::DispatchKey::Profiler, grad_output, self, negative_slope, self_is_result);
}
Tensor lerp_Scalar(const Tensor & self, const Tensor & end, Scalar weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Scalar")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("lerp", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, end, weight);
}
Tensor lerp_Tensor(const Tensor & self, const Tensor & end, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lerp", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, end, weight);
}
Tensor & lerp__Scalar(Tensor & self, const Tensor & end, Scalar weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp_", "Scalar")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("lerp_", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, end, weight);
}
Tensor & lerp__Tensor(Tensor & self, const Tensor & end, const Tensor & weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::lerp_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("lerp_", std::vector<c10::IValue>({self, end, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, end, weight);
}
Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::linear", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("linear", std::vector<c10::IValue>({input, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, weight, bias);
}
Tensor log_sigmoid(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & log_sigmoid_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::log_sigmoid_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("log_sigmoid_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, buffer}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, buffer);
}
Tensor & logcumsumexp_out_out(Tensor & out, const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("logcumsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, dim);
}
Tensor & logcumsumexp_out_dimname_out(Tensor & out, const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logcumsumexp", "dimname_out")
      .typed<Tensor & (Tensor &, const Tensor &, Dimname)>();
  RECORD_FUNCTION("logcumsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, out, self, dim);
}
Tensor & logical_or_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_or", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_or_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & logical_xor_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logical_xor", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("logical_xor_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & logspace_out_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logspace", "out")
      .typed<Tensor & (Tensor &, Scalar, Scalar, int64_t, double)>();
  RECORD_FUNCTION("logspace_out", std::vector<c10::IValue>({out, start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar, int64_t, double>(op, c10::DispatchKey::Profiler, out, start, end, steps, base);
}
Tensor & logsumexp_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("logsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor & logsumexp_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::logsumexp", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool)>();
  RECORD_FUNCTION("logsumexp_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim);
}
Tensor & matmul_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::matmul", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("matmul_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
std::tuple<Tensor &,Tensor &> max_out_dim_max(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "dim_max")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({max, max_values, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, max, max_values, self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> max_out_names_dim_max(Tensor & max, Tensor & max_values, const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "names_dim_max")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({max, max_values, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, max, max_values, self, dim, keepdim);
}
Tensor & max_out_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("max_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d", std::vector<c10::IValue>({self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, indices, output_size);
}
Tensor & max_unpool2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, indices, output_size);
}
Tensor & max_unpool3d_out_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::max_unpool3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("max_unpool3d_out", std::vector<c10::IValue>({out, self, indices}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, indices, output_size, stride, padding);
}
Tensor min_values(const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min_values", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("min_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor min_values_names(const Tensor & self, DimnameList dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::min_values", "names")
      .typed<Tensor (const Tensor &, DimnameList, bool)>();
  RECORD_FUNCTION("min_values", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>)>();
  RECORD_FUNCTION("miopen_convolution_backward", std::vector<c10::IValue>({self, grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool, std::array<bool,3>>(op, c10::DispatchKey::Profiler, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
Tensor miopen_convolution_backward_bias(const Tensor & grad_output) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_backward_bias", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("miopen_convolution_backward_bias", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output);
}
Tensor miopen_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_backward_input", "")
      .typed<Tensor (IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_backward_input", std::vector<c10::IValue>({grad_output, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, IntArrayRef, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::miopen_convolution_transpose", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool)>();
  RECORD_FUNCTION("miopen_convolution_transpose", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
Tensor mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_adaptive_avg_pool2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("mkldnn_adaptive_avg_pool2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, output_size);
}
Tensor mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_convolution", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("mkldnn_convolution", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, self, weight, bias, padding, stride, dilation, groups);
}
Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mkldnn_reorder_conv2d_weight", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t)>();
  RECORD_FUNCTION("mkldnn_reorder_conv2d_weight", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, int64_t>(op, c10::DispatchKey::Profiler, self, padding, stride, dilation, groups);
}
std::tuple<Tensor &,Tensor &> mode_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, keepdim);
}
std::tuple<Tensor &,Tensor &> mode_out_dimname_out(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mode", "dimname_out")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("mode_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, keepdim);
}
Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mse_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("mse_loss_backward", std::vector<c10::IValue>({grad_output, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction);
}
Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &)>();
  RECORD_FUNCTION("multilabel_margin_loss_backward", std::vector<c10::IValue>({grad_output, self, target, is_target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_output, self, target, reduction, is_target);
}
std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out_output(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multilabel_margin_loss_forward", "output")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("multilabel_margin_loss_forward_out", std::vector<c10::IValue>({output, is_target, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, output, is_target, self, target, reduction);
}
Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::multinomial", "")
      .typed<Tensor (const Tensor &, int64_t, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("multinomial", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, num_samples, replacement, generator);
}
Tensor mvlgamma(const Tensor & self, int64_t p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mvlgamma", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("mvlgamma", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, p);
}
Tensor & mvlgamma_(Tensor & self, int64_t p) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::mvlgamma_", "")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("mvlgamma_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, p);
}
Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::narrow", "")
      .typed<Tensor (const Tensor &, int64_t, int64_t, int64_t)>();
  RECORD_FUNCTION("narrow", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim, start, length);
}
Tensor narrow_Tensor(const Tensor & self, int64_t dim, const Tensor & start, int64_t length) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::narrow", "Tensor")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, int64_t)>();
  RECORD_FUNCTION("narrow", std::vector<c10::IValue>({self, start}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim, start, length);
}
std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::native_batch_norm", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double)>();
  RECORD_FUNCTION("native_batch_norm", std::vector<c10::IValue>({input, weight, bias, running_mean, running_var}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, double, double>(op, c10::DispatchKey::Profiler, input, weight, bias, running_mean, running_var, training, momentum, eps);
}
Tensor & ne_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("ne_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & ne_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ne", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("ne_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor new_full(const Tensor & self, IntArrayRef size, Scalar fill_value, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::new_full", "")
      .typed<Tensor (const Tensor &, IntArrayRef, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("new_full", std::vector<c10::IValue>({self, fill_value}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, self, size, fill_value, options);
}
Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction, ignore_index);
}
Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("nll_loss2d", std::vector<c10::IValue>({self, target, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, target, weight, reduction, ignore_index);
}
Tensor & nll_loss2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor & nll_loss_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nll_loss_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &)>();
  RECORD_FUNCTION("nll_loss_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self, target, weight, total_weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}
Tensor nuclear_norm(const Tensor & self, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "")
      .typed<Tensor (const Tensor &, bool)>();
  RECORD_FUNCTION("nuclear_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, keepdim);
}
Tensor nuclear_norm_dim(const Tensor & self, IntArrayRef dim, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::nuclear_norm", "dim")
      .typed<Tensor (const Tensor &, IntArrayRef, bool)>();
  RECORD_FUNCTION("nuclear_norm", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool>(op, c10::DispatchKey::Profiler, self, dim, keepdim);
}
Tensor orgqr(const Tensor & self, const Tensor & input2) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::orgqr", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("orgqr", std::vector<c10::IValue>({self, input2}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, input2);
}
Tensor & ormqr_out_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::ormqr", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool)>();
  RECORD_FUNCTION("ormqr_out", std::vector<c10::IValue>({out, self, input2, input3}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, c10::DispatchKey::Profiler, out, self, input2, input3, left, transpose);
}
Tensor permute(const Tensor & self, IntArrayRef dims) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::permute", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("permute", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, dims);
}
Tensor pixel_shuffle(const Tensor & self, int64_t upscale_factor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::pixel_shuffle", "")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("pixel_shuffle", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, upscale_factor);
}
Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::put_", "")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>();
  RECORD_FUNCTION("put_", std::vector<c10::IValue>({self, index, source}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, c10::DispatchKey::Profiler, self, index, source, accumulate);
}
int64_t q_zero_point(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::q_zero_point", "")
      .typed<int64_t (const Tensor &)>();
  RECORD_FUNCTION("q_zero_point", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<int64_t, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor quantize_per_tensor(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantize_per_tensor", "")
      .typed<Tensor (const Tensor &, double, int64_t, ScalarType)>();
  RECORD_FUNCTION("quantize_per_tensor", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, double, int64_t, ScalarType>(op, c10::DispatchKey::Profiler, self, scale, zero_point, dtype);
}
std::vector<Tensor> quantize_per_tensor_tensors(TensorList tensors, const Tensor & scales, const Tensor & zero_points, ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantize_per_tensor", "tensors")
      .typed<std::vector<Tensor> (TensorList, const Tensor &, const Tensor &, ScalarType)>();
  RECORD_FUNCTION("quantize_per_tensor", std::vector<c10::IValue>({scales, zero_points}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, TensorList, const Tensor &, const Tensor &, ScalarType>(op, c10::DispatchKey::Profiler, tensors, scales, zero_points, dtype);
}
std::tuple<Tensor,Tensor> quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::quantized_lstm_cell", "")
      .typed<std::tuple<Tensor,Tensor> (const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar)>();
  RECORD_FUNCTION("quantized_lstm_cell", std::vector<c10::IValue>({input, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor>, const Tensor &, TensorList, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar, Scalar, Scalar>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}
Tensor rad2deg(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("rad2deg", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & rad2deg_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rad2deg_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("rad2deg_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & rand_out_out(Tensor & out, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "out")
      .typed<Tensor & (Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("rand_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, size);
}
Tensor & rand_out_generator_out(Tensor & out, IntArrayRef size, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rand", "generator_out")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<Generator>)>();
  RECORD_FUNCTION("rand_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, size, generator);
}
Tensor & randn_out_out(Tensor & out, IntArrayRef size) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "out")
      .typed<Tensor & (Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("randn_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, size);
}
Tensor & randn_out_generator_out(Tensor & out, IntArrayRef size, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::randn", "generator_out")
      .typed<Tensor & (Tensor &, IntArrayRef, c10::optional<Generator>)>();
  RECORD_FUNCTION("randn_out", std::vector<c10::IValue>({out}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, IntArrayRef, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, out, size, generator);
}
Tensor range_step(Scalar start, Scalar end, Scalar step, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "step")
      .typed<Tensor (Scalar, Scalar, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("range", std::vector<c10::IValue>({start, end, step}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, step, options);
}
Tensor range(Scalar start, Scalar end, const TensorOptions & options) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::range", "")
      .typed<Tensor (Scalar, Scalar, const TensorOptions &)>();
  RECORD_FUNCTION("range", std::vector<c10::IValue>({start, end}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, Scalar, Scalar, const TensorOptions &>(op, c10::DispatchKey::Profiler, start, end, options);
}
Tensor real(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::real", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("real", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor reciprocal(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("reciprocal", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & reciprocal_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reciprocal_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("reciprocal_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor refine_names(const Tensor & self, DimnameList names) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::refine_names", "")
      .typed<Tensor (const Tensor &, DimnameList)>();
  RECORD_FUNCTION("refine_names", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, DimnameList>(op, c10::DispatchKey::Profiler, self, names);
}
Tensor reflection_pad1d(const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, self, padding);
}
Tensor & reflection_pad1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, grad_input, grad_output, self, padding);
}
Tensor & reflection_pad2d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::reflection_pad2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("reflection_pad2d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor relu(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::relu", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("relu", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & relu_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::relu_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("relu_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & remainder_out_Scalar_out(Tensor & out, const Tensor & self, Scalar other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Scalar_out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("remainder_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & remainder_out_Tensor_out(Tensor & out, const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::remainder", "Tensor_out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("remainder_out", std::vector<c10::IValue>({out, self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, other);
}
Tensor & replication_pad1d_out_out(Tensor & out, const Tensor & self, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::replication_pad1d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef)>();
  RECORD_FUNCTION("replication_pad1d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, padding);
}
Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::resize_as_", "")
      .typed<Tensor & (Tensor &, const Tensor &, c10::optional<MemoryFormat>)>();
  RECORD_FUNCTION("resize_as_", std::vector<c10::IValue>({self, the_template}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, c10::optional<MemoryFormat>>(op, c10::DispatchKey::Profiler, self, the_template, memory_format);
}
Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rnn_relu_cell", "")
      .typed<Tensor (const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("rnn_relu_cell", std::vector<c10::IValue>({input, hx, w_ih, w_hh, b_ih, b_hh}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, input, hx, w_ih, w_hh, b_ih, b_hh);
}
Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("rrelu_with_noise", std::vector<c10::IValue>({self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, noise, lower, upper, training, generator);
}
Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, c10::optional<Generator> generator) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::rrelu_with_noise_", "")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>)>();
  RECORD_FUNCTION("rrelu_with_noise_", std::vector<c10::IValue>({self, noise, lower, upper}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar, bool, c10::optional<Generator>>(op, c10::DispatchKey::Profiler, self, noise, lower, upper, training, generator);
}
Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_add", "")
      .typed<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("scatter_add", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, src);
}
Tensor scatter_add_dimname(const Tensor & self, Dimname dim, const Tensor & index, const Tensor & src) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_add", "dimname")
      .typed<Tensor (const Tensor &, Dimname, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("scatter_add", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, src);
}
Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::scatter_add_", "")
      .typed<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("scatter_add_", std::vector<c10::IValue>({self, index, src}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, dim, index, src);
}
Tensor select_Dimname(const Tensor & self, Dimname dim, int64_t index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::select", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, int64_t)>();
  RECORD_FUNCTION("select", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, int64_t>(op, c10::DispatchKey::Profiler, self, dim, index);
}
Tensor select_int(const Tensor & self, int64_t dim, int64_t index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::select", "int")
      .typed<Tensor (const Tensor &, int64_t, int64_t)>();
  RECORD_FUNCTION("select", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, int64_t>(op, c10::DispatchKey::Profiler, self, dim, index);
}
Tensor sin(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sin", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("sin", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & sin_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sin_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("sin_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
std::tuple<Tensor,Tensor,Tensor> slow_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::slow_conv_dilated3d_backward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>)>();
  RECORD_FUNCTION("slow_conv_dilated3d_backward", std::vector<c10::IValue>({grad_output, self, weight}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef, std::array<bool,3>>(op, c10::DispatchKey::Profiler, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
Tensor & soft_margin_loss_out_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::soft_margin_loss", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, int64_t)>();
  RECORD_FUNCTION("soft_margin_loss_out", std::vector<c10::IValue>({out, self, target}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, out, self, target, reduction);
}
Tensor softmax_int(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softmax", "int")
      .typed<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor softmax_Dimname(const Tensor & self, Dimname dim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softmax", "Dimname")
      .typed<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("softmax", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, self, dim, dtype);
}
Tensor & softplus_out_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softplus", "out")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("softplus_out", std::vector<c10::IValue>({out, self, beta, threshold}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, out, self, beta, threshold);
}
Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::softshrink_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("softshrink_backward", std::vector<c10::IValue>({grad_output, self, lambd}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, grad_output, self, lambd);
}
std::tuple<Tensor &,Tensor &> sort_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, bool)>();
  RECORD_FUNCTION("sort_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, descending);
}
std::tuple<Tensor &,Tensor &> sort_out_dimname_values(Tensor & values, Tensor & indices, const Tensor & self, Dimname dim, bool descending) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sort", "dimname_values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, Dimname, bool)>();
  RECORD_FUNCTION("sort_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, Dimname, bool>(op, c10::DispatchKey::Profiler, values, indices, self, dim, descending);
}
Tensor squeeze(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor squeeze_dim(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze", "dim")
      .typed<Tensor (const Tensor &, int64_t)>();
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor squeeze_dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze", "dimname")
      .typed<Tensor (const Tensor &, Dimname)>();
  RECORD_FUNCTION("squeeze", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & squeeze_(Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze_", "")
      .typed<Tensor & (Tensor &)>();
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor & squeeze__dim(Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze_", "dim")
      .typed<Tensor & (Tensor &, int64_t)>();
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & squeeze__dimname(Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::squeeze_", "dimname")
      .typed<Tensor & (Tensor &, Dimname)>();
  RECORD_FUNCTION("squeeze_", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
Tensor & std_out_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, bool)>();
  RECORD_FUNCTION("std_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, bool>(op, c10::DispatchKey::Profiler, out, self, dim, unbiased, keepdim);
}
Tensor & std_out_names_out(Tensor & out, const Tensor & self, DimnameList dim, bool unbiased, bool keepdim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::std", "names_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool, bool)>();
  RECORD_FUNCTION("std_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool, bool>(op, c10::DispatchKey::Profiler, out, self, dim, unbiased, keepdim);
}
Tensor sub_Tensor(const Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "Tensor")
      .typed<Tensor (const Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("sub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor sub_Scalar(const Tensor & self, Scalar other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub", "Scalar")
      .typed<Tensor (const Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("sub", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor & sub__Tensor(Tensor & self, const Tensor & other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub_", "Tensor")
      .typed<Tensor & (Tensor &, const Tensor &, Scalar)>();
  RECORD_FUNCTION("sub_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor & sub__Scalar(Tensor & self, Scalar other, Scalar alpha) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sub_", "Scalar")
      .typed<Tensor & (Tensor &, Scalar, Scalar)>();
  RECORD_FUNCTION("sub_", std::vector<c10::IValue>({self, other, alpha}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, Scalar, Scalar>(op, c10::DispatchKey::Profiler, self, other, alpha);
}
Tensor & sum_out_IntList_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "IntList_out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("sum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim, dtype);
}
Tensor & sum_out_DimnameList_out(Tensor & out, const Tensor & self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::sum", "DimnameList_out")
      .typed<Tensor & (Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>();
  RECORD_FUNCTION("sum_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, DimnameList, bool, c10::optional<ScalarType>>(op, c10::DispatchKey::Profiler, out, self, dim, keepdim, dtype);
}
Tensor & take_out_out(Tensor & out, const Tensor & self, const Tensor & index) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::take", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("take_out", std::vector<c10::IValue>({out, self, index}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self, index);
}
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv2d_forward", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv2d_forward", std::vector<c10::IValue>({self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, self, weight, kernel_size, bias, stride, padding);
}
Tensor & thnn_conv_depthwise2d_out_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::thnn_conv_depthwise2d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef)>();
  RECORD_FUNCTION("thnn_conv_depthwise2d_out", std::vector<c10::IValue>({out, self, weight, bias}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, const Tensor &, IntArrayRef, const Tensor &, IntArrayRef, IntArrayRef, IntArrayRef>(op, c10::DispatchKey::Profiler, out, self, weight, kernel_size, bias, stride, padding, dilation);
}
Tensor to_dense_backward(const Tensor & grad, const Tensor & input) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::to_dense_backward", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("to_dense_backward", std::vector<c10::IValue>({grad, input}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, grad, input);
}
std::tuple<Tensor &,Tensor &> topk_out_values(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::topk", "values")
      .typed<std::tuple<Tensor &,Tensor &> (Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool)>();
  RECORD_FUNCTION("topk_out", std::vector<c10::IValue>({values, indices, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor &,Tensor &>, Tensor &, Tensor &, const Tensor &, int64_t, int64_t, bool, bool>(op, c10::DispatchKey::Profiler, values, indices, self, k, dim, largest, sorted);
}
Tensor & trunc_out_out(Tensor & out, const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::trunc", "out")
      .typed<Tensor & (Tensor &, const Tensor &)>();
  RECORD_FUNCTION("trunc_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, out, self);
}
std::vector<Tensor> unbind_int(const Tensor & self, int64_t dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unbind", "int")
      .typed<std::vector<Tensor> (const Tensor &, int64_t)>();
  RECORD_FUNCTION("unbind", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, int64_t>(op, c10::DispatchKey::Profiler, self, dim);
}
std::vector<Tensor> unbind_Dimname(const Tensor & self, Dimname dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unbind", "Dimname")
      .typed<std::vector<Tensor> (const Tensor &, Dimname)>();
  RECORD_FUNCTION("unbind", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::vector<Tensor>, const Tensor &, Dimname>(op, c10::DispatchKey::Profiler, self, dim);
}
std::tuple<Tensor,Tensor,Tensor> unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::unique_consecutive", "")
      .typed<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool, c10::optional<int64_t>)>();
  RECORD_FUNCTION("unique_consecutive", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool, c10::optional<int64_t>>(op, c10::DispatchKey::Profiler, self, return_inverse, return_counts, dim);
}
Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_bilinear2d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_bilinear2d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}
Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, align_corners, scales);
}
Tensor & upsample_linear1d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_linear1d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_linear1d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, bool, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, align_corners, scales);
}
Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest1d_backward", "")
      .typed<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest1d_backward", std::vector<c10::IValue>({grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_output, output_size, input_size, scales);
}
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d", "")
      .typed<Tensor (const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, self, output_size, scales_h, scales_w);
}
Tensor & upsample_nearest2d_backward_out_grad_input(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest2d_backward", "grad_input")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest2d_backward_out", std::vector<c10::IValue>({grad_input, grad_output}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}
Tensor & upsample_nearest3d_out_out(Tensor & out, const Tensor & self, IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::upsample_nearest3d", "out")
      .typed<Tensor & (Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>)>();
  RECORD_FUNCTION("upsample_nearest3d_out", std::vector<c10::IValue>({out, self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor &, Tensor &, const Tensor &, IntArrayRef, c10::optional<double>, c10::optional<double>, c10::optional<double>>(op, c10::DispatchKey::Profiler, out, self, output_size, scales_d, scales_h, scales_w);
}
Tensor vander(const Tensor & x, c10::optional<int64_t> N, bool increasing) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::vander", "")
      .typed<Tensor (const Tensor &, c10::optional<int64_t>, bool)>();
  RECORD_FUNCTION("vander", std::vector<c10::IValue>({x}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, c10::DispatchKey::Profiler, x, N, increasing);
}
Tensor view_as(const Tensor & self, const Tensor & other) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view_as", "")
      .typed<Tensor (const Tensor &, const Tensor &)>();
  RECORD_FUNCTION("view_as", std::vector<c10::IValue>({self, other}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &, const Tensor &>(op, c10::DispatchKey::Profiler, self, other);
}
Tensor view_as_complex(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view_as_complex", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("view_as_complex", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
Tensor view_as_real(const Tensor & self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::view_as_real", "")
      .typed<Tensor (const Tensor &)>();
  RECORD_FUNCTION("view_as_real", std::vector<c10::IValue>({self}), Node::peek_at_next_sequence_nr());
  return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor &>(op, c10::DispatchKey::Profiler, self);
}
}  // namespace
}  // namespace ProfiledType

namespace {

TORCH_LIBRARY_IMPL(aten, Profiler, m) {
  m.impl_UNBOXED("__ilshift__.Scalar", &ProfiledType::__ilshift___Scalar);
  m.impl_UNBOXED("__ilshift__.Tensor", &ProfiledType::__ilshift___Tensor);
  m.impl_UNBOXED("__ior__.Scalar", &ProfiledType::__ior___Scalar);
  m.impl_UNBOXED("__ior__.Tensor", &ProfiledType::__ior___Tensor);
  m.impl_UNBOXED("__ixor__.Scalar", &ProfiledType::__ixor___Scalar);
  m.impl_UNBOXED("__ixor__.Tensor", &ProfiledType::__ixor___Tensor);
  m.impl("__lshift__.Scalar", TORCH_FN(ProfiledType::__lshift___Scalar));
  m.impl("__lshift__.Tensor", TORCH_FN(ProfiledType::__lshift___Tensor));
  m.impl("__or__.Scalar", TORCH_FN(ProfiledType::__or___Scalar));
  m.impl("__or__.Tensor", TORCH_FN(ProfiledType::__or___Tensor));
  m.impl("__xor__.Scalar", TORCH_FN(ProfiledType::__xor___Scalar));
  m.impl("__xor__.Tensor", TORCH_FN(ProfiledType::__xor___Tensor));
  m.impl_UNBOXED("_addr.out", &ProfiledType::_addr_out_out);
  m.impl_UNBOXED("_baddbmm_mkl_", &ProfiledType::_baddbmm_mkl_);
  m.impl_UNBOXED("_bmm.out", &ProfiledType::_bmm_out_out);
  m.impl("_cast_Double", TORCH_FN(ProfiledType::_cast_Double));
  m.impl("_cast_Short", TORCH_FN(ProfiledType::_cast_Short));
  m.impl_UNBOXED("_cudnn_rnn", &ProfiledType::_cudnn_rnn);
  m.impl_UNBOXED("_cumsum.out", &ProfiledType::_cumsum_out_out);
  m.impl("_dimV", TORCH_FN(ProfiledType::_dimV));
  m.impl_UNBOXED("_embedding_bag", &ProfiledType::_embedding_bag);
  m.impl_UNBOXED("_embedding_bag_sparse_backward", &ProfiledType::_embedding_bag_sparse_backward);
  m.impl("_gather_sparse_backward", TORCH_FN(ProfiledType::_gather_sparse_backward));
  m.impl_UNBOXED("_index_put_impl_", &ProfiledType::_index_put_impl_);
  m.impl("_indices", TORCH_FN(ProfiledType::_indices));
  m.impl("_local_scalar_dense", TORCH_FN(ProfiledType::_local_scalar_dense));
  m.impl_UNBOXED("_logcumsumexp.out", &ProfiledType::_logcumsumexp_out_out);
  m.impl("_mkldnn_transpose", TORCH_FN(ProfiledType::_mkldnn_transpose));
  m.impl_UNBOXED("_mkldnn_transpose_", &ProfiledType::_mkldnn_transpose_);
  m.impl_UNBOXED("_mode.values", &ProfiledType::_mode_out_values);
  m.impl("_nnpack_spatial_convolution_backward_weight", TORCH_FN(ProfiledType::_nnpack_spatial_convolution_backward_weight));
  m.impl("_pdist_forward", TORCH_FN(ProfiledType::_pdist_forward));
  m.impl("_softmax", TORCH_FN(ProfiledType::_softmax));
  m.impl_UNBOXED("_sparse_coo_tensor_with_dims", &ProfiledType::_sparse_coo_tensor_with_dims);
  m.impl_UNBOXED("_sparse_softmax_backward_data", &ProfiledType::_sparse_softmax_backward_data);
  m.impl("_sparse_sum", TORCH_FN(ProfiledType::_sparse_sum));
  m.impl_UNBOXED("_sparse_sum.dtype", &ProfiledType::_sparse_sum_dtype);
  m.impl("_sparse_sum.dim", TORCH_FN(ProfiledType::_sparse_sum_dim));
  m.impl_UNBOXED("_sparse_sum.dim_dtype", &ProfiledType::_sparse_sum_dim_dtype);
  m.impl_UNBOXED("_thnn_differentiable_lstm_cell_backward", &ProfiledType::_thnn_differentiable_lstm_cell_backward);
  m.impl_UNBOXED("_thnn_fused_gru_cell", &ProfiledType::_thnn_fused_gru_cell);
  m.impl_UNBOXED("_thnn_fused_lstm_cell_backward", &ProfiledType::_thnn_fused_lstm_cell_backward);
  m.impl("_trilinear", TORCH_FN(ProfiledType::_trilinear));
  m.impl("_unsafe_view", TORCH_FN(ProfiledType::_unsafe_view));
  m.impl("_weight_norm_cuda_interface", TORCH_FN(ProfiledType::_weight_norm_cuda_interface));
  m.impl("abs", TORCH_FN(ProfiledType::abs));
  m.impl_UNBOXED("abs_", &ProfiledType::abs_);
  m.impl_UNBOXED("acosh.out", &ProfiledType::acosh_out_out);
  m.impl("adaptive_avg_pool1d", TORCH_FN(ProfiledType::adaptive_avg_pool1d));
  m.impl_UNBOXED("adaptive_avg_pool2d.out", &ProfiledType::adaptive_avg_pool2d_out_out);
  m.impl("adaptive_max_pool2d_backward", TORCH_FN(ProfiledType::adaptive_max_pool2d_backward));
  m.impl("adaptive_max_pool3d", TORCH_FN(ProfiledType::adaptive_max_pool3d));
  m.impl_UNBOXED("adaptive_max_pool3d_backward.grad_input", &ProfiledType::adaptive_max_pool3d_backward_out_grad_input);
  m.impl("addcmul", TORCH_FN(ProfiledType::addcmul));
  m.impl_UNBOXED("addcmul_", &ProfiledType::addcmul_);
  m.impl("addmm", TORCH_FN(ProfiledType::addmm));
  m.impl_UNBOXED("addmm_", &ProfiledType::addmm_);
  m.impl_UNBOXED("addr.out", &ProfiledType::addr_out_out);
  m.impl("align_as", TORCH_FN(ProfiledType::align_as));
  m.impl("align_tensors", TORCH_FN(ProfiledType::align_tensors));
  m.impl_UNBOXED("align_to", &ProfiledType::align_to);
  m.impl_UNBOXED("align_to.ellipsis_idx", &ProfiledType::align_to_ellipsis_idx);
  m.impl("argmax", TORCH_FN(ProfiledType::argmax));
  m.impl("argsort", TORCH_FN(ProfiledType::argsort));
  m.impl_UNBOXED("argsort.dimname", &ProfiledType::argsort_dimname);
  m.impl_UNBOXED("asinh.out", &ProfiledType::asinh_out_out);
  m.impl("atan", TORCH_FN(ProfiledType::atan));
  m.impl("atan2", TORCH_FN(ProfiledType::atan2));
  m.impl_UNBOXED("atan2_", &ProfiledType::atan2_);
  m.impl_UNBOXED("atan_", &ProfiledType::atan_);
  m.impl("avg_pool2d", TORCH_FN(ProfiledType::avg_pool2d));
  m.impl_UNBOXED("avg_pool2d_backward.grad_input", &ProfiledType::avg_pool2d_backward_out_grad_input);
  m.impl_UNBOXED("avg_pool3d.out", &ProfiledType::avg_pool3d_out_out);
  m.impl_UNBOXED("baddbmm.out", &ProfiledType::baddbmm_out_out);
  m.impl_UNBOXED("bartlett_window", &ProfiledType::bartlett_window);
  m.impl_UNBOXED("bartlett_window.periodic", &ProfiledType::bartlett_window_periodic);
  m.impl_UNBOXED("batch_norm_update_stats", &ProfiledType::batch_norm_update_stats);
  m.impl_UNBOXED("binary_cross_entropy_backward", &ProfiledType::binary_cross_entropy_backward);
  m.impl_UNBOXED("bitwise_and.Tensor_out", &ProfiledType::bitwise_and_out_Tensor_out);
  m.impl_UNBOXED("bitwise_and.Scalar_out", &ProfiledType::bitwise_and_out_Scalar_out);
  m.impl_UNBOXED("bitwise_not.out", &ProfiledType::bitwise_not_out_out);
  m.impl_UNBOXED("bmm.out", &ProfiledType::bmm_out_out);
  m.impl_UNBOXED("bucketize.Tensor_out", &ProfiledType::bucketize_out_Tensor_out);
  m.impl("cdist", TORCH_FN(ProfiledType::cdist));
  m.impl("celu", TORCH_FN(ProfiledType::celu));
  m.impl_UNBOXED("celu_", &ProfiledType::celu_);
  m.impl("cholesky_inverse", TORCH_FN(ProfiledType::cholesky_inverse));
  m.impl_UNBOXED("cholesky.out", &ProfiledType::cholesky_out_out);
  m.impl("clamp", TORCH_FN(ProfiledType::clamp));
  m.impl_UNBOXED("clamp_", &ProfiledType::clamp_);
  m.impl_UNBOXED("clamp_max.out", &ProfiledType::clamp_max_out_out);
  m.impl_UNBOXED("conj.out", &ProfiledType::conj_out_out);
  m.impl_UNBOXED("conv_transpose3d.input", &ProfiledType::conv_transpose3d_input);
  m.impl("convolution_backward_overrideable", TORCH_FN(ProfiledType::convolution_backward_overrideable));
  m.impl("cos", TORCH_FN(ProfiledType::cos));
  m.impl_UNBOXED("cos_", &ProfiledType::cos_);
  m.impl_UNBOXED("cudnn_batch_norm_backward", &ProfiledType::cudnn_batch_norm_backward);
  m.impl("cudnn_grid_sampler_backward", TORCH_FN(ProfiledType::cudnn_grid_sampler_backward));
  m.impl_UNBOXED("cummax.out", &ProfiledType::cummax_out_out);
  m.impl_UNBOXED("cummax.dimname_out", &ProfiledType::cummax_out_dimname_out);
  m.impl_UNBOXED("cumsum.out", &ProfiledType::cumsum_out_out);
  m.impl_UNBOXED("cumsum.dimname_out", &ProfiledType::cumsum_out_dimname_out);
  m.impl("data", TORCH_FN(ProfiledType::data));
  m.impl("deg2rad", TORCH_FN(ProfiledType::deg2rad));
  m.impl_UNBOXED("deg2rad_", &ProfiledType::deg2rad_);
  m.impl("diag", TORCH_FN(ProfiledType::diag));
  m.impl("digamma", TORCH_FN(ProfiledType::digamma));
  m.impl_UNBOXED("digamma_", &ProfiledType::digamma_);
  m.impl_UNBOXED("elu.out", &ProfiledType::elu_out_out);
  m.impl_UNBOXED("embedding_bag", &ProfiledType::embedding_bag);
  m.impl("embedding_dense_backward", TORCH_FN(ProfiledType::embedding_dense_backward));
  m.impl_UNBOXED("empty_like", &ProfiledType::empty_like);
  m.impl_UNBOXED("empty_quantized", &ProfiledType::empty_quantized);
  m.impl_UNBOXED("empty_strided", &ProfiledType::empty_strided);
  m.impl_UNBOXED("erfc.out", &ProfiledType::erfc_out_out);
  m.impl("erfinv", TORCH_FN(ProfiledType::erfinv));
  m.impl_UNBOXED("erfinv_", &ProfiledType::erfinv_);
  m.impl("expand", TORCH_FN(ProfiledType::expand));
  m.impl_UNBOXED("expm1.out", &ProfiledType::expm1_out_out);
  m.impl("fake_quantize_per_tensor_affine_backward", TORCH_FN(ProfiledType::fake_quantize_per_tensor_affine_backward));
  m.impl("fft", TORCH_FN(ProfiledType::fft));
  m.impl("flatten.using_ints", TORCH_FN(ProfiledType::flatten_using_ints));
  m.impl_UNBOXED("flatten.named_out_dim", &ProfiledType::flatten_named_out_dim);
  m.impl_UNBOXED("flatten.using_names", &ProfiledType::flatten_using_names);
  m.impl_UNBOXED("flatten.DimnameList", &ProfiledType::flatten_DimnameList);
  m.impl("floor_divide", TORCH_FN(ProfiledType::floor_divide));
  m.impl("floor_divide.Scalar", TORCH_FN(ProfiledType::floor_divide_Scalar));
  m.impl_UNBOXED("floor_divide_.Tensor", &ProfiledType::floor_divide__Tensor);
  m.impl_UNBOXED("floor_divide_.Scalar", &ProfiledType::floor_divide__Scalar);
  m.impl_UNBOXED("floor.out", &ProfiledType::floor_out_out);
  m.impl_UNBOXED("full.names", &ProfiledType::full_names);
  m.impl_UNBOXED("full", &ProfiledType::full);
  m.impl("gather", TORCH_FN(ProfiledType::gather));
  m.impl_UNBOXED("gather.dimname", &ProfiledType::gather_dimname);
  m.impl("gelu_backward", TORCH_FN(ProfiledType::gelu_backward));
  m.impl("grid_sampler_3d_backward", TORCH_FN(ProfiledType::grid_sampler_3d_backward));
  m.impl_UNBOXED("gru_cell", &ProfiledType::gru_cell);
  m.impl_UNBOXED("hann_window", &ProfiledType::hann_window);
  m.impl_UNBOXED("hann_window.periodic", &ProfiledType::hann_window_periodic);
  m.impl("hardshrink", TORCH_FN(ProfiledType::hardshrink));
  m.impl("ifft", TORCH_FN(ProfiledType::ifft));
  m.impl_UNBOXED("index_select.out", &ProfiledType::index_select_out_out);
  m.impl_UNBOXED("index_select.dimname_out", &ProfiledType::index_select_out_dimname_out);
  m.impl("indices", TORCH_FN(ProfiledType::indices));
  m.impl("is_complex", TORCH_FN(ProfiledType::is_complex));
  m.impl("is_same_size", TORCH_FN(ProfiledType::is_same_size));
  m.impl_UNBOXED("l1_loss.out", &ProfiledType::l1_loss_out_out);
  m.impl_UNBOXED("layer_norm", &ProfiledType::layer_norm);
  m.impl("leaky_relu_backward", TORCH_FN(ProfiledType::leaky_relu_backward));
  m.impl("lerp.Scalar", TORCH_FN(ProfiledType::lerp_Scalar));
  m.impl("lerp.Tensor", TORCH_FN(ProfiledType::lerp_Tensor));
  m.impl_UNBOXED("lerp_.Scalar", &ProfiledType::lerp__Scalar);
  m.impl_UNBOXED("lerp_.Tensor", &ProfiledType::lerp__Tensor);
  m.impl_UNBOXED("linear", &ProfiledType::linear);
  m.impl("log_sigmoid", TORCH_FN(ProfiledType::log_sigmoid));
  m.impl_UNBOXED("log_sigmoid_backward.grad_input", &ProfiledType::log_sigmoid_backward_out_grad_input);
  m.impl_UNBOXED("logcumsumexp.out", &ProfiledType::logcumsumexp_out_out);
  m.impl_UNBOXED("logcumsumexp.dimname_out", &ProfiledType::logcumsumexp_out_dimname_out);
  m.impl_UNBOXED("logical_or.out", &ProfiledType::logical_or_out_out);
  m.impl_UNBOXED("logical_xor.out", &ProfiledType::logical_xor_out_out);
  m.impl_UNBOXED("logspace.out", &ProfiledType::logspace_out_out);
  m.impl_UNBOXED("logsumexp.out", &ProfiledType::logsumexp_out_out);
  m.impl_UNBOXED("logsumexp.names_out", &ProfiledType::logsumexp_out_names_out);
  m.impl_UNBOXED("matmul.out", &ProfiledType::matmul_out_out);
  m.impl_UNBOXED("max.dim_max", &ProfiledType::max_out_dim_max);
  m.impl_UNBOXED("max.names_dim_max", &ProfiledType::max_out_names_dim_max);
  m.impl_UNBOXED("max.out", &ProfiledType::max_out_out);
  m.impl("max_unpool2d", TORCH_FN(ProfiledType::max_unpool2d));
  m.impl_UNBOXED("max_unpool2d_backward.grad_input", &ProfiledType::max_unpool2d_backward_out_grad_input);
  m.impl_UNBOXED("max_unpool3d.out", &ProfiledType::max_unpool3d_out_out);
  m.impl("min_values", TORCH_FN(ProfiledType::min_values));
  m.impl_UNBOXED("min_values.names", &ProfiledType::min_values_names);
  m.impl("miopen_convolution_backward", TORCH_FN(ProfiledType::miopen_convolution_backward));
  m.impl("miopen_convolution_backward_bias", TORCH_FN(ProfiledType::miopen_convolution_backward_bias));
  m.impl("miopen_convolution_backward_input", TORCH_FN(ProfiledType::miopen_convolution_backward_input));
  m.impl_UNBOXED("miopen_convolution_transpose", &ProfiledType::miopen_convolution_transpose);
  m.impl("mkldnn_adaptive_avg_pool2d", TORCH_FN(ProfiledType::mkldnn_adaptive_avg_pool2d));
  m.impl_UNBOXED("mkldnn_convolution", &ProfiledType::mkldnn_convolution);
  m.impl("mkldnn_reorder_conv2d_weight", TORCH_FN(ProfiledType::mkldnn_reorder_conv2d_weight));
  m.impl_UNBOXED("mode.values", &ProfiledType::mode_out_values);
  m.impl_UNBOXED("mode.dimname_out", &ProfiledType::mode_out_dimname_out);
  m.impl("mse_loss_backward", TORCH_FN(ProfiledType::mse_loss_backward));
  m.impl("multilabel_margin_loss_backward", TORCH_FN(ProfiledType::multilabel_margin_loss_backward));
  m.impl_UNBOXED("multilabel_margin_loss_forward.output", &ProfiledType::multilabel_margin_loss_forward_out_output);
  m.impl_UNBOXED("multinomial", &ProfiledType::multinomial);
  m.impl("mvlgamma", TORCH_FN(ProfiledType::mvlgamma));
  m.impl_UNBOXED("mvlgamma_", &ProfiledType::mvlgamma_);
  m.impl("narrow", TORCH_FN(ProfiledType::narrow));
  m.impl("narrow.Tensor", TORCH_FN(ProfiledType::narrow_Tensor));
  m.impl_UNBOXED("native_batch_norm", &ProfiledType::native_batch_norm);
  m.impl_UNBOXED("ne.Scalar_out", &ProfiledType::ne_out_Scalar_out);
  m.impl_UNBOXED("ne.Tensor_out", &ProfiledType::ne_out_Tensor_out);
  m.impl_UNBOXED("new_full", &ProfiledType::new_full);
  m.impl_UNBOXED("nll_loss", &ProfiledType::nll_loss);
  m.impl_UNBOXED("nll_loss2d", &ProfiledType::nll_loss2d);
  m.impl_UNBOXED("nll_loss2d_backward.grad_input", &ProfiledType::nll_loss2d_backward_out_grad_input);
  m.impl_UNBOXED("nll_loss_backward.grad_input", &ProfiledType::nll_loss_backward_out_grad_input);
  m.impl("nuclear_norm", TORCH_FN(ProfiledType::nuclear_norm));
  m.impl("nuclear_norm.dim", TORCH_FN(ProfiledType::nuclear_norm_dim));
  m.impl("orgqr", TORCH_FN(ProfiledType::orgqr));
  m.impl_UNBOXED("ormqr.out", &ProfiledType::ormqr_out_out);
  m.impl("permute", TORCH_FN(ProfiledType::permute));
  m.impl("pixel_shuffle", TORCH_FN(ProfiledType::pixel_shuffle));
  m.impl_UNBOXED("put_", &ProfiledType::put_);
  m.impl("q_zero_point", TORCH_FN(ProfiledType::q_zero_point));
  m.impl_UNBOXED("quantize_per_tensor", &ProfiledType::quantize_per_tensor);
  m.impl_UNBOXED("quantize_per_tensor.tensors", &ProfiledType::quantize_per_tensor_tensors);
  m.impl("quantized_lstm_cell", TORCH_FN(ProfiledType::quantized_lstm_cell));
  m.impl("rad2deg", TORCH_FN(ProfiledType::rad2deg));
  m.impl_UNBOXED("rad2deg_", &ProfiledType::rad2deg_);
  m.impl_UNBOXED("rand.out", &ProfiledType::rand_out_out);
  m.impl_UNBOXED("rand.generator_out", &ProfiledType::rand_out_generator_out);
  m.impl_UNBOXED("randn.out", &ProfiledType::randn_out_out);
  m.impl_UNBOXED("randn.generator_out", &ProfiledType::randn_out_generator_out);
  m.impl_UNBOXED("range.step", &ProfiledType::range_step);
  m.impl_UNBOXED("range", &ProfiledType::range);
  m.impl("real", TORCH_FN(ProfiledType::real));
  m.impl("reciprocal", TORCH_FN(ProfiledType::reciprocal));
  m.impl_UNBOXED("reciprocal_", &ProfiledType::reciprocal_);
  m.impl_UNBOXED("refine_names", &ProfiledType::refine_names);
  m.impl("reflection_pad1d", TORCH_FN(ProfiledType::reflection_pad1d));
  m.impl_UNBOXED("reflection_pad1d_backward.grad_input", &ProfiledType::reflection_pad1d_backward_out_grad_input);
  m.impl_UNBOXED("reflection_pad2d.out", &ProfiledType::reflection_pad2d_out_out);
  m.impl("relu", TORCH_FN(ProfiledType::relu));
  m.impl_UNBOXED("relu_", &ProfiledType::relu_);
  m.impl_UNBOXED("remainder.Scalar_out", &ProfiledType::remainder_out_Scalar_out);
  m.impl_UNBOXED("remainder.Tensor_out", &ProfiledType::remainder_out_Tensor_out);
  m.impl_UNBOXED("replication_pad1d.out", &ProfiledType::replication_pad1d_out_out);
  m.impl_UNBOXED("resize_as_", &ProfiledType::resize_as_);
  m.impl_UNBOXED("rnn_relu_cell", &ProfiledType::rnn_relu_cell);
  m.impl_UNBOXED("rrelu_with_noise", &ProfiledType::rrelu_with_noise);
  m.impl_UNBOXED("rrelu_with_noise_", &ProfiledType::rrelu_with_noise_);
  m.impl("scatter_add", TORCH_FN(ProfiledType::scatter_add));
  m.impl_UNBOXED("scatter_add.dimname", &ProfiledType::scatter_add_dimname);
  m.impl_UNBOXED("scatter_add_", &ProfiledType::scatter_add_);
  m.impl_UNBOXED("select.Dimname", &ProfiledType::select_Dimname);
  m.impl("select.int", TORCH_FN(ProfiledType::select_int));
  m.impl("sin", TORCH_FN(ProfiledType::sin));
  m.impl_UNBOXED("sin_", &ProfiledType::sin_);
  m.impl("slow_conv_dilated3d_backward", TORCH_FN(ProfiledType::slow_conv_dilated3d_backward));
  m.impl_UNBOXED("soft_margin_loss.out", &ProfiledType::soft_margin_loss_out_out);
  m.impl_UNBOXED("softmax.int", &ProfiledType::softmax_int);
  m.impl_UNBOXED("softmax.Dimname", &ProfiledType::softmax_Dimname);
  m.impl_UNBOXED("softplus.out", &ProfiledType::softplus_out_out);
  m.impl("softshrink_backward", TORCH_FN(ProfiledType::softshrink_backward));
  m.impl_UNBOXED("sort.values", &ProfiledType::sort_out_values);
  m.impl_UNBOXED("sort.dimname_values", &ProfiledType::sort_out_dimname_values);
  m.impl("squeeze", TORCH_FN(ProfiledType::squeeze));
  m.impl("squeeze.dim", TORCH_FN(ProfiledType::squeeze_dim));
  m.impl_UNBOXED("squeeze.dimname", &ProfiledType::squeeze_dimname);
  m.impl_UNBOXED("squeeze_", &ProfiledType::squeeze_);
  m.impl_UNBOXED("squeeze_.dim", &ProfiledType::squeeze__dim);
  m.impl_UNBOXED("squeeze_.dimname", &ProfiledType::squeeze__dimname);
  m.impl_UNBOXED("std.out", &ProfiledType::std_out_out);
  m.impl_UNBOXED("std.names_out", &ProfiledType::std_out_names_out);
  m.impl("sub.Tensor", TORCH_FN(ProfiledType::sub_Tensor));
  m.impl("sub.Scalar", TORCH_FN(ProfiledType::sub_Scalar));
  m.impl_UNBOXED("sub_.Tensor", &ProfiledType::sub__Tensor);
  m.impl_UNBOXED("sub_.Scalar", &ProfiledType::sub__Scalar);
  m.impl_UNBOXED("sum.IntList_out", &ProfiledType::sum_out_IntList_out);
  m.impl_UNBOXED("sum.DimnameList_out", &ProfiledType::sum_out_DimnameList_out);
  m.impl_UNBOXED("take.out", &ProfiledType::take_out_out);
  m.impl_UNBOXED("thnn_conv2d_forward", &ProfiledType::thnn_conv2d_forward);
  m.impl_UNBOXED("thnn_conv_depthwise2d.out", &ProfiledType::thnn_conv_depthwise2d_out_out);
  m.impl("to_dense_backward", TORCH_FN(ProfiledType::to_dense_backward));
  m.impl_UNBOXED("topk.values", &ProfiledType::topk_out_values);
  m.impl_UNBOXED("trunc.out", &ProfiledType::trunc_out_out);
  m.impl("unbind.int", TORCH_FN(ProfiledType::unbind_int));
  m.impl_UNBOXED("unbind.Dimname", &ProfiledType::unbind_Dimname);
  m.impl("unique_consecutive", TORCH_FN(ProfiledType::unique_consecutive));
  m.impl("upsample_bilinear2d_backward", TORCH_FN(ProfiledType::upsample_bilinear2d_backward));
  m.impl("upsample_linear1d", TORCH_FN(ProfiledType::upsample_linear1d));
  m.impl_UNBOXED("upsample_linear1d_backward.grad_input", &ProfiledType::upsample_linear1d_backward_out_grad_input);
  m.impl("upsample_nearest1d_backward", TORCH_FN(ProfiledType::upsample_nearest1d_backward));
  m.impl("upsample_nearest2d", TORCH_FN(ProfiledType::upsample_nearest2d));
  m.impl_UNBOXED("upsample_nearest2d_backward.grad_input", &ProfiledType::upsample_nearest2d_backward_out_grad_input);
  m.impl_UNBOXED("upsample_nearest3d.out", &ProfiledType::upsample_nearest3d_out_out);
  m.impl("vander", TORCH_FN(ProfiledType::vander));
  m.impl("view_as", TORCH_FN(ProfiledType::view_as));
  m.impl("view_as_complex", TORCH_FN(ProfiledType::view_as_complex));
  m.impl("view_as_real", TORCH_FN(ProfiledType::view_as_real));;
}

}  // namespace

} // namespace torch
